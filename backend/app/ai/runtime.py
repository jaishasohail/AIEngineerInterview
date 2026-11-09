from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path

from langgraph.types import Command

from app.ai.context import CTX, RequestCtx, load_actor
from app.ai.graph import build_graph
from app.ai.memory import (
    history_for_prompt,
    inherit_workflow_state,
    recent_messages,
    resolve_followup,
)
from app.clock import utcnow
from app.config import get_settings
from app.models import Conversation, Message, User

_graph = None
_connection = None

CONFIRM = re.compile(r"^(yes|confirm|submit|go ahead|please submit|file it)\b", re.IGNORECASE)
CANCEL = re.compile(r"^(no|nope|cancel|stop|don'?t submit|do not submit)\b[.!]?\s*$", re.IGNORECASE)


def reset_graph() -> None:
    global _graph, _connection
    _graph = None
    if _connection is not None:
        _connection.close()
        _connection = None


def _checkpoint_conninfo(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return "postgresql://" + database_url[len("postgresql+psycopg://") :]
    return database_url


def get_graph():
    global _graph, _connection
    if _graph is None:
        database_url = get_settings().database_url
        if database_url.startswith("postgresql"):
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg.rows import dict_row

            import psycopg

            _connection = psycopg.connect(
                _checkpoint_conninfo(database_url),
                autocommit=True,
                prepare_threshold=0,
                row_factory=dict_row,
            )
            saver = PostgresSaver(_connection)
            saver.setup()
        else:
            from langgraph.checkpoint.sqlite import SqliteSaver

            path = Path(get_settings().checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            _connection = sqlite3.connect(str(path), check_same_thread=False)
            saver = SqliteSaver(_connection)
            saver.setup()
        _graph = build_graph().compile(checkpointer=saver)
    return _graph


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}


def _interrupt_value(result: dict, snapshot) -> dict | None:
    raw = None
    if isinstance(result, dict) and result.get("__interrupt__"):
        raw = result["__interrupt__"][0]
    else:
        interrupts = getattr(snapshot, "interrupts", None) or []
        if interrupts:
            raw = interrupts[0]
    if raw is None:
        return None
    value = getattr(raw, "value", raw)
    return value if isinstance(value, dict) else {"value": value}


def _public(conversation_id: int, values: dict, interrupt_value: dict | None) -> dict:
    pending = None
    if interrupt_value and not values.get("execution_result"):
        pending = interrupt_value.get("pending_action")
    confidence = values.get("retrieval_confidence") or "n/a"
    if values.get("authorized") is False:
        confidence = "denied"
    elif values.get("clarify"):
        confidence = "weak"
    elif (values.get("retrieval") or {}).get("confidence"):
        confidence = values["retrieval"]["confidence"]
    memory = values.get("memory") or {}
    return {
        "conversation_id": conversation_id,
        "message": values.get("response") or "",
        "intent": values.get("intent"),
        "decision_summary": values.get("decision_summary") or "",
        "confidence": confidence,
        "citations": values.get("citations") or [],
        "tool_trace": values.get("tool_trace") or [],
        "workflow": values.get("workflow") or [],
        "pending_action": pending,
        "analysis": values.get("analysis"),
        "analytics": values.get("analytics"),
        "orchestration_mode": values.get("orchestration_mode") or "deterministic",
        "authorization": {
            "allowed": values.get("authorized", True),
            "reason": values.get("denial_reason") or "",
        },
        "memory": {
            "is_followup": bool(memory.get("is_followup")),
            "resolver": memory.get("resolver"),
            "rewritten_message": memory.get("rewritten_message"),
            "raw_message": values.get("raw_message"),
        }
        if memory
        else None,
    }


def _resume_command(message: str) -> dict | None:
    text = message.strip()
    if CANCEL.match(text):
        return {"confirmed": False}
    if CONFIRM.match(text):
        option = "alternative" if re.search(r"alternative|recommend|lower|clearer|january", text, re.IGNORECASE) else "primary"
        return {"confirmed": True, "option_id": option}
    return None


def _prepare_turn_state(db, user: User, conversation: Conversation, message: str, prior_values: dict | None) -> dict:
    _, employee, _department = load_actor(db, user.id)
    history = recent_messages(
        db,
        conversation_id=conversation.id,
        user_id=user.id,
        limit=12,
        exclude_trailing_user=message,
    )
    inherited = inherit_workflow_state(prior_values, user_id=user.id, employee_id=employee.id)
    inherited_for_memory = {
        **(inherited.get("record_context") or {}),
        "intent": inherited.get("prior_intent"),
        "decision_summary": inherited.get("prior_decision_summary"),
    }
    # Drop null-only placeholders so resolve_followup does not treat empty inheritance as context.
    inherited_for_memory = {
        key: value for key, value in inherited_for_memory.items() if value not in (None, "", [], {})
    }
    memory = resolve_followup(
        message=message,
        role=user.role,
        history=history,
        inherited=inherited_for_memory,
        prior_intent=inherited.get("prior_intent"),
    )
    rewritten = memory.get("rewritten_message") or message
    return {
        "message": rewritten,
        "raw_message": message,
        "role": user.role,
        "tool_trace": [],
        "workflow": [],
        "pending_action": None,
        "citations": [],
        "analysis": inherited.get("analysis"),
        "analytics": inherited.get("analytics"),
        "retrieval": None,
        "record_context": inherited.get("record_context"),
        "pending_clarification": inherited.get("pending_clarification"),
        "conversation_history": history_for_prompt(history),
        "memory": memory,
    }


def run_turn(db, user: User, conversation: Conversation, message: str, *, ip: str | None = None) -> dict:
    graph = get_graph()
    discarded = False
    try:
        if conversation.user_id != user.id:
            raise PermissionError("Conversation does not belong to this user.")
        if conversation.active_thread_id:
            existing = _config(conversation.active_thread_id)
            snapshot = graph.get_state(existing)
            if snapshot.next:
                resume = _resume_command(message)
                token = CTX.set(
                    RequestCtx(
                        db=db,
                        user_id=user.id,
                        ip=ip,
                        conversation_id=conversation.id,
                        thread_id=conversation.active_thread_id,
                    )
                )
                try:
                    if resume is None:
                        # Safe interrupt discard: cancel the pending write, then continue as a new turn
                        # with conversation history intact.
                        graph.invoke(Command(resume={"confirmed": False}), existing)
                        conversation.active_thread_id = None
                        conversation.status = "active"
                        conversation.pending_action_id = None
                        discarded = True
                    else:
                        result = graph.invoke(Command(resume=resume), existing)
                        snapshot = graph.get_state(existing)
                        interrupt_value = _interrupt_value(result, snapshot)
                        conversation.active_thread_id = conversation.active_thread_id if interrupt_value else None
                        conversation.status = "awaiting_confirmation" if interrupt_value else "active"
                        if not interrupt_value:
                            conversation.pending_action_id = None
                        conversation.updated_at = utcnow()
                        payload = _public(conversation.id, snapshot.values, interrupt_value)
                        return payload
                finally:
                    CTX.reset(token)

        prior_values = None
        if conversation.active_thread_id:
            prior = graph.get_state(_config(conversation.active_thread_id))
            if prior is not None:
                prior_values = prior.values or {}

        turn_state = _prepare_turn_state(db, user, conversation, message, prior_values)
        thread_id = f"u{user.id}-c{conversation.id}-{uuid.uuid4().hex[:12]}"
        token = CTX.set(
            RequestCtx(
                db=db,
                user_id=user.id,
                ip=ip,
                conversation_id=conversation.id,
                thread_id=thread_id,
            )
        )
        try:
            result = graph.invoke(turn_state, _config(thread_id))
            snapshot = graph.get_state(_config(thread_id))
            interrupt_value = _interrupt_value(result, snapshot)
            conversation.active_thread_id = thread_id
            conversation.status = "awaiting_confirmation" if interrupt_value else "active"
            pending = (interrupt_value or {}).get("pending_action") or (snapshot.values or {}).get("pending_action")
            if interrupt_value and isinstance(pending, dict) and pending.get("pending_action_id"):
                conversation.pending_action_id = int(pending["pending_action_id"])
            elif not interrupt_value:
                conversation.pending_action_id = None
            conversation.updated_at = utcnow()
            payload = _public(conversation.id, snapshot.values, interrupt_value)
            if discarded:
                payload["message"] = "The previous draft was discarded without changing any records.\n\n" + payload["message"]
            return payload
        finally:
            CTX.reset(token)
    except Exception:
        # Ensure context is cleared if setup failed before the inner try.
        raise


def stream_turn(db, user: User, conversation: Conversation, message: str, *, ip: str | None = None) -> Iterator[dict]:
    graph = get_graph()
    if conversation.active_thread_id and graph.get_state(_config(conversation.active_thread_id)).next:
        yield {"type": "status", "detail": "Resolving the previous confirmation"}
    yield {"type": "status", "detail": "Running the HR workflow"}
    payload = run_turn(db, user, conversation, message, ip=ip)
    for step in payload.get("workflow") or []:
        yield {"type": "workflow", "step": step}
    for tool in payload.get("tool_trace") or []:
        yield {"type": "tool", "tool": tool}
    yield {"type": "final", "payload": payload}


def confirm_turn(db, user: User, conversation: Conversation, *, confirmed: bool, option_id: str, ip: str | None = None) -> dict:
    if conversation.user_id != user.id:
        raise PermissionError("Conversation does not belong to this user.")
    if not conversation.active_thread_id:
        raise ValueError("There is no action waiting for confirmation.")
    phrase = "yes, submit"
    if confirmed and option_id == "alternative":
        phrase = "yes, submit the alternative"
    elif not confirmed:
        phrase = "cancel"
    return run_turn(db, user, conversation, phrase, ip=ip)


def add_message(db, conversation: Conversation, role: str, content: str, payload: dict | None = None) -> Message:
    row = Message(
        conversation_id=conversation.id,
        role=role,
        content=content,
        payload_json=payload,
        created_at=utcnow(),
    )
    db.add(row)
    if conversation.title == "New conversation" and role == "user":
        conversation.title = content[:80]
    conversation.updated_at = utcnow()
    db.flush()
    return row


def json_safe(payload: dict) -> dict:
    return json.loads(json.dumps(payload, default=str))
