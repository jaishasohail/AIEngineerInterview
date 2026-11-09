from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path

from langgraph.types import Command

from app.ai.context import (
    CTX,
    RequestCtx,
    leave_history_record_context,
    load_actor,
    pending_defer_message,
    pending_draft_cancellation_message,
    pending_hitl_clarification_message,
    pending_explicit_confirmation_message,
    workforce_corrective_clarification_message,
)
from app.ai.graph import build_graph
from app.ai.memory import (
    history_for_prompt,
    inherit_workflow_state,
    recent_messages,
    record_context_from_history,
    resolve_followup,
)
from app.ai.narrate import narrate, result_message
from app.clock import utcnow
from app.config import get_settings
from app.models import Conversation, Message, PendingAction, User
from app.services.confirmation import (
    already_executed_response,
    cancel_pending,
    execute_pending,
    expire_if_needed,
    get_pending,
    public_pending,
)

_graph = None
_connection = None

CONFIRM = re.compile(r"^(yes|confirm|submit|go ahead|please submit|file it)\b", re.IGNORECASE)
CANCEL = re.compile(
    r"^(no|nope|cancel( that| this| it| the prepared (?:leave )?request|"
    r"the prepared (?:approval|rejection|action)| my pending (?:leave )?request)?|stop|"
    r"never\s*mind|i\s+changed\s+my\s+mind|don'?t\s+submit(\s+it)?|do\s+not\s+submit(\s+it)?|forget\s+it)\b[.!]?\s*$",
    re.IGNORECASE,
)


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


def checkpoint_store_backend() -> str:
    return "postgres" if _use_postgres_checkpoint_store() else "sqlite"


def _use_postgres_checkpoint_store() -> bool:
    settings = get_settings()
    store = (settings.checkpoint_store or "auto").strip().lower()
    if store == "sqlite":
        return False
    if store == "postgres":
        return settings.database_url.startswith("postgresql")
    if settings.database_url.startswith("sqlite"):
        return False
    return False


def get_graph():
    global _graph, _connection
    if _graph is None:
        settings = get_settings()
        database_url = settings.database_url
        if _use_postgres_checkpoint_store():
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

            path = Path(settings.checkpoint_path)
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
        "record_context": values.get("record_context"),
        "execution_result": values.get("execution_result"),
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
    if CANCEL.match(text) or pending_draft_cancellation_message(text):
        return {"confirmed": False}
    if CONFIRM.match(text) or pending_explicit_confirmation_message(text):
        option = "alternative" if re.search(r"alternative|recommend|lower|clearer|january", text, re.IGNORECASE) else "primary"
        return {"confirmed": True, "option_id": option}
    return None


def _cancel_pending_via_chat(db, user: User, conversation: Conversation, *, ip: str | None) -> dict | None:
    """Cancel a durable pending action when the user declines in chat (no active interrupt)."""
    row = get_pending(db, conversation.pending_action_id)
    if row is None or row.status != "awaiting_confirmation":
        row = (
            db.query(PendingAction)
            .filter(
                PendingAction.conversation_id == conversation.id,
                PendingAction.owner_user_id == user.id,
                PendingAction.status == "awaiting_confirmation",
            )
            .order_by(PendingAction.id.desc())
            .first()
        )
    if row is None:
        return None
    if row is None or row.status != "awaiting_confirmation":
        return None
    result = cancel_pending(db, row, user=user, ip=ip)
    conversation.pending_action_id = None
    conversation.active_thread_id = None
    conversation.status = "active"
    conversation.updated_at = utcnow()
    return _execution_payload(conversation.id, result)


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
    history_ctx = record_context_from_history(history)
    merged_context = {**(inherited.get("record_context") or {}), **history_ctx}
    if merged_context:
        inherited = {**inherited, "record_context": merged_context}
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
    try:
        if conversation.user_id != user.id:
            raise PermissionError("Conversation does not belong to this user.")
        if pending_draft_cancellation_message(message):
            cancelled = _cancel_pending_via_chat(db, user, conversation, ip=ip)
            if cancelled is not None:
                return cancelled
        if pending_explicit_confirmation_message(message) and conversation.pending_action_id:
            row = get_pending(db, conversation.pending_action_id)
            if row is not None:
                row = expire_if_needed(db, row, ip=ip)
                if row.status == "cancelled" and not conversation.active_thread_id:
                    result = execute_pending(db, user, row, ip=ip)
                    return _execution_payload(conversation.id, result, intent="workforce_analytics")
        if conversation.active_thread_id:
            existing = _config(conversation.active_thread_id)
            snapshot = graph.get_state(existing)
            if snapshot.next:
                resume = _resume_command(message)
                if resume is None and pending_draft_cancellation_message(message):
                    resume = {"confirmed": False}
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
                        row = get_pending(db, conversation.pending_action_id)
                        pending_payload = public_pending(row) if row is not None else {}
                        if row is not None and (
                            workforce_corrective_clarification_message(message)
                            or pending_hitl_clarification_message(message)
                        ):
                            values = dict(snapshot.values or {})
                            if pending_defer_message(message):
                                text = (
                                    "The prepared action is still waiting. Nothing was submitted or executed. "
                                    "Reply Confirm to proceed, or cancel / never mind to discard it."
                                )
                                summary = "Prepared action not confirmed yet."
                            elif workforce_corrective_clarification_message(message):
                                focus = (
                                    "proposed_action"
                                    if "proposed" in message.lower()
                                    else "evidence"
                                )
                                stub = {
                                    "intent": "workforce_analytics",
                                    "message": message,
                                    "pending_action": pending_payload,
                                    "analysis": {
                                        "focus": focus,
                                        "proposal": (row.payload_json or {}).get("proposal"),
                                    },
                                    "analytics": values.get("analytics") or {},
                                    "role": user.role,
                                }
                                text, summary = narrate(stub)
                            else:
                                stub = {
                                    **values,
                                    "message": message,
                                    "pending_action": pending_payload,
                                    "role": user.role,
                                }
                                text, summary = narrate(stub)
                                if not text:
                                    text = values.get("response") or (
                                        "Prepared action details are unchanged. "
                                        "Confirm to proceed or cancel to discard without executing."
                                    )
                                    summary = values.get("decision_summary") or text[:120]
                            clarify = _public(
                                conversation.id,
                                {**values, "response": text, "decision_summary": summary},
                                _interrupt_value({}, snapshot),
                            )
                            return {**clarify, "message": text, "decision_summary": summary, "pending_action": pending_payload}
                        raise ValueError(
                            "There is a prepared action waiting for confirmation. "
                            "Reply yes to confirm, or cancel / never mind to discard it without submitting."
                        )
                    graph.invoke(Command(resume=resume), existing)
                    snapshot = graph.get_state(existing)
                    interrupt_value = _interrupt_value({}, snapshot)
                    conversation.active_thread_id = conversation.active_thread_id if interrupt_value else None
                    conversation.status = "awaiting_confirmation" if interrupt_value else "active"
                    exec_result = (snapshot.values or {}).get("execution_result") or {}
                    if not interrupt_value and exec_result.get("status") not in {
                        "submitted",
                        "approved",
                        "rejected",
                        "pending_hr",
                        "noop",
                        "duplicate",
                        "recorded",
                        "cancelled",
                    }:
                        conversation.pending_action_id = None
                    conversation.updated_at = utcnow()
                    values = dict(snapshot.values or {})
                    if not values.get("response") and exec_result:
                        text, summary = result_message(exec_result)
                        values = {**values, "response": text, "decision_summary": summary}
                    return _public(conversation.id, values, interrupt_value)
                finally:
                    CTX.reset(token)

        if pending_draft_cancellation_message(message):
            cancelled = _cancel_pending_via_chat(db, user, conversation, ip=ip)
            if cancelled is not None:
                return cancelled
            return {
                "conversation_id": conversation.id,
                "message": "No prepared action is waiting to cancel. Nothing was submitted.",
                "intent": "general_hr",
                "decision_summary": "No pending draft to cancel.",
                "confidence": "n/a",
                "citations": [],
                "tool_trace": [],
                "workflow": [],
                "authorization": {"allowed": True, "reason": ""},
                "pending_action": None,
            }

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
                exec_result = (snapshot.values or {}).get("execution_result") or {}
                if exec_result.get("status") not in {
                    "submitted",
                    "approved",
                    "rejected",
                    "pending_hr",
                    "noop",
                    "duplicate",
                    "recorded",
                    "cancelled",
                }:
                    conversation.pending_action_id = None
            conversation.updated_at = utcnow()
            return _public(conversation.id, snapshot.values, interrupt_value)
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


def _execution_payload(
    conversation_id: int,
    result: dict,
    *,
    record_context: dict | None = None,
    intent: str | None = None,
) -> dict:
    text, summary = result_message(result)
    execution = result.get("result") if result.get("status") == "duplicate" else result
    request_id = (execution or {}).get("leave_request_id")
    payload = {
        "conversation_id": conversation_id,
        "message": text,
        "intent": intent or ("leave_request" if request_id else None),
        "decision_summary": summary,
        "confidence": "n/a",
        "citations": [],
        "tool_trace": [
            {
                "tool": "execute_authorized_action",
                "status": "duplicate" if result.get("status") == "duplicate" else "ok",
                "summary": summary,
            }
        ],
        "workflow": [
            {
                "node": "execute_workflow",
                "label": "Workflow execution",
                "status": result.get("status", "error"),
                "detail": summary[:120],
            }
        ],
        "pending_action": None,
        "execution_result": result,
        "authorization": {"allowed": True, "reason": ""},
    }
    if record_context:
        payload["record_context"] = record_context
    return payload


def _submission_record_context(db, user: User, result: dict) -> tuple[dict | None, str | None]:
    request_id = result.get("leave_request_id") or (result.get("result") or {}).get("leave_request_id")
    status = str(result.get("status") or "")
    if not request_id or status not in {"submitted", "pending_manager", "pending_hr", "approved", "rejected"}:
        return None, None
    _user, employee, _department = load_actor(db, user.id)
    return (
        leave_history_record_context(
            user_id=user.id,
            employee_id=employee.id,
            request_id=int(request_id),
            report_filter="latest",
        ),
        "leave_request_status",
    )


def confirm_turn(db, user: User, conversation: Conversation, *, confirmed: bool, option_id: str, ip: str | None = None) -> dict:
    if conversation.user_id != user.id:
        raise PermissionError("Conversation does not belong to this user.")
    if not conversation.active_thread_id:
        row = get_pending(db, conversation.pending_action_id)
        if row is not None:
            row = expire_if_needed(db, row, ip=ip)
            if not confirmed:
                result = cancel_pending(db, row, user=user, ip=ip)
                conversation.pending_action_id = None
                conversation.active_thread_id = None
                conversation.status = "active"
                return _execution_payload(conversation.id, result)
            if row.status == "cancelled":
                return _execution_payload(
                    conversation.id,
                    execute_pending(db, user, row, option_id=option_id or "primary", ip=ip),
                )
            if row.status == "executed" and row.result_json:
                return _execution_payload(conversation.id, already_executed_response(row))
            if row.status == "awaiting_confirmation":
                result = execute_pending(db, user, row, option_id=option_id or "primary", ip=ip)
                ctx, submitted_intent = _submission_record_context(db, user, result)
                return _execution_payload(
                    conversation.id, result, record_context=ctx, intent=submitted_intent
                )
            if row.status == "executed":
                return _execution_payload(conversation.id, already_executed_response(row))
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
