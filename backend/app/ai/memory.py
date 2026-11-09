"""Persistent multi-turn conversation memory.

Message history is stored per Conversation and scoped to the owning user.
Follow-up resolution uses the LLM when available, with deterministic cues as a
fallback. Structured workflow context from the prior turn is preserved separately
from LangGraph interrupt threads.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy.orm import Session

from app.ai.llm import LLMClient, PermanentLLMError, TransientLLMError
from app.models import Conversation, Message

logger = logging.getLogger("helios.memory")

FOLLOWUP_CUE = re.compile(
    r"(?:"
    r"^\s*(and|also|then|why)\b|"
    r"\bwhat if\b|"
    r"\b(what|how) about\b|"
    r"\bonly one week\b|"
    r"\b(starting|from)\s+(?:\d{1,2}|january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b|"
    r"\bfor\s+(?:one|two|three|four|five|six|seven|\d+)\s+(?:working\s+)?days?\b|"
    r"\bmake it\b|"
    r"\b(review|prepare)\s+the\b|"
    r"\b(it|that|those|these|them|this|the first(?: one)?|the second(?: one)?|"
    r"the previous(?: one|request)?|the same|next month|that month|that department|"
    r"that request|the previous request|instead|recommend|these dates)\b"
    r")",
    re.IGNORECASE,
)

ALLOWED_INTENT_HINTS = {
    "leave_planning",
    "leave_request",
    "leave_eligibility",
    "leave_policy",
    "leave_request_status",
    "leave_cancellation",
    "leave_balance",
    "team_conflict",
    "team_roster",
    "team_analytics",
    "workforce_analytics",
    "org_transfer",
    "policy_question",
    "employee_profile",
    "attendance_query",
    "holiday_query",
    "employee_directory",
    "org_attendance",
    "org_leave",
    "compensation_query",
    "hr_metrics",
    "approval_decision",
    "general_hr",
}


def looks_like_followup(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    if len(text.split()) <= 8 and FOLLOWUP_CUE.search(text):
        return True
    return bool(FOLLOWUP_CUE.search(text)) and len(text) < 160


def recent_messages(
    db: Session,
    *,
    conversation_id: int,
    user_id: int,
    limit: int = 12,
    exclude_trailing_user: str | None = None,
) -> list[dict[str, Any]]:
    """Load recent messages for a conversation owned by user_id only."""
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user_id:
        return []
    rows = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )
    if exclude_trailing_user and rows:
        last = rows[-1]
        if last.role == "user" and (last.content or "").strip() == exclude_trailing_user.strip():
            rows = rows[:-1]
    trimmed = rows[-limit:]
    history: list[dict[str, Any]] = []
    for row in trimmed:
        item: dict[str, Any] = {
            "role": row.role,
            "content": (row.content or "")[:1200],
        }
        payload = row.payload_json or {}
        if row.role == "assistant" and isinstance(payload, dict):
            compact = {
                key: payload.get(key)
                for key in ("intent", "decision_summary", "confidence", "orchestration_mode")
                if payload.get(key) not in (None, "", [], {})
            }
            record_context = payload.get("record_context")
            if isinstance(record_context, dict) and record_context.get("subject"):
                compact["record_context"] = {
                    key: record_context.get(key)
                    for key in (
                        "subject",
                        "intent",
                        "leave_type",
                        "start",
                        "end",
                        "label",
                        "business_days",
                        "workflow_state",
                        "focus",
                        "filter",
                        "request_id",
                        "user_id",
                        "employee_id",
                    )
                    if record_context.get(key) not in (None, "", [], {})
                }
            execution_result = payload.get("execution_result")
            if isinstance(execution_result, dict) and execution_result.get("leave_request_id"):
                compact["submitted_request_id"] = execution_result.get("leave_request_id")
            analysis = payload.get("analysis")
            if isinstance(analysis, dict):
                compact["analysis_kind"] = analysis.get("kind") or analysis.get("focus")
                if analysis.get("start") and analysis.get("end"):
                    compact["window"] = {"start": analysis.get("start"), "end": analysis.get("end")}
                if analysis.get("remaining") is not None:
                    compact["remaining"] = analysis.get("remaining")
                requested = analysis.get("requested_window") or {}
                if requested.get("start") and not compact.get("record_context"):
                    compact["window"] = {"start": requested.get("start"), "end": requested.get("end")}
            analytics = payload.get("analytics")
            if isinstance(analytics, dict):
                compact["analytics_focus"] = analytics.get("focus")
                compact["analytics_months"] = analytics.get("months")
            citations = payload.get("citations") or []
            if citations:
                compact["cited_policies"] = [
                    {"policy_id": c.get("policy_id"), "title": c.get("title")}
                    for c in citations[:3]
                    if isinstance(c, dict)
                ]
            if compact:
                item["state"] = compact
        history.append(item)
    return history


def workflow_memory_from_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the latest assistant workflow hints from stored message payloads."""
    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        state = item.get("state") or {}
        if not state:
            continue
        return {
            "prior_intent": state.get("intent"),
            "decision_summary": state.get("decision_summary"),
            "analysis_kind": state.get("analysis_kind"),
            "window": state.get("window"),
            "remaining": state.get("remaining"),
            "analytics_focus": state.get("analytics_focus"),
            "analytics_months": state.get("analytics_months"),
            "cited_policies": state.get("cited_policies") or [],
        }
    return {}


def record_context_from_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        state = item.get("state") or {}
        submitted_id = state.get("submitted_request_id")
        ctx = state.get("record_context") if isinstance(state.get("record_context"), dict) else {}
        if submitted_id:
            merged = dict(ctx)
            if merged.get("subject") != "leave_history":
                merged.update(
                    {
                        "subject": "leave_history",
                        "intent": "leave_request_status",
                        "request_id": int(submitted_id),
                        "filter": "latest",
                        "focus": "latest",
                        "workflow_state": "SUBMITTED",
                    }
                )
            else:
                merged.setdefault("request_id", int(submitted_id))
            if merged.get("subject"):
                return merged
        if isinstance(ctx, dict) and ctx.get("subject"):
            return ctx
    return {}


def inherit_workflow_state(prior_values: dict | None, *, user_id: int, employee_id: int) -> dict[str, Any]:
    """Copy safe structured state from the previous checkpoint values."""
    values = prior_values or {}
    context = values.get("record_context") or {}
    same_actor = context.get("user_id") in {None, user_id} and context.get("employee_id") in {None, employee_id}
    if context and not same_actor:
        context = {}
    analysis = values.get("analysis") if same_actor else None
    analytics = values.get("analytics") if same_actor else None
    return {
        "record_context": context or None,
        "pending_clarification": values.get("pending_clarification") if same_actor else None,
        "analysis": analysis if isinstance(analysis, dict) else None,
        "analytics": analytics if isinstance(analytics, dict) else None,
        "prior_intent": values.get("intent") if same_actor else None,
        "prior_citations": (values.get("citations") or [])[:4] if same_actor else [],
        "prior_decision_summary": values.get("decision_summary") if same_actor else None,
    }


def _meaningful_context(data: dict[str, Any] | None) -> bool:
    if not data:
        return False
    return any(
        data.get(key) not in (None, "", [], {})
        for key in (
            "intent",
            "decision_summary",
            "subject",
            "leave_focus",
            "prior_intent",
            "window_start",
            "window_end",
            "analytics_focus",
            "analysis_kind",
        )
    )


def resolve_followup(
    *,
    message: str,
    role: str,
    history: list[dict[str, Any]],
    inherited: dict[str, Any] | None,
    prior_intent: str | None = None,
) -> dict[str, Any]:
    """Rewrite a follow-up into a standalone request using LLM when possible."""
    inherited = inherited or {}
    workflow = workflow_memory_from_history(history)
    if prior_intent and not workflow.get("prior_intent"):
        workflow["prior_intent"] = prior_intent
    base = {
        "is_followup": bool(history) and looks_like_followup(message),
        "rewritten_message": message,
        "intent_hint": None,
        "leave_focus": None,
        "references": {},
        "workflow": workflow,
        "resolver": "none",
    }
    if not history:
        return base
    has_context = _meaningful_context(inherited) or _meaningful_context(workflow)
    if not base["is_followup"] and not has_context:
        return base

    client = LLMClient()
    if client.enabled:
        try:
            parsed = client.complete_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You resolve multi-turn HR chatbot follow-ups. "
                            "Given conversation history and the latest user message, return JSON with keys: "
                            "is_followup (boolean), "
                            "rewritten_message (standalone question preserving user intent), "
                            "intent_hint (one of: "
                            + "|".join(sorted(ALLOWED_INTENT_HINTS))
                            + " or null), "
                            "leave_focus (string|null), "
                            "references (object of resolved entities like month, department, request, duration_days). "
                            "Never invent private employee facts. "
                            "If the message is not a follow-up, set is_followup=false and rewritten_message to the original. "
                            "Do not mark unrelated topics as the previous intent."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Role: {role}\n"
                            f"Inherited workflow context: {inherited}\n"
                            f"Workflow memory: {workflow}\n"
                            f"History: {history[-8:]}\n"
                            f"Latest message: {message}"
                        ),
                    },
                ],
                temperature=0,
                max_tokens=350,
                raise_on_error=True,
            )
            rewritten = str(parsed.get("rewritten_message") or message).strip() or message
            hint = parsed.get("intent_hint")
            if hint not in ALLOWED_INTENT_HINTS:
                hint = None
            is_followup = bool(parsed.get("is_followup", base["is_followup"]))
            if base["is_followup"] and has_context:
                # Cue-detected follow-ups keep domain continuity even if the model under-flags them.
                is_followup = True
                if hint is None:
                    hint = workflow.get("prior_intent") or inherited.get("intent") or prior_intent
                    if hint not in ALLOWED_INTENT_HINTS:
                        hint = None
            # Do not let the rewriter turn approval-inbox follow-ups into policy/transfer asks.
            from app.ai.context import approval_inbox_followup_focus

            prior_subject = inherited.get("subject")
            follow_focus = approval_inbox_followup_focus(message)
            if prior_subject in {"approval_inbox", "approval_decision"} or str(inherited.get("focus") or "").startswith(
                "awaiting_approval"
            ):
                if follow_focus is not None:
                    rewritten = message
                    if follow_focus == "prepare_approve":
                        hint = "approval_decision"
                    else:
                        hint = "team_analytics"
            return {
                "is_followup": is_followup,
                "rewritten_message": rewritten,
                "intent_hint": hint,
                "leave_focus": (str(parsed.get("leave_focus")).strip() if parsed.get("leave_focus") else None),
                "references": parsed.get("references") if isinstance(parsed.get("references"), dict) else {},
                "workflow": workflow,
                "resolver": "llm",
            }
        except (TransientLLMError, PermanentLLMError) as exc:
            logger.warning("Follow-up LLM resolution unavailable: %s", exc.__class__.__name__)

    if not base["is_followup"]:
        return base

    # Deterministic fallback for common anaphora when the model is unavailable.
    rewritten = message
    hint = workflow.get("prior_intent") or inherited.get("intent") or prior_intent
    refs: dict[str, Any] = {}
    leave_focus = None
    matched = False
    lowered = message.lower().strip()
    prior_subject = inherited.get("subject") or workflow.get("analysis_kind")
    attendance_thread = prior_subject in {"org_attendance", "workforce_analytics"} or hint in {
        "org_attendance",
        "workforce_analytics",
    }
    if re.search(r"\bwhat about october\b|\boctober\b\??$", lowered):
        rewritten = (
            "How many employees were absent in October?"
            if prior_subject == "org_attendance"
            else f"{_prior_topic_phrase(workflow, inherited)} for October"
        )
        refs["month"] = "October"
        hint = "org_attendance" if prior_subject == "org_attendance" else (hint or "workforce_analytics")
        matched = True
    elif re.search(r"\bwhat about (september|november|december|january|february|march|april|may|june|july|august)\b", lowered):
        month = re.search(
            r"\b(september|november|december|january|february|march|april|may|june|july|august)\b",
            lowered,
        ).group(1)
        rewritten = (
            f"How many employees were absent in {month.title()}?"
            if prior_subject == "org_attendance"
            else f"{_prior_topic_phrase(workflow, inherited)} for {month.title()}"
        )
        refs["month"] = month.title()
        hint = "org_attendance" if prior_subject == "org_attendance" else (hint or "workforce_analytics")
        matched = True
    elif re.search(r"\bhow much did it increase\b|\bdid it increase\b", lowered):
        rewritten = (
            "How much did organization absences increase from the previous month to the latest month, "
            "including percentage change and percentage-point change?"
            if attendance_thread
            else f"How much did {_prior_topic_phrase(workflow, inherited)} increase compared with the previous period?"
        )
        hint = "org_attendance" if prior_subject == "org_attendance" else (hint or "workforce_analytics")
        matched = True
    elif re.search(r"\bwhich department (?:contributed most|had the largest increase)\b|\bdepartment contributed most\b", lowered):
        rewritten = (
            "Which department contributed most to the change in organization absences between the compared months?"
            if attendance_thread
            else f"Which department contributed most to {_prior_topic_phrase(workflow, inherited)}?"
        )
        hint = "org_attendance" if prior_subject == "org_attendance" else (hint or "workforce_analytics")
        matched = True
    elif re.search(r"\bwhat if i take only one week\b|\bonly one week\b", lowered):
        rewritten = "What if I take only one week of annual leave instead of the previously discussed duration?"
        refs["duration_days"] = 5
        hint = "leave_planning"
        leave_focus = "duration"
        matched = True
    elif re.search(
        r"\bcan i take (?:leave|two weeks|time off).{0,30}(?:next month|from now|later)\b",
        lowered,
    ):
        rewritten = message
        hint = "leave_eligibility"
        matched = True
    elif re.search(r"\bwould that affect my team'?s coverage\b|\bteam'?s coverage\b|\benough coverage\b", lowered):
        rewritten = "Would the leave window we just discussed affect my team's coverage?"
        hint = "leave_planning"
        leave_focus = "coverage"
        matched = True
    elif re.search(r"\bwhich (?:one|request|approval) has the highest (?:operational )?risk\b", lowered):
        rewritten = "Which of the pending approvals we just listed has the highest operational risk?"
        hint = "team_analytics"
        matched = True
    elif re.search(r"\b(pending approvals?|waiting for my (?:approval|decision)|awaiting my approval)\b", lowered):
        rewritten = message
        hint = "team_analytics"
        matched = True
    elif re.search(r"^\s*why\??\s*$", lowered):
        rewritten = f"Why is that the case for the previous answer about {_prior_topic_phrase(workflow, inherited)}?"
        hint = hint or workflow.get("prior_intent")
        matched = True
    elif re.search(r"\bwhat would you recommend\b", lowered):
        rewritten = f"What would you recommend based on the previous discussion about {_prior_topic_phrase(workflow, inherited)}?"
        hint = hint or workflow.get("prior_intent")
        matched = True
    return {
        "is_followup": True,
        "rewritten_message": rewritten if matched else message,
        "intent_hint": (hint if hint in ALLOWED_INTENT_HINTS else None) if matched or has_context else None,
        "leave_focus": leave_focus,
        "references": refs,
        "workflow": workflow,
        "resolver": "deterministic" if matched else "cue",
    }


def _prior_topic_phrase(workflow: dict[str, Any], inherited: dict[str, Any]) -> str:
    summary = workflow.get("decision_summary") or inherited.get("decision_summary")
    if summary:
        return str(summary).rstrip(".")
    intent = workflow.get("prior_intent") or inherited.get("subject") or "the previous HR question"
    return str(intent).replace("_", " ")


def history_for_prompt(history: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, str]]:
    """Compact role/content pairs for LLM evidence packs."""
    out: list[dict[str, str]] = []
    for item in history[-limit:]:
        out.append({"role": item.get("role") or "user", "content": (item.get("content") or "")[:800]})
    return out
