"""Deterministic HR tool registry.

The LLM may select tools and propose arguments. Python validates arguments,
re-checks authorization, and executes existing services. The model never runs
SQL and never bypasses RBAC.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.access import AuthorizationError, employee_for_user, sanitize_tool_arguments
from app.ai.context import load_actor, parse_business_days
from app.ai.retrieval import retrieve
from app.clock import add_business_days
from app.models import Approval, Employee, LeaveRequest, User
from app.services.actions import submit_leave
from app.services.analytics_engine import absence_analytics
from app.services.holidays import holidays_for_employee
from app.services.hr_ops import run_hr_report
from app.services.leave_engine import (
    analyze_team,
    assess_window,
    balance_payload,
    get_balance,
    investigate_routing,
    investigate_routing_discrepancies,
    plan_leave,
)
from app.services.records import attendance_report, leave_balance_report, parse_leave_dates


ToolHandler = Callable[..., dict]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    required_roles: frozenset[str]
    mutating: bool = False
    required_args: tuple[str, ...] = ()


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass
class ToolResult:
    name: str
    status: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    missing_args: list[str] = field(default_factory=list)
    error: str | None = None
    call_id: str | None = None

    def as_trace(self) -> dict:
        return {"tool": self.name, "status": self.status, "summary": self.summary[:180]}


TOOL_CATALOG: dict[str, ToolSpec] = {
    "get_leave_balance": ToolSpec(
        name="get_leave_balance",
        description="Retrieve the signed-in employee's leave balances (annual/sick).",
        parameters={
            "type": "object",
            "properties": {
                "leave_type": {
                    "type": "string",
                    "enum": ["annual", "sick", "both"],
                    "description": "Which leave balance to return. Default annual.",
                }
            },
            "additionalProperties": False,
        },
        required_roles=frozenset({"employee", "manager", "hr_admin"}),
    ),
    "evaluate_leave_feasibility": ToolSpec(
        name="evaluate_leave_feasibility",
        description="Evaluate whether a leave window is feasible (balance, notice, conflicts).",
        parameters={
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "ISO start date YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "ISO end date YYYY-MM-DD"},
                "business_days": {"type": "integer", "description": "Requested business days if end unknown"},
                "leave_type": {"type": "string", "enum": ["annual", "sick"]},
            },
            "additionalProperties": False,
        },
        required_roles=frozenset({"employee", "manager", "hr_admin"}),
    ),
    "create_leave_draft": ToolSpec(
        name="create_leave_draft",
        description="Build a leave draft/plan with balance, holidays, projects, and approvals. Does not submit.",
        parameters={
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "business_days": {"type": "integer"},
                "leave_type": {"type": "string", "enum": ["annual", "sick"]},
                "focus": {
                    "type": "string",
                    "description": "Optional leave focus such as holidays, projects, approvals, draft_review",
                },
            },
            "additionalProperties": False,
        },
        required_roles=frozenset({"employee", "manager", "hr_admin"}),
    ),
    "submit_leave_request": ToolSpec(
        name="submit_leave_request",
        description=(
            "Prepare or execute leave submission. Writes require confirmed=true after human confirmation. "
            "Without confirmation, returns a draft pending_action only."
        ),
        parameters={
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "leave_type": {"type": "string", "enum": ["annual", "sick"]},
                "reason": {"type": "string"},
                "confirmed": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        required_roles=frozenset({"employee", "manager", "hr_admin"}),
        mutating=True,
        required_args=("start_date", "end_date"),
    ),
    "get_attendance": ToolSpec(
        name="get_attendance",
        description="Retrieve attendance records or summaries for the signed-in employee.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language attendance question or window"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "additionalProperties": False,
        },
        required_roles=frozenset({"employee", "manager", "hr_admin"}),
    ),
    "analyze_team_coverage": ToolSpec(
        name="analyze_team_coverage",
        description="Analyze overlapping leave and coverage risk for the manager's team.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional focus such as overlapping leave"},
            },
            "additionalProperties": False,
        },
        required_roles=frozenset({"manager", "hr_admin"}),
    ),
    "get_approval_inbox": ToolSpec(
        name="get_approval_inbox",
        description="List pending leave approvals visible to the signed-in manager or HR admin.",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["pending", "all"], "default": "pending"},
            },
            "additionalProperties": False,
        },
        required_roles=frozenset({"manager", "hr_admin"}),
    ),
    "execute_approval": ToolSpec(
        name="execute_approval",
        description="Approve a pending leave request by leave request id or approval id.",
        parameters={
            "type": "object",
            "properties": {
                "request_id": {"type": "integer"},
                "approval_id": {"type": "integer"},
                "note": {"type": "string"},
            },
            "additionalProperties": False,
        },
        required_roles=frozenset({"manager", "hr_admin"}),
        mutating=True,
    ),
    "execute_rejection": ToolSpec(
        name="execute_rejection",
        description="Reject a pending leave request. Requires a rejection reason.",
        parameters={
            "type": "object",
            "properties": {
                "request_id": {"type": "integer"},
                "approval_id": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "additionalProperties": False,
        },
        required_roles=frozenset({"manager", "hr_admin"}),
        mutating=True,
        required_args=("reason",),
    ),
    "department_attendance_analytics": ToolSpec(
        name="department_attendance_analytics",
        description="Organization or department attendance analytics for HR Admin (rates, absences, trends).",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "department": {"type": "string"},
                "months": {"type": "integer"},
            },
            "additionalProperties": False,
        },
        required_roles=frozenset({"hr_admin"}),
    ),
    "compare_date_ranges": ToolSpec(
        name="compare_date_ranges",
        description="Compare attendance or absence metrics across named months or date ranges (HR Admin).",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Comparison question including periods, e.g. September vs October absences",
                },
                "department": {"type": "string"},
            },
            "additionalProperties": False,
        },
        required_roles=frozenset({"hr_admin"}),
        required_args=("query",),
    ),
    "investigate_employee_discrepancy": ToolSpec(
        name="investigate_employee_discrepancy",
        description=(
            "Investigate employee-record routing discrepancies such as manager vs leave-approval mismatches "
            "after department transfers (HR Admin)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "employee_name": {"type": "string"},
                "org_wide": {"type": "boolean", "description": "Scan all active employees when true"},
            },
            "additionalProperties": False,
        },
        required_roles=frozenset({"hr_admin"}),
    ),
    "search_hr_policies": ToolSpec(
        name="search_hr_policies",
        description="Retrieve relevant HR policy excerpts via hybrid RAG.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Policy question or topic"},
            },
            "additionalProperties": False,
        },
        required_roles=frozenset({"employee", "manager", "hr_admin"}),
        required_args=("query",),
    ),
}


def openai_tool_schemas() -> list[dict]:
    """OpenAI/OpenRouter function-calling schemas derived from the catalog."""
    tools = []
    for spec in TOOL_CATALOG.values():
        params = dict(spec.parameters)
        if spec.required_args:
            params = {**params, "required": list(spec.required_args)}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": params,
                },
            }
        )
    return tools


def validate_tool_call(call: ToolCall) -> ToolResult | None:
    """Return a clarification ToolResult when validation fails; otherwise None."""
    spec = TOOL_CATALOG.get(call.name)
    if spec is None:
        return ToolResult(
            name=call.name,
            status="error",
            summary=f"Unknown tool '{call.name}'.",
            error="unknown_tool",
            call_id=call.call_id,
        )
    if not isinstance(call.arguments, dict):
        return ToolResult(
            name=call.name,
            status="error",
            summary="Tool arguments must be a JSON object.",
            error="invalid_arguments",
            call_id=call.call_id,
        )
    missing = [key for key in spec.required_args if call.arguments.get(key) in (None, "")]
    # execute_approval needs request_id or approval_id
    if call.name in {"execute_approval", "execute_rejection"}:
        if call.arguments.get("request_id") in (None, "") and call.arguments.get("approval_id") in (None, ""):
            missing.append("request_id_or_approval_id")
    if missing:
        return ToolResult(
            name=call.name,
            status="clarify",
            summary=f"Missing required argument(s): {', '.join(missing)}.",
            missing_args=missing,
            data={"needs_clarification": True, "missing_args": missing},
            call_id=call.call_id,
        )
    return None


def authorize_tool(user: User, tool_name: str) -> None:
    spec = TOOL_CATALOG.get(tool_name)
    if spec is None:
        raise AuthorizationError(f"Unknown tool '{tool_name}'.", resource="tool", action="execute")
    if user.role not in spec.required_roles:
        raise AuthorizationError(
            f"The '{tool_name}' tool is not available for role '{user.role}'.",
            resource="tool",
            action="execute",
        )


def _parse_iso(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _approval_visible(user: User, actor: Employee, approval: Approval) -> bool:
    if user.role == "hr_admin":
        return True
    if approval.approver_employee_id == actor.id:
        return True
    if approval.subject_employee_id == actor.id:
        return True
    return False


def _resolve_approval(db: Session, *, request_id: int | None, approval_id: int | None) -> Approval | None:
    from app.access import resolve_leave_approval_reference

    return resolve_leave_approval_reference(
        db, request_id=request_id, approval_id=approval_id, pending_only=False
    )


def execute_tool(
    db: Session,
    user: User,
    call: ToolCall,
    *,
    message: str = "",
    inherited: dict | None = None,
    ip: str | None = None,
    confirmed: bool = False,
) -> ToolResult:
    """Validate, authorize, and execute one tool against deterministic services."""
    invalid = validate_tool_call(call)
    if invalid is not None:
        return invalid
    try:
        authorize_tool(user, call.name)
    except AuthorizationError as exc:
        return ToolResult(
            name=call.name,
            status="denied",
            summary=str(exc),
            error="unauthorized",
            call_id=call.call_id,
        )

    employee = employee_for_user(db, user)
    # Never trust LLM-supplied identity or privilege fields.
    args = sanitize_tool_arguments(call.arguments if isinstance(call.arguments, dict) else {})
    try:
        data = _DISPATCH[call.name](
            db,
            user,
            employee,
            args,
            message=message,
            inherited=inherited or {},
            ip=ip,
            confirmed=confirmed,
        )
    except AuthorizationError as exc:
        return ToolResult(
            name=call.name,
            status="denied",
            summary=str(exc),
            error="unauthorized",
            call_id=call.call_id,
        )
    except Exception as exc:  # noqa: BLE001 — surface tool failures without crashing the turn
        return ToolResult(
            name=call.name,
            status="error",
            summary=f"Tool execution failed: {exc.__class__.__name__}",
            error=str(exc)[:300],
            call_id=call.call_id,
        )

    summary = str(data.get("summary") or data.get("message") or f"{call.name} completed")[:180]
    status = str(data.get("tool_status") or "ok")
    return ToolResult(
        name=call.name,
        status=status,
        summary=summary,
        data=data,
        call_id=call.call_id,
        missing_args=list(data.get("missing_args") or []),
    )


def _get_leave_balance(db, user, employee, args, **_kwargs) -> dict:
    leave_type = (args.get("leave_type") or "annual").lower()
    query = "annual leave balance"
    if leave_type == "sick":
        query = "sick leave balance"
    elif leave_type == "both":
        query = "annual and sick leave balance compare"
    report = leave_balance_report(db, employee.id, query)
    return {
        **report,
        "kind": report.get("kind") or "leave_balance",
        "summary": f"Leave balance retrieved for {employee.full_name}.",
        "state_patch": {"analysis": report, "record_context_subject": "leave_balance"},
    }


def _evaluate_leave_feasibility(db, user, employee, args, *, message="", inherited=None, **_kwargs) -> dict:
    leave_type = args.get("leave_type") or "annual"
    start = _parse_iso(args.get("start_date"))
    end = _parse_iso(args.get("end_date"))
    business_days = args.get("business_days")
    if business_days is None:
        business_days = parse_business_days(message) or 10
    if start is None or end is None:
        parsed = parse_leave_dates(message) or {}
        start = start or parsed.get("start")
        end = end or parsed.get("end")
    if start is None and end is None and inherited and inherited.get("start"):
        try:
            start = date.fromisoformat(str(inherited["start"])[:10])
            end = date.fromisoformat(str(inherited["end"])[:10]) if inherited.get("end") else None
        except ValueError:
            pass
    if start is None:
        return {
            "kind": "leave_feasibility",
            "tool_status": "clarify",
            "needs_clarification": True,
            "missing_args": ["start_date"],
            "message": "I need a start date (and preferably an end date) to evaluate leave feasibility.",
            "summary": "Missing leave dates.",
        }
    if end is None:
        closed = holidays_for_employee(db, employee, start, start + timedelta(days=60)).dates
        end = add_business_days(start, max(int(business_days), 1), closed)
    assessment = assess_window(db, employee, start, end, leave_type=leave_type)
    balance = balance_payload(get_balance(db, employee.id, leave_type))
    return {
        "kind": "leave_feasibility",
        "assessment": assessment,
        "balance": balance,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "leave_type": leave_type,
        "hard_blocks": assessment.get("hard_blocks") or [],
        "summary": (
            f"Feasibility for {start.isoformat()}–{end.isoformat()}: "
            + ("blocked" if assessment.get("hard_blocks") else "eligible with confirmation")
        ),
        "state_patch": {
            "analysis": {
                "kind": "leave_feasibility",
                "requested_window": assessment,
                "balance": balance,
                "leave_type": leave_type,
            }
        },
    }


def _create_leave_draft(db, user, employee, args, *, message="", inherited=None, **_kwargs) -> dict:
    inherited = inherited or {}
    focus = args.get("focus") or inherited.get("focus") or "plan"
    leave_type = args.get("leave_type") or inherited.get("leave_type") or "annual"
    stated = args.get("business_days")
    if stated is None:
        stated = parse_business_days(message)
    if stated is None and inherited.get("business_days"):
        stated = int(inherited["business_days"])
    days = int(stated or 10)
    start = _parse_iso(args.get("start_date"))
    end = _parse_iso(args.get("end_date"))
    parsed = parse_leave_dates(message) if message else None
    if start is None and parsed:
        start = parsed.get("start")
    if end is None and parsed:
        end = parsed.get("end")
    if start is None and inherited.get("start"):
        try:
            start = date.fromisoformat(str(inherited["start"])[:10])
        except ValueError:
            start = None
    if end is None and inherited.get("end") and stated is None:
        try:
            end = date.fromisoformat(str(inherited["end"])[:10])
        except ValueError:
            end = None
    if start is not None and end is None:
        closed = holidays_for_employee(db, employee, start, start + timedelta(days=60)).dates
        end = add_business_days(start, max(days, 1), closed)
    if (
        stated is not None
        and start is not None
        and parsed is None
        and inherited.get("subject") == "leave_plan"
        and not re.search(r"\b(from|starting|until|through|to)\b", (message or "").lower())
    ):
        closed = holidays_for_employee(db, employee, start, start + timedelta(days=60)).dates
        end = add_business_days(start, max(int(stated), 1), closed)
        days = int(stated)
    analysis = plan_leave(
        db,
        employee,
        business_days=max(days, 1),
        start=start,
        end=end,
        leave_type=leave_type,
    )
    requested = analysis.get("requested_window") or {}
    hard_blocks = requested.get("hard_blocks") or []
    if hard_blocks:
        workflow_state = "VALIDATING"
    elif focus in {"draft_review", "confirmation_gate", "prepare"}:
        workflow_state = "DRAFT_READY" if focus != "confirmation_gate" else "AWAITING_CONFIRMATION"
    else:
        workflow_state = "PLANNING"
    analysis = {
        **analysis,
        "kind": "leave_plan",
        "focus": focus or "plan",
        "workflow_state": workflow_state,
        "confirmation_required": True,
        "explicit_confirmation_received": False,
    }
    record_context = {
        "subject": "leave_plan",
        "intent": "leave_planning",
        "user_id": user.id,
        "employee_id": employee.id,
        "leave_type": leave_type,
        "start": requested.get("start"),
        "end": requested.get("end"),
        "label": requested.get("label"),
        "business_days": requested.get("business_days"),
        "remaining_before": requested.get("remaining_before"),
        "workflow_state": workflow_state,
        "hard_blocks": hard_blocks,
        "focus": focus or "plan",
        "confirmation_required": True,
    }
    return {
        **analysis,
        "summary": requested.get("label") or "leave draft",
        "state_patch": {
            "analysis": analysis,
            "record_context": record_context,
            "leave_focus": focus or "plan",
            "intent": "leave_planning",
        },
    }


def _submit_leave_request(db, user, employee, args, *, message="", ip=None, confirmed=False, **_kwargs) -> dict:
    start = _parse_iso(args.get("start_date"))
    end = _parse_iso(args.get("end_date"))
    leave_type = args.get("leave_type") or "annual"
    reason = args.get("reason") or "Planned annual leave"
    if start is None or end is None:
        return {
            "tool_status": "clarify",
            "needs_clarification": True,
            "missing_args": [k for k, v in (("start_date", start), ("end_date", end)) if v is None],
            "message": "Leave submission needs start_date and end_date.",
            "summary": "Missing leave dates for submission.",
        }
    assessment = assess_window(db, employee, start, end, leave_type=leave_type)
    if assessment.get("hard_blocks"):
        return {
            "kind": "leave_submit",
            "tool_status": "blocked",
            "status": "blocked",
            "hard_blocks": assessment["hard_blocks"],
            "message": " ".join(assessment["hard_blocks"]),
            "summary": "Leave submission blocked by policy/balance rules.",
            "assessment": assessment,
        }
    if not confirmed and not args.get("confirmed"):
        pending = {
            "type": "submit_leave",
            "confirmation_prompt": "Confirm which leave request I should file. I will not approve it.",
            "options": {
                "primary": {
                    "id": "primary",
                    "label": f"Submit {start.isoformat()} to {end.isoformat()}",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "business_days": assessment.get("business_days"),
                    "risk_level": assessment.get("risk_level"),
                    "hr_required": assessment.get("hr_required"),
                    "leave_type": leave_type,
                }
            },
            "default_option": "primary",
            "workflow_state": "AWAITING_CONFIRMATION",
        }
        return {
            "kind": "leave_submit",
            "tool_status": "pending_confirmation",
            "status": "pending_confirmation",
            "message": "Leave is drafted and waiting for your confirmation. No record was written.",
            "summary": "Awaiting confirmation before submit_leave.",
            "assessment": assessment,
            "state_patch": {"pending_action": pending, "analysis": {"kind": "leave_plan", "requested_window": assessment, "leave_type": leave_type}},
        }
    result = submit_leave(
        db,
        user,
        start=start,
        end=end,
        leave_type=leave_type,
        reason=reason,
        ip_address=ip,
    )
    return {
        **result,
        "kind": "leave_submit",
        "summary": result.get("summary") or result.get("message") or "Leave submitted",
        "state_patch": {"execution_result": result},
    }


def _get_attendance(db, user, employee, args, *, message="", inherited=None, **_kwargs) -> dict:
    # Prefer the user message for window parsing so LLM-proposed calendar years
    # cannot drift away from ORG_CLOCK (e.g. inventing 2023 for "October").
    query = message or args.get("query") or "my attendance"
    report = attendance_report(db, employee.id, query, inherited)
    return {
        **report,
        "summary": f"Attendance report for {employee.full_name}.",
        "state_patch": {"analysis": report, "record_context_subject": "attendance"},
    }


def _analyze_team_coverage(db, user, employee, args, *, message="", **_kwargs) -> dict:
    from app.services.leave_engine import resolve_team_overlap_window

    if user.role not in {"manager", "hr_admin"}:
        raise AuthorizationError("Team coverage analysis is limited to managers and HR administrators.", resource="team")
    window_start, window_end, window_label, window_err = resolve_team_overlap_window(message or "")
    if window_err:
        return {
            "kind": "team_conflict",
            **window_err,
            "summary": window_err.get("message", "Clarify the overlap review window."),
            "state_patch": {
                "analysis": {"kind": "team_conflict", **window_err},
                "intent": "team_conflict",
            },
        }
    analysis = analyze_team(
        db,
        employee,
        filter_start=window_start,
        filter_end=window_end,
        filter_label=window_label or None,
        message=message or "",
    )
    pair_count = len(analysis.get("overlap_pairs") or [])
    request_count = len(analysis.get("requests") or [])
    if pair_count:
        summary = (
            f"{pair_count} overlapping leave pair(s) among direct reports "
            f"(pending/approved); densest week has {request_count} request(s)"
        )
    else:
        summary = (
            f"{request_count} requests in "
            f"{(analysis.get('window') or {}).get('label', 'the review window')}"
        )
    return {
        **analysis,
        "summary": summary,
        "state_patch": {
            "analysis": analysis,
            "record_context": {
                "subject": "team_conflict",
                "user_id": user.id,
                "employee_id": employee.id,
                "focus": "overlap",
            },
            "intent": "team_conflict",
        },
    }


def _get_approval_inbox(db, user, employee, args, *, message="", inherited=None, **_kwargs) -> dict:
    from app.ai.context import approval_inbox_followup_focus
    from app.services.team_ops import team_analytics_report

    status = (args.get("status") or "pending").lower()
    display_limit = int(args.get("display_limit") or 20)
    follow_focus = approval_inbox_followup_focus(message) or args.get("focus")
    # Managers/HR: use the same ranked pending_manager inbox as team analytics so urgency
    # follow-ups keep request ids and operational-risk explanations.
    if user.role in {"manager", "hr_admin"}:
        report_message = message or "Show me my pending approvals."
        if follow_focus == "urgency":
            report_message = "Which request is most urgent?"
        elif follow_focus == "why_urgent":
            report_message = "Why is it urgent?"
        elif follow_focus == "approve_impact":
            report_message = "What happens if I approve it?"
        elif follow_focus == "reject_impact":
            report_message = "What happens if I reject it?"
        elif follow_focus == "approve_guidance":
            report_message = message or "Should I approve this leave request?"
        elif follow_focus == "reject_guidance":
            report_message = message or "Should I reject this leave request?"
        analysis = team_analytics_report(db, employee, report_message, inherited or {})
        selected_id = analysis.get("selected_request_id")
        if selected_id is None and (analysis.get("highest") or {}).get("request_id"):
            selected_id = analysis["highest"]["request_id"]
        people = analysis.get("people") or analysis.get("inbox") or []
        rows = []
        for item in people:
            rows.append(
                {
                    "approval_id": item.get("request_id"),
                    "leave_request_id": item.get("request_id"),
                    "status": item.get("status") or "pending",
                    "stage": "manager",
                    "subject": item.get("full_name"),
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "business_days": item.get("business_days"),
                    "leave_type": item.get("leave_type"),
                    "risk_level": item.get("risk_level"),
                    "risks": item.get("risks") or [],
                }
            )
        # Keep the richer team_analytics analysis for narration, while preserving inbox rows.
        analysis = {
            **analysis,
            "kind": "approval_inbox",
            "status_filter": status,
            "count": int(analysis.get("people_count") or len(rows)),
            "display_limit": display_limit,
            "displayed_count": min(len(rows), display_limit),
            "truncated": len(rows) > display_limit,
            "approvals": rows[:display_limit],
            "selected_request_id": selected_id,
        }
        focus = analysis.get("focus") or follow_focus or "awaiting_approval"
        return {
            **analysis,
            "summary": f"{analysis.get('count', 0)} pending approval(s) in inbox.",
            "state_patch": {
                "analysis": analysis,
                "intent": "team_analytics",
                "record_context": {
                    "subject": "approval_inbox",
                    "user_id": user.id,
                    "employee_id": employee.id,
                    "focus": focus,
                    "selected_request_id": selected_id,
                    "request_id": selected_id,
                },
            },
        }

    query = db.query(Approval)
    if status != "all":
        query = query.filter(Approval.status == "pending")
    rows = []
    for approval in query.order_by(Approval.created_at.desc()).limit(100).all():
        if not _approval_visible(user, employee, approval):
            continue
        subject = db.get(Employee, approval.subject_employee_id) if approval.subject_employee_id else None
        leave = db.get(LeaveRequest, approval.leave_request_id) if approval.leave_request_id else None
        rows.append(
            {
                "approval_id": approval.id,
                "leave_request_id": approval.leave_request_id,
                "status": approval.status,
                "stage": approval.stage,
                "subject": None if subject is None else subject.full_name,
                "start": None if leave is None else leave.start_date.isoformat(),
                "end": None if leave is None else leave.end_date.isoformat(),
                "business_days": None if leave is None else leave.business_days,
                "leave_type": None if leave is None else leave.leave_type,
            }
        )
    selected_id = rows[0]["leave_request_id"] if rows else None
    analysis = {
        "kind": "approval_inbox",
        "status_filter": status,
        "count": len(rows),
        "display_limit": display_limit,
        "displayed_count": min(len(rows), display_limit),
        "truncated": len(rows) > display_limit,
        "approvals": rows,
        "selected_request_id": selected_id,
    }
    return {
        **analysis,
        "summary": f"{len(rows)} approval(s) in inbox.",
        "state_patch": {
            "analysis": analysis,
            "record_context": {
                "subject": "approval_inbox",
                "user_id": user.id,
                "employee_id": employee.id,
                "focus": "approval_inbox",
                "selected_request_id": selected_id,
                "request_id": selected_id,
            },
        },
    }


def _execute_approval(db, user, employee, args, *, message="", ip=None, **_kwargs) -> dict:
    from app.access import approval_decision_access
    from app.ai.context import find_mentioned, get_ctx
    from app.services.approval_resolution import nl_approval_resolution
    from app.services.confirmation import prepare_decision_pending

    request_id = args.get("request_id")
    approval_id = args.get("approval_id")
    rid = int(request_id) if request_id is not None else None
    aid = int(approval_id) if approval_id is not None else None
    if rid is None and aid is None and (message or "").strip():
        mentioned = find_mentioned(db, message, employee.id)
        resolution, _parsed = nl_approval_resolution(
            db, user, employee, message, [m.id for m in mentioned]
        )
        if resolution:
            if resolution["status"] == "clarify":
                return {
                    "kind": "approval_decision",
                    "tool_status": "clarify",
                    "status": "needs_clarification",
                    "message": resolution.get("message") or "",
                    "summary": "Clarify which leave request.",
                }
            if resolution["status"] in {"not_found", "unauthorized"}:
                return {
                    "kind": "approval_decision",
                    "tool_status": "denied",
                    "status": "denied",
                    "message": resolution.get("message") or "",
                    "summary": "Approval denied by authorization.",
                }
            if resolution["status"] == "resolved":
                rid = int(resolution["request_id"])
    ok, gate_reason = approval_decision_access(db, user, request_id=rid, approval_id=aid)
    if not ok:
        return {
            "kind": "approval_decision",
            "tool_status": "denied",
            "status": "denied",
            "message": gate_reason,
            "summary": "Approval denied by authorization.",
        }
    approval = _resolve_approval(db, request_id=rid, approval_id=aid)
    if approval is None:
        return {
            "kind": "approval_decision",
            "tool_status": "denied",
            "status": "not_found",
            "message": "No pending approval was found for that id.",
            "summary": "Approval not found.",
        }
    if approval.status != "pending":
        return {
            "kind": "approval_decision",
            "tool_status": "denied",
            "status": "duplicate",
            "decision": "approve",
            "request_id": approval.leave_request_id or request_id,
            "approval_id": approval.id,
            "message": (
                f"Approval {approval.id} is already {approval.status}. "
                "It cannot be decided again."
            ),
            "summary": "Approval already decided.",
        }
    if not can_decide_approval(user, approval):
        return {
            "kind": "approval_decision",
            "tool_status": "denied",
            "status": "denied",
            "decision": "approve",
            "request_id": approval.leave_request_id or request_id,
            "approval_id": approval.id,
            "message": (
                "You are not authorized to approve this leave request. "
                "No record was changed."
            ),
            "summary": "Approval denied by authorization.",
        }
    ctx = get_ctx()
    pending = prepare_decision_pending(
        db,
        user,
        conversation_id=getattr(ctx, "conversation_id", None),
        approval_id=approval.id,
        request_id=approval.leave_request_id or request_id,
        decision="approve",
        reason=args.get("note") or "",
        thread_id=getattr(ctx, "thread_id", None),
        ip=ip,
    )
    analysis = {
        "kind": "approval_decision",
        "status": "awaiting_confirmation",
        "decision": "approve",
        "request_id": approval.leave_request_id or request_id,
        "approval_id": approval.id,
        "confirmation_required": True,
        "message": (
            f"I prepared an approval for leave request {approval.leave_request_id or request_id}. "
            "Confirm to apply it. No record has been changed yet."
        ),
    }
    return {
        **analysis,
        "summary": "Approval prepared; awaiting confirmation.",
        "state_patch": {
            "analysis": analysis,
            "pending_action": pending,
            "intent": "approval_decision",
        },
    }


def _execute_rejection(db, user, employee, args, *, message="", ip=None, **_kwargs) -> dict:
    from app.access import approval_decision_access, can_decide_approval
    from app.ai.context import find_mentioned, get_ctx
    from app.services.approval_resolution import nl_approval_resolution
    from app.services.confirmation import prepare_decision_pending

    request_id = args.get("request_id")
    approval_id = args.get("approval_id")
    rid = int(request_id) if request_id is not None else None
    aid = int(approval_id) if approval_id is not None else None
    if rid is None and aid is None and (message or "").strip():
        mentioned = find_mentioned(db, message, employee.id)
        resolution, _parsed = nl_approval_resolution(
            db, user, employee, message, [m.id for m in mentioned]
        )
        if resolution:
            if resolution["status"] == "clarify":
                return {
                    "kind": "approval_decision",
                    "tool_status": "clarify",
                    "status": "needs_clarification",
                    "message": resolution.get("message") or "",
                    "summary": "Clarify which leave request.",
                }
            if resolution["status"] in {"not_found", "unauthorized"}:
                return {
                    "kind": "approval_decision",
                    "tool_status": "denied",
                    "status": "denied",
                    "message": resolution.get("message") or "",
                    "summary": "Rejection denied by authorization.",
                }
            if resolution["status"] == "resolved":
                rid = int(resolution["request_id"])

    reason = (args.get("reason") or "").strip()
    if not reason:
        return {
            "kind": "approval_decision",
            "tool_status": "clarify",
            "status": "needs_reason",
            "needs_clarification": True,
            "missing_args": ["reason"],
            "message": "A rejection reason is required.",
            "summary": "Rejection reason required.",
            "request_id": rid,
            "summary_context": {"request_id": rid},
        }
    ok, gate_reason = approval_decision_access(db, user, request_id=rid, approval_id=aid)
    if not ok:
        return {
            "kind": "approval_decision",
            "tool_status": "denied",
            "status": "denied",
            "message": gate_reason,
            "summary": "Rejection denied by authorization.",
        }
    approval = _resolve_approval(db, request_id=rid, approval_id=aid)
    if approval is None:
        return {
            "kind": "approval_decision",
            "tool_status": "denied",
            "status": "not_found",
            "message": "No pending approval was found for that id.",
            "summary": "Approval not found.",
        }
    if approval.status != "pending":
        return {
            "kind": "approval_decision",
            "tool_status": "denied",
            "status": "duplicate",
            "decision": "reject",
            "request_id": approval.leave_request_id or request_id,
            "approval_id": approval.id,
            "reason": reason,
            "message": (
                f"Approval {approval.id} is already {approval.status}. "
                "It cannot be decided again."
            ),
            "summary": "Approval already decided.",
        }
    if not can_decide_approval(user, approval):
        return {
            "kind": "approval_decision",
            "tool_status": "denied",
            "status": "denied",
            "decision": "reject",
            "request_id": approval.leave_request_id or request_id,
            "approval_id": approval.id,
            "reason": reason,
            "message": (
                "You are not authorized to reject this leave request. "
                "No record was changed."
            ),
            "summary": "Rejection denied by authorization.",
        }
    ctx = get_ctx()
    pending = prepare_decision_pending(
        db,
        user,
        conversation_id=getattr(ctx, "conversation_id", None),
        approval_id=approval.id,
        request_id=approval.leave_request_id or request_id,
        decision="reject",
        reason=reason,
        thread_id=getattr(ctx, "thread_id", None),
        ip=ip,
    )
    analysis = {
        "kind": "approval_decision",
        "status": "awaiting_confirmation",
        "decision": "reject",
        "request_id": approval.leave_request_id or request_id,
        "approval_id": approval.id,
        "reason": reason,
        "confirmation_required": True,
        "message": (
            f"I prepared a rejection for leave request {approval.leave_request_id or request_id}. "
            "Confirm to apply it. No record has been changed yet."
        ),
    }
    return {
        **analysis,
        "summary": "Rejection prepared; awaiting confirmation.",
        "state_patch": {
            "analysis": analysis,
            "pending_action": pending,
            "intent": "approval_decision",
        },
    }


def _department_attendance_analytics(db, user, employee, args, *, message="", inherited=None, **_kwargs) -> dict:
    # Always ground period/department parsing on the user message, not LLM paraphrases.
    # Never append model-invented department filters — that silently shrinks org-wide counts.
    # Managers/employees must never hit the organization attendance report; scope is applied
    # inside absence_analytics before attendance rows are loaded.
    from app.ai.context import (
        org_attendance_month_comparison_question,
        org_attendance_named_month_rate_question,
        workforce_analytics_focus,
        workforce_followup_question,
    )

    query = message or args.get("query") or ""
    months = int(args.get("months") or 3)
    inherited = inherited or {}
    org_month_ask = org_attendance_named_month_rate_question(query) or org_attendance_month_comparison_question(
        query
    )
    use_scoped_analytics = user.role != "hr_admin" or (
        not org_month_ask
        and (
            workforce_followup_question(query)
            or (
                re.search(r"\b(pattern|recommend|concern|trend|workforce)\b", query, re.I)
                and not re.search(r"\b(how many|september|october|november|absent employees)\b", query, re.I)
            )
        )
    )
    if use_scoped_analytics:
        payload = absence_analytics(db, user, months=months)
        focus = workforce_analytics_focus(
            query,
            prior_focus=str(inherited.get("focus") or ""),
            subject=str(inherited.get("subject") or ""),
        )
        return {
            **payload,
            "summary": f"Workforce absence analytics computed ({focus}, scope={payload.get('scope')}).",
            "state_patch": {
                "analytics": {**payload, "followup_focus": focus},
                "analysis": {
                    **(payload if isinstance(payload, dict) else {}),
                    "kind": "workforce_analytics",
                    "focus": focus,
                    "scope": payload.get("scope"),
                    "scope_label": payload.get("scope_label"),
                    "employee_ids": payload.get("employee_ids") or [],
                    "priority_employees": payload.get("priority_employees") or [],
                },
                "intent": "workforce_analytics",
                "record_context": {
                    "subject": "workforce_analytics",
                    "focus": focus,
                    "months": months,
                    "user_id": user.id,
                    "employee_id": employee.id,
                },
            },
        }
    report, context = run_hr_report(db, "org_attendance", query, inherited, [])
    if context is not None:
        context = {**context, "user_id": user.id, "employee_id": employee.id}
    return {
        **report,
        "summary": report.get("summary") or "Department attendance analytics computed.",
        "state_patch": {
            "analysis": report,
            "record_context": context,
            "intent": "org_attendance",
        },
    }


def _compare_date_ranges(db, user, employee, args, *, message="", inherited=None, **_kwargs) -> dict:
    query = message or args.get("query") or ""
    if "compare" not in query.lower() and not re.search(r"\b(versus|vs\.?|increase|decrease|change)\b", query, re.I):
        query = f"compare {query}"
    # Organization attendance comparisons are HR Admin only. Managers get scoped analytics.
    if user.role != "hr_admin":
        payload = absence_analytics(db, user, months=3)
        return {
            **payload,
            "summary": f"Scoped attendance comparison computed (scope={payload.get('scope')}).",
            "state_patch": {
                "analytics": payload,
                "analysis": {
                    **payload,
                    "kind": "workforce_analytics",
                    "focus": "month_comparison",
                    "scope": payload.get("scope"),
                    "scope_label": payload.get("scope_label"),
                    "employee_ids": payload.get("employee_ids") or [],
                },
                "intent": "workforce_analytics",
                "record_context": {
                    "subject": "workforce_analytics",
                    "focus": "month_comparison",
                    "user_id": user.id,
                    "employee_id": employee.id,
                },
            },
        }
    report, context = run_hr_report(db, "org_attendance", query, inherited, [])
    if context is not None:
        context = {**context, "user_id": user.id, "employee_id": employee.id}
    return {
        **report,
        "summary": report.get("summary") or "Date-range comparison computed.",
        "state_patch": {
            "analysis": report,
            "record_context": context,
            "intent": "org_attendance",
        },
    }


def _investigate_employee_discrepancy(db, user, employee, args, *, message="", **_kwargs) -> dict:
    from app.ai.context import org_transfer_followup_focus

    name = (args.get("employee_name") or "").strip()
    org_wide = bool(args.get("org_wide")) or not name
    focus = args.get("focus") or org_transfer_followup_focus(message) or "summary"
    if org_wide or re.search(r"\b(an employee|previous manager|approval still)\b", name, re.I):
        analysis = investigate_routing_discrepancies(db)
    else:
        subject = (
            db.query(Employee)
            .filter(Employee.employment_status == "active")
            .all()
        )
        match = None
        needle = name.lower()
        for row in subject:
            if needle in row.full_name.lower():
                match = row
                break
        if match is None:
            analysis = investigate_routing_discrepancies(db)
        else:
            analysis = investigate_routing(db, match)
    primary = analysis.get("most_urgent") or analysis
    analysis = {**analysis, "focus": focus}
    return {
        **analysis,
        "summary": analysis.get("issue") or "Routing investigation complete.",
        "state_patch": {
            "analysis": analysis,
            "intent": "org_transfer",
            "record_context": {
                "subject": "routing_audit" if analysis.get("kind") == "routing_audit" else "routing_investigation",
                "focus": focus,
                "user_id": user.id,
                "employee_id": employee.id,
                "subject_employee_id": primary.get("employee_id") or analysis.get("employee_id"),
                "severity": primary.get("severity") or analysis.get("severity"),
                "issue": primary.get("issue") or analysis.get("issue"),
            },
        },
    }


def _search_hr_policies(db, user, employee, args, *, message="", **_kwargs) -> dict:
    query = args.get("query") or message
    if not query:
        return {
            "tool_status": "clarify",
            "needs_clarification": True,
            "missing_args": ["query"],
            "message": "Which HR policy topic should I look up?",
            "summary": "Missing policy query.",
        }
    _user, _employee, department = load_actor(db, user.id)
    result = retrieve(
        db,
        query,
        role=user.role,
        department=department.name if department else None,
        location=employee.location,
    )
    return {
        "kind": "policy_retrieval",
        "retrieval": result,
        "citations": result.get("citations") or [],
        "retrieval_confidence": result.get("confidence"),
        "summary": f"Retrieved {len(result.get('citations') or [])} policy citation(s).",
        "state_patch": {
            "retrieval": result,
            "citations": result.get("citations") or [],
            "retrieval_confidence": result.get("confidence"),
            "intent": "policy_question",
        },
    }


_DISPATCH: dict[str, ToolHandler] = {
    "get_leave_balance": _get_leave_balance,
    "evaluate_leave_feasibility": _evaluate_leave_feasibility,
    "create_leave_draft": _create_leave_draft,
    "submit_leave_request": _submit_leave_request,
    "get_attendance": _get_attendance,
    "analyze_team_coverage": _analyze_team_coverage,
    "get_approval_inbox": _get_approval_inbox,
    "execute_approval": _execute_approval,
    "execute_rejection": _execute_rejection,
    "department_attendance_analytics": _department_attendance_analytics,
    "compare_date_ranges": _compare_date_ranges,
    "investigate_employee_discrepancy": _investigate_employee_discrepancy,
    "search_hr_policies": _search_hr_policies,
}


def merge_tool_results_into_state(results: list[ToolResult]) -> dict[str, Any]:
    """Fold tool outputs into LangGraph state fields."""
    patch: dict[str, Any] = {
        "tool_results": [
            {
                "name": item.name,
                "status": item.status,
                "summary": item.summary,
                "missing_args": item.missing_args,
                "error": item.error,
                "data_kind": (item.data or {}).get("kind"),
            }
            for item in results
        ]
    }
    for item in results:
        data = item.data or {}
        state_patch = data.get("state_patch") or {}
        for key, value in state_patch.items():
            if key == "record_context_subject":
                continue
            patch[key] = value
        if "analysis" not in state_patch and data.get("kind") and item.status == "ok":
            # Prefer explicit analysis payloads from tools.
            if any(k in data for k in ("requested_window", "approvals", "assessment", "balances", "cases")):
                patch["analysis"] = {k: v for k, v in data.items() if k not in {"state_patch", "summary", "tool_status"}}
        if data.get("pending_action"):
            patch["pending_action"] = data["pending_action"]
        if item.status == "clarify" or data.get("needs_clarification"):
            patch["clarify"] = True
            patch["clarification"] = data.get("message") or item.summary
    return patch
