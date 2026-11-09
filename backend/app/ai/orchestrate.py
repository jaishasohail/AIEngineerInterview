"""LLM tool selection + validated backend execution.

Flow:
1. Authorized actor/context already loaded by the graph
2. Relevant knowledge may already be attached
3. Conversation history is assembled
4. LLM (or deterministic planner fallback) selects tools + args
5. Backend validates, authorizes, and executes services
6. Results are returned for grounded final response generation
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy.orm import Session

from app.ai.llm import LLMClient, PermanentLLMError, TransientLLMError, parse_json_object
from app.ai.tools import (
    TOOL_CATALOG,
    ToolCall,
    ToolResult,
    execute_tool,
    merge_tool_results_into_state,
    openai_tool_schemas,
)
from app.ai.context import (
    approval_inbox_followup_focus,
    approval_inbox_question,
    org_transfer_followup_focus,
    parse_approval_decision,
    routing_discrepancy_question,
    workforce_followup_question,
)
from app.models import User

logger = logging.getLogger("helios.orchestrate")

MAX_TOOL_ROUNDS = 3


def deterministic_tool_plan(
    *,
    message: str,
    role: str,
    intent: str | None,
    leave_focus: str | None = None,
    record_context: dict | None = None,
) -> list[ToolCall]:
    """Rule-based tool planner used when the LLM is unavailable or returns no tools.

    Mirrors assessment-relevant capabilities without inventing analytics.
    """
    text = (message or "").lower()
    intent = intent or ""
    calls: list[ToolCall] = []
    inherited = record_context or {}

    if intent == "approval_decision" or parse_approval_decision(message):
        parsed = parse_approval_decision(message) or {}
        decision = parsed.get("decision")
        request_id = parsed.get("request_id")
        if request_id is None:
            for key in ("selected_request_id", "request_id"):
                if inherited.get(key) is not None and inherited.get("subject") in {
                    "approval_decision",
                    "approval_inbox",
                    "team_analytics",
                }:
                    request_id = int(inherited[key])
                    break
        if decision is None and re.search(
            r"\b(prepare (?:the |this |that )?(?:approval|decision)|wait for my confirmation)\b",
            text,
        ):
            decision = "approve"
        if decision == "approve" and request_id is not None:
            calls.append(ToolCall("execute_approval", {"request_id": request_id, "note": parsed.get("reason") or ""}))
        elif decision == "reject" and request_id is not None:
            args = {"request_id": request_id, "reason": parsed.get("reason") or ""}
            calls.append(ToolCall("execute_rejection", args))
        else:
            calls.append(ToolCall("get_approval_inbox", {"status": "pending"}))
        return calls

    from app.ai.context import delayed_approval_advice_question, leave_date_recommendation_question

    # Date-recommendation / delayed-approval advice must never open a leave draft.
    if leave_date_recommendation_question(message) or leave_focus == "recommend_dates":
        calls.append(ToolCall("get_leave_balance", {"leave_type": "annual"}))
        calls.append(ToolCall("search_hr_policies", {"query": message or "annual leave notice coverage"}))
        return calls
    if delayed_approval_advice_question(message):
        calls.append(ToolCall("search_hr_policies", {"query": "leave approval manager People Operations co-approval"}))
        return calls

    if intent in {"leave_planning", "leave_request"} or re.search(
        r"\b(plan leave|apply for leave|book leave|request leave|take leave)\b", text
    ):
        args: dict[str, Any] = {}
        if leave_focus:
            args["focus"] = leave_focus
        calls.append(ToolCall("create_leave_draft", args))
        if re.search(r"\b(policy|notice|approval chain|eligib)\b", text):
            calls.append(ToolCall("search_hr_policies", {"query": message}))
        return calls

    if intent == "leave_eligibility" or (
        re.search(r"\b(eligib|enough (leave )?days|can i take)\b", text)
        and re.search(r"\b(leave|annual|vacation)\b", text)
    ):
        calls.append(ToolCall("get_leave_balance", {"leave_type": "annual"}))
        calls.append(ToolCall("evaluate_leave_feasibility", {}))
        calls.append(ToolCall("search_hr_policies", {"query": message or "annual leave eligibility notice"}))
        return calls

    if intent == "leave_balance" or re.search(r"\b(leave balance|how many .*leave days|remaining (annual )?leave)\b", text):
        leave_type = "both" if re.search(r"\b(sick|both|compare)\b", text) else "annual"
        if "sick" in text and "annual" not in text:
            leave_type = "sick"
        calls.append(ToolCall("get_leave_balance", {"leave_type": leave_type}))
        return calls

    if intent == "attendance_query" or (
        re.search(r"\b(attendance|present|remote|late|absent)\b", text)
        and role != "hr_admin"
        and not re.search(r"\b(department|organization|org-wide|across)\b", text)
    ):
        calls.append(ToolCall("get_attendance", {"query": message}))
        return calls

    if intent == "team_conflict" or re.search(
        r"\b("
        r"overlap(?:ping)? leave(?: dates?)?|"
        r"employees? .{0,40}overlap|"
        r"leave(?: dates?)? overlap|"
        r"team coverage|coverage risk|who is out|"
        r"leave conflicts?"
        r")\b",
        text,
    ):
        calls.append(ToolCall("analyze_team_coverage", {"query": message}))
        return calls

    inbox_follow = approval_inbox_followup_focus(message)
    if intent == "team_analytics" and role in {"manager", "hr_admin"} and (
        approval_inbox_question(message)
        or inbox_follow in {"urgency", "why_urgent", "approve_impact", "reject_impact"}
    ):
        args: dict[str, Any] = {"status": "pending"}
        if inbox_follow:
            args["focus"] = inbox_follow
        calls.append(ToolCall("get_approval_inbox", args))
        return calls
    if re.search(
        r"\b(approval inbox|pending approvals?|awaiting my approval|sign[- ]?off|sitting in my inbox|"
        r"leave asks?|waiting on (?:a )?decision from me|"
        r"team(?:'s)? pending (?:leave )?requests?)\b",
        text,
    ):
        calls.append(ToolCall("get_approval_inbox", {"status": "pending"}))
        return calls

    from app.ai.context import hr_corrective_action_question

    transfer_focus = org_transfer_followup_focus(message)
    if transfer_focus is None and hr_corrective_action_question(message):
        transfer_focus = "remediation"
    # Do not steal manager approval-inbox urgency/impact follow-ups into routing investigation.
    if inbox_follow in {"urgency", "why_urgent", "approve_impact", "reject_impact", "prepare_approve"}:
        transfer_focus = None
    if (
        intent == "org_transfer"
        or routing_discrepancy_question(text)
        or transfer_focus is not None
        or hr_corrective_action_question(text)
        or (
            re.search(
                r"\b(previous manager|approval still|routing|discrepan|transfer|department(?:s)? (?:and|&) (?:their )?reporting)\b",
                text,
            )
            and inbox_follow is None
        )
    ):
        args: dict = {"org_wide": True}
        if transfer_focus:
            args["focus"] = transfer_focus
        calls.append(ToolCall("investigate_employee_discrepancy", args))
        return calls

    if intent in {"org_attendance", "workforce_analytics", "hr_metrics"} or (
        role == "hr_admin"
        and re.search(r"\b(absent|absence|absenteeism|attendance|department|trend|pattern|compare|october|september)\b", text)
        and not routing_discrepancy_question(text)
    ):
        # Managers never plan organization attendance tools; team remaps happen in classify/authorize.
        if role == "manager":
            calls.append(ToolCall("department_attendance_analytics", {"query": message, "months": 3}))
            return calls
        comparison_ask = bool(
            re.search(r"\b(compare|versus|vs\.?|increase|decrease|contributed|change|what about)\b", text)
            or re.search(r"\b(september|october|november).{0,40}\b(september|october|november)\b", text)
        )
        if comparison_ask:
            calls.append(ToolCall("compare_date_ranges", {"query": message}))
        elif intent == "workforce_analytics" or (
            workforce_followup_question(message)
            and not re.search(r"\bcontributed most\b", text)
        ) or re.search(r"\b(pattern|recommend|concern|operational)\b", text):
            calls.append(ToolCall("department_attendance_analytics", {"query": message, "months": 3}))
        else:
            calls.append(ToolCall("department_attendance_analytics", {"query": message}))
        return calls

    if intent in {"policy_question", "leave_policy", "general_hr"} or re.search(
        r"\b(policy|policies|handbook|notice period)\b", text
    ):
        calls.append(ToolCall("search_hr_policies", {"query": message}))
        return calls

    if intent in {"team_roster", "team_analytics"} and role in {"manager", "hr_admin"}:
        # Coverage analysis is the assessment-relevant team tool; roster/analytics stay in legacy nodes
        # when this planner is not selected — still offer inbox/coverage when asked.
        if re.search(r"\b(coverage|overlap|leave)\b", text):
            calls.append(ToolCall("analyze_team_coverage", {"query": message}))
            return calls

    if intent == "holiday_query":
        if re.search(r"\b(notice|policy|rulebook|handbook|advance)\b", text):
            calls.append(ToolCall("search_hr_policies", {"query": message}))
            return calls
        calls.append(ToolCall("create_leave_draft", {"focus": "holidays"}))
        return calls

    # Default: no backend tool — conversational/policy path only.
    return calls


def llm_tool_plan(
    *,
    message: str,
    role: str,
    intent: str | None,
    employee_context: dict | None,
    conversation_history: list | None,
    citations: list | None,
    record_context: dict | None,
) -> tuple[list[ToolCall], str]:
    """Ask the LLM which backend tools to run. Returns (calls, mode)."""
    client = LLMClient()
    if not client.enabled:
        return (
            deterministic_tool_plan(
                message=message,
                role=role,
                intent=intent,
                record_context=record_context,
            ),
            "deterministic_planner",
        )

    catalog = [
        {
            "name": spec.name,
            "description": spec.description,
            "required_roles": sorted(spec.required_roles),
            "required_args": list(spec.required_args),
            "mutating": spec.mutating,
            "parameters": spec.parameters,
        }
        for spec in TOOL_CATALOG.values()
        if role in spec.required_roles
    ]
    system = (
        "You are Helios tool planner for an enterprise HR copilot. "
        "Choose zero or more backend tools required to answer the user. "
        "Return JSON only: "
        '{"tool_calls":[{"name":"tool_name","arguments":{...}}],"needs_clarification":false,'
        '"clarification_question":null,"reasoning_summary":"short"}. '
        "Use only tools from the provided catalog. "
        "Do not invent SQL. Do not invent HR numbers. "
        "If dates, request ids, or rejection reasons are missing for a mutating tool, "
        "either omit that tool and set needs_clarification, or include the tool with partial args "
        "so the backend can ask for missing fields. "
        "Multiple tools are allowed when the question genuinely needs them "
        "(for example balance + feasibility + policy). "
        "Prefer compare_date_ranges for period comparisons and department_attendance_analytics for org trends. "
        "Prefer investigate_employee_discrepancy for transfer/manager/approval routing issues. "
        f"The signed-in role is {role}. Only select tools allowed for that role."
    )
    payload = {
        "message": message,
        "intent_hint": intent,
        "role": role,
        "employee": {
            key: (employee_context or {}).get(key)
            for key in ("full_name", "job_title", "department", "location", "employee_number")
            if (employee_context or {}).get(key) is not None
        },
        "conversation_history": (conversation_history or [])[-8:],
        "record_context": record_context or {},
        "policy_citations_already_retrieved": [
            {"policy_id": item.get("policy_id"), "title": item.get("title")}
            for item in (citations or [])[:6]
        ],
        "available_tools": catalog,
    }
    try:
        parsed = client.complete_json(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            temperature=0.0,
            max_tokens=900,
            raise_on_error=True,
        )
    except (TransientLLMError, PermanentLLMError) as exc:
        logger.warning("LLM tool planning failed (%s); using deterministic planner", exc)
        return (
            deterministic_tool_plan(
                message=message,
                role=role,
                intent=intent,
                record_context=record_context,
            ),
            "deterministic_planner_fallback",
        )

    raw_calls = parsed.get("tool_calls") or parsed.get("tools") or []
    calls: list[ToolCall] = []
    if isinstance(raw_calls, list):
        for index, item in enumerate(raw_calls):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("tool") or "").strip()
            if not name or name not in TOOL_CATALOG:
                continue
            arguments = item.get("arguments") or item.get("args") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(ToolCall(name=name, arguments=arguments, call_id=f"llm-{index}"))

    if not calls and not parsed.get("needs_clarification"):
        # Empty plan from the model — fall back so operational asks still execute.
        fallback = deterministic_tool_plan(
            message=message,
            role=role,
            intent=intent,
            record_context=record_context,
        )
        if fallback:
            return fallback, "deterministic_planner_empty_llm"
    return calls, "llm_planner"


def run_tool_orchestration(
    db: Session,
    user: User,
    *,
    message: str,
    intent: str | None,
    employee_context: dict | None = None,
    conversation_history: list | None = None,
    citations: list | None = None,
    record_context: dict | None = None,
    leave_focus: str | None = None,
    ip: str | None = None,
    prefer_llm: bool = True,
) -> dict[str, Any]:
    """Select and execute tools; return state patch + traces."""
    if prefer_llm:
        calls, planner_mode = llm_tool_plan(
            message=message,
            role=user.role,
            intent=intent,
            employee_context=employee_context,
            conversation_history=conversation_history,
            citations=citations,
            record_context=record_context,
        )
    else:
        calls = deterministic_tool_plan(
            message=message,
            role=user.role,
            intent=intent,
            leave_focus=leave_focus,
            record_context=record_context,
        )
        planner_mode = "deterministic_planner"

    results: list[ToolResult] = []
    for call in calls[:8]:
        # Inject leave focus for drafts when the planner omitted it.
        if call.name == "create_leave_draft" and leave_focus and not call.arguments.get("focus"):
            call.arguments = {**call.arguments, "focus": leave_focus}
        # Reuse the selected pending request when the planner omitted the id.
        if (
            call.name in {"execute_approval", "execute_rejection"}
            and call.arguments.get("request_id") in (None, "")
            and (record_context or {}).get("selected_request_id") is not None
        ):
            call.arguments = {
                **call.arguments,
                "request_id": int(record_context["selected_request_id"]),
            }
        elif (
            call.name in {"execute_approval", "execute_rejection"}
            and call.arguments.get("request_id") in (None, "")
            and (record_context or {}).get("request_id") is not None
        ):
            call.arguments = {
                **call.arguments,
                "request_id": int(record_context["request_id"]),
            }
        result = execute_tool(
            db,
            user,
            call,
            message=message,
            inherited=record_context,
            ip=ip,
            confirmed=False,
        )
        results.append(result)

    patch = merge_tool_results_into_state(results)
    patch["orchestration_mode"] = "tool_llm" if planner_mode.startswith("llm") else "tool_deterministic"
    patch["tool_plan"] = {
        "mode": planner_mode,
        "selected": [{"name": c.name, "arguments": c.arguments} for c in calls],
    }
    patch["tool_trace"] = [
        {"tool": "plan_tools", "status": "ok", "summary": f"{planner_mode}:{len(calls)} tool(s)"},
        *[item.as_trace() for item in results],
    ]
    if not results and intent in {"policy_question", "leave_policy", "general_hr"}:
        # Ensure policy turns still retrieve knowledge when the planner chose nothing.
        result = execute_tool(
            db,
            user,
            ToolCall("search_hr_policies", {"query": message}),
            message=message,
            inherited=record_context,
            ip=ip,
        )
        results = [result]
        patch = merge_tool_results_into_state(results)
        patch["orchestration_mode"] = "tool_deterministic"
        patch["tool_plan"] = {"mode": "policy_default", "selected": [{"name": "search_hr_policies", "arguments": {"query": message}}]}
        patch["tool_trace"] = [
            {"tool": "plan_tools", "status": "ok", "summary": "policy_default:1 tool(s)"},
            result.as_trace(),
        ]
    return {
        "results": results,
        "state_patch": patch,
        "planner_mode": planner_mode,
        "selected_tools": [c.name for c in calls],
    }


def schemas_for_prompt() -> list[dict]:
    return openai_tool_schemas()
