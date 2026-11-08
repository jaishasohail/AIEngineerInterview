"""LangGraph orchestrator.

A deterministic rule wins over the model. The model classifies only when no rule matches,
and it may polish a briefing. Tools, balances, authorization, and record changes run in deterministic code.
Human approval is a real interrupt: the execute node is unreachable until resume.
"""

from __future__ import annotations

import operator
import re
from datetime import date, timedelta
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt

from app.ai.context import (
    ambiguous_team_or_org_scope,
    classify_with_model,
    continuation_intent,
    find_mentioned,
    get_ctx,
    hypothetical_remaining_days,
    is_clarification_reply,
    leave_plan_followup,
    load_actor,
    manager_team_ops_question,
    organization_intent,
    parse_approval_decision,
    parse_business_days,
    post_approval_question,
    profile_for,
    remote_policy_question,
    requested_leave_days,
    rule_intent,
)
from app.ai.guard import (
    broad_private_data_request,
    detect_injection,
    hard_security_violation,
    organization_wide_request,
    other_employee_private_request,
    out_of_hierarchy_request,
    outside_authorized_scope_request,
    sensitive_kinds,
    unsupported_destructive_request,
    wants_bulk_export,
)
from app.ai.llm import polish
from app.ai.narrate import narrate, result_message
from app.ai.query_spec import intent_from_spec, interpret_and_validate
from app.ai.security import assess_security
from app.ai.retrieval import conflict_groups_in_force, retrieve
from app.audit import write_audit
from app.access import AuthorizationError
from app.services.actions import decide_approval, file_routing_correction, submit_leave
from app.services.analytics_engine import absence_analytics
from app.services.hr_ops import run_hr_report
from app.services.leave_engine import (
    analyze_team,
    balance_payload,
    get_balance,
    investigate_routing,
    plan_leave,
    post_approval_facts,
    team_roster_report,
)
from app.services.team_ops import team_analytics_report
from app.models import Approval, Department, Employee, LeaveRequest, PolicyChunk, PolicyDocument
from app.services.records import (
    WindowError,
    attendance_report,
    cancellation_report,
    holiday_report,
    leave_balance_report,
    leave_status_report,
    parse_leave_dates,
    profile_report,
)

_REQUEST_INTENTS = {"leave_planning", "leave_request"}
from app.clock import add_business_days, org_today, parse_day
from app.services.holidays import holidays_for_employee

TRANSIENT = RetryPolicy(max_attempts=2)


class HRState(TypedDict, total=False):
    message: str
    role: str
    intent: str
    orchestration_mode: str
    injection_flag: bool
    authorized: bool
    denial_reason: str
    security_decision: dict | None
    clarify: bool
    clarification: str
    business_days: int
    target_employee_ids: list[int]
    sensitive_kinds: list[str]
    employee_context: dict
    analysis: dict
    analytics: dict
    retrieval: dict
    citations: list
    retrieval_confidence: str
    policy_note: str
    pending_action: dict | None
    confirmed: bool | None
    selected_option: str
    execution_result: dict
    record_context: dict | None
    leave_focus: str | None
    pending_clarification: dict | None
    response: str
    decision_summary: str
    tool_trace: Annotated[list, operator.add]
    workflow: Annotated[list, operator.add]


def _step(node: str, label: str, status: str, detail: str = "") -> dict:
    return {"node": node, "label": label, "status": status, "detail": detail}


def _tool(name: str, status: str, summary: str) -> dict:
    return {"tool": name, "status": status, "summary": summary}


def classify_intent(state: HRState) -> dict:
    ctx = get_ctx()
    user, employee, _department = load_actor(ctx.db, ctx.user_id)
    message = state.get("message", "")
    injections = detect_injection(message)
    kinds = sensitive_kinds(message)
    mentioned = find_mentioned(ctx.db, message, employee.id)
    bulk = wants_bulk_export(message)
    ruled = rule_intent(message, mentioned_other=bool(mentioned), kinds=kinds, bulk=bulk)
    mode = "deterministic"
    leave_focus = None
    # The compositional security layer runs on the whole message before any intent is chosen,
    # so a malicious instruction embedded in an otherwise ordinary question denies the whole
    # message instead of letting the benign half pick the intent. It never grants access.
    security = assess_security(message, role=user.role, mentioned_other=bool(mentioned))
    security_audit = security.as_audit() if security.denied else None
    # Hard security probes always win. Broad org analytics are role-gated: HR Admin may ask
    # organization-wide questions; other roles are denied before retrieval.
    hr_employee_delete = bool(
        user.role == "hr_admin"
        and re.search(r"\b(delete|remove)\b", message, re.IGNORECASE)
        and re.search(r"\bemployee record\b", message, re.IGNORECASE)
    )
    if security.denied:
        intent = security.intent
        violation = True
    elif unsupported_destructive_request(message) and not hr_employee_delete:
        intent = "unsupported_mutation"
        violation = True
    elif out_of_hierarchy_request(message) and user.role != "hr_admin":
        intent = "unauthorized_access"
        violation = True
    elif hard_security_violation(message):
        intent = "unauthorized_access"
        violation = True
    elif other_employee_private_request(message) and user.role != "hr_admin":
        intent = "unauthorized_access"
        violation = True
    elif mentioned and user.role == "employee" and re.search(
        r"\b(medical|sick leave|leave history|leave balance|salary|compensation|phone)\b",
        message,
        re.IGNORECASE,
    ):
        intent = "unauthorized_access"
        violation = True
    elif broad_private_data_request(message) and user.role != "hr_admin":
        intent = "unauthorized_access"
        violation = True
    else:
        violation = False
        intent = ruled
        # Concrete leave-request approve/reject uses the same decide_approval service as REST.
        approval_ask = parse_approval_decision(message)
        if approval_ask:
            intent = "approval_decision"
            mode = "deterministic"
        inherited = state.get("record_context") or {}
        same_actor = inherited.get("user_id") in {None, user.id} and inherited.get("employee_id") in {None, employee.id}
        query_spec = interpret_and_validate(message, role=user.role, inherited=inherited)
        # Structured interpretation fills gaps before LLM fallback. It may also replace weak
        # catch-all intents (policy/general_hr) when the QuerySpec has a confident domain —
        # never for authorization or SQL, and never when security already denied.
        hinted = intent_from_spec(query_spec, role=user.role)
        if query_spec.domain == "unsupported" and query_spec.unsupported:
            # Keep a domain-aligned intent so narrate can disclose the gap without substituting metrics.
            intent = hinted or "attendance_query"
            mode = "deterministic"
        elif hinted and (
            not intent
            or (
                intent in {"general_hr", "out_of_scope"}
                and query_spec.domain
                and query_spec.domain not in {"unknown", ""}
            )
        ):
            # Do not let QuerySpec steal a confident policy_question rule into leave_balance
            # or directory. Policy-conflict stays on policy_question when hinted that way.
            if query_spec.metric == "policy_conflict":
                intent = "policy_question"
            else:
                intent = hinted
            mode = "deterministic"
        elif intent == "policy_question" and query_spec.metric == "policy_conflict":
            mode = "deterministic"
        elif hinted and intent == "hr_metrics" and query_spec.intent_hint == "hr_metrics":
            mode = "deterministic"
        # Leave-plan multi-turn follow-ups keep the planned window even when a weak
        # policy/holiday rule would otherwise steal the turn.
        leave_follow = leave_plan_followup(message, inherited) if same_actor else None
        if leave_follow and not violation:
            intent, leave_focus = leave_follow
            mode = "deterministic"
        # Explicit organization-wide phrasing must not inherit a prior self/team subject.
        if (
            leave_follow is None
            and not ruled
            and same_actor
            and not mentioned
            and not organization_wide_request(message)
        ):
            continued = continuation_intent(message, inherited)
            if continued:
                intent = continued
        # Short period/metric switches keep the prior attendance/team/directory subject even when
        # a weak rule/LLM would pivot to policy.
        if (
            same_actor
            and inherited.get("subject") in {"attendance", "team_analytics", "team_roster", "employee_directory", "leave_balance"}
            and re.search(
                r"\b(switch(?:\s+that)?|instead|now compare|those two|narrow that|focus that|same month|that month)\b",
                message,
                re.IGNORECASE,
            )
            and not organization_wide_request(message)
        ):
            continued = continuation_intent(message, inherited) or {
                "attendance": "attendance_query",
                "team_analytics": "team_analytics",
                "team_roster": "team_analytics",
                "employee_directory": "employee_directory",
                "leave_balance": "leave_balance",
            }.get(str(inherited.get("subject")))
            if continued:
                intent = continued
                mode = "deterministic"
        # Managers asking about their own reporting line get team analytics deterministically,
        # so a month/status/department filter never depends on the model or falls to self records.
        if not intent and user.role == "manager" and manager_team_ops_question(message):
            intent = "team_analytics"
        if not intent and not ruled and not injections:
            modeled = classify_with_model(message, user.role)
            if modeled:
                intent = modeled
                mode = "llm"
        # HR may ask organization salary totals/averages. Bulk "export/records/all salaries"
        # dumps remain confidential_lookup and stay denied in authorize.
        org_ruled = organization_intent(message)
        if (
            user.role == "hr_admin"
            and org_ruled == "compensation_query"
            and not re.search(
                r"\b(export|salary records|all salaries|payroll export|entire payroll)\b",
                message,
                re.IGNORECASE,
            )
        ):
            intent = "compensation_query"
            mode = "deterministic"
        elif ruled == "confidential_lookup" or bulk:
            intent = "confidential_lookup"
        if injections and intent is None:
            intent = "general_hr"
        intent = intent or "general_hr"
        if intent == "leave_policy" and (
            not re.search(r"\bnotice\b", message, re.IGNORECASE) or re.search(r"\bresign", message, re.IGNORECASE)
        ):
            intent = "policy_question"
        if intent == "org_transfer" and re.search(r"\b(policy|policies)\b", message, re.IGNORECASE):
            intent = "policy_question"
        # Managers get team-scoped analytics for team ops phrasing; HR keeps org_* intents.
        # Do NOT remap team_conflict — overlap/coverage analysis must stay on analyze_team.
        if (
            user.role == "manager"
            and manager_team_ops_question(message)
            and intent
            in {
                "org_attendance",
                "org_leave",
                "employee_directory",
                "workforce_analytics",
                "general_hr",
                "leave_request_status",
                "hr_metrics",
                "attendance_query",
                "leave_balance",
                "employee_profile",
                "out_of_scope",
                "confidential_lookup",
            }
            and not re.search(
                r"\b(overlap|overlapping|conflict|coverage|staffing|understaff)\b",
                message,
                re.IGNORECASE,
            )
        ):
            intent = "team_analytics"
            mode = "deterministic"
        # Multi-turn manager follow-ups that refer to prior direct reports ("them", "those",
        # "that department", "that group") stay on team analytics even when the wording
        # resembles organization-wide attendance.
        if (
            user.role == "manager"
            and same_actor
            and inherited.get("subject") in {"team_roster", "team_analytics"}
            and not organization_wide_request(message)
            and re.search(
                r"\b(them|those|they|their|my (?:direct )?reports?|my team|"
                r"that (?:department|group|cohort|team)|in that department|employees in that)\b",
                message,
                re.IGNORECASE,
            )
            and re.search(
                r"\b(remote(?:ly)?|attendance|absent|present|late|sick|leave|department|departments?|"
                r"patterns?|september|october|november|august)\b",
                message,
                re.IGNORECASE,
            )
            and intent
            in {
                "org_attendance",
                "org_leave",
                "employee_directory",
                "hr_metrics",
                "general_hr",
                "attendance_query",
                "out_of_scope",
                "policy_question",
            }
        ):
            intent = "team_analytics"
            mode = "deterministic"
        # Peer-manager team expansion is unauthorized for managers (never policy).
        if user.role == "manager" and re.search(
            r"\b(another manager(?:'s)?|other manager(?:'s)?|someone else(?:'s)? team|"
            r"employees from another manager|treat (?:their|his|her) employees as my|"
            r"peer manager(?:'s)? (?:team|reports?)|manager peer(?:'s)? team|"
            r"include .{0,40}another manager|expand .{0,30}(?:to|into) .{0,30}(?:other|another) manager|"
            r"access .{0,20}(?:another|other) manager)\b",
            message,
            re.IGNORECASE,
        ):
            intent = "unauthorized_access"
            mode = "deterministic"
        # Expanding a prior team-scoped thread to the whole company is unauthorized for managers
        # when the ask is privilege expansion (dashboard/cover every department). Clear org_*
        # domain asks (pending leave across the org, org attendance) keep their org_* intent so
        # authorize can deny with the HR-admin-only message.
        if (
            user.role == "manager"
            and same_actor
            and inherited.get("subject") in {"team_analytics", "team_roster"}
            and organization_wide_request(message)
        ):
            org_follow = organization_intent(message)
            expand_cue = bool(
                re.search(
                    r"\b(expand|widen|broaden|extend|cover|include|dashboard|every department|"
                    r"all departments|same summary|same report)\b",
                    message,
                    re.IGNORECASE,
                )
            )
            if org_follow in {
                "org_leave",
                "org_attendance",
                "employee_directory",
                "compensation_query",
                "hr_metrics",
            } and not expand_cue:
                intent = org_follow
                mode = "deterministic"
            else:
                intent = "unauthorized_access"
                mode = "deterministic"
                violation = True
        # Department breakdown of a prior team remote/attendance answer stays on team analytics
        # even when the message says "remote-work" (which otherwise looks like policy).
        if (
            user.role == "manager"
            and same_actor
            and inherited.get("subject") in {"team_roster", "team_analytics"}
            and not organization_wide_request(message)
            and re.search(r"\b(break(?:\s+\w+)?\s+down|by department|department breakdown)\b", message, re.IGNORECASE)
            and intent
            in {
                "policy_question",
                "general_hr",
                "out_of_scope",
                "hr_metrics",
                "org_attendance",
                "employee_directory",
            }
        ):
            intent = "team_analytics"
            mode = "deterministic"
        # Explicit org-wide requests stay on org_* so authorize can deny non-HR roles.
        # Manager team-scope expansion is unauthorized_access (never soft team_analytics allow).
        if (
            user.role == "manager"
            and organization_wide_request(message)
            and intent in {"team_analytics", "team_roster", "attendance_query", "general_hr", "out_of_scope"}
        ):
            intent = "unauthorized_access"
            mode = "deterministic"
            violation = True
        elif organization_wide_request(message) and intent in {
            "leave_request_status",
            "attendance_query",
            "leave_balance",
            "team_analytics",
            "team_roster",
            "general_hr",
        }:
            if re.search(r"\b(leave|pending|approval|request)\b", message, re.IGNORECASE):
                intent = "org_leave"
            elif re.search(r"\b(attendance|absent|remote|present)\b", message, re.IGNORECASE):
                intent = "org_attendance"
            else:
                intent = "employee_directory"
            mode = "deterministic"
        # Employee self attendance without org cues must remain attendance_query.
        # Do not steal overtime/policy questions that merely mention "hours" (premium thresholds).
        if (
            user.role == "employee"
            and intent in {"org_attendance", "general_hr", "out_of_scope", "policy_question"}
            and re.search(r"\b(attendance|remote|present|hours|absent)\b", message, re.IGNORECASE)
            and not organization_wide_request(message)
            and not re.search(
                r"\b(all employees|every employee|organization|department|everyone|company-wide)\b",
                message,
                re.IGNORECASE,
            )
            and not re.search(
                r"\b(overtime|premium|policy|policies|authorize|authorise|approval rule|"
                r"working hours|core hours|official working)\b",
                message,
                re.IGNORECASE,
            )
        ):
            intent = "attendance_query"
            mode = "deterministic"
        # HR Admin attendance questions without first-person scope are organization attendance.
        if (
            user.role == "hr_admin"
            and intent in {"attendance_query", "general_hr", "out_of_scope"}
            and re.search(
                r"\b(attendance|absent|remote|present|late|working hours|hours were recorded|recorded .{0,12}hours)\b",
                message,
                re.IGNORECASE,
            )
            and not re.search(r"\b(my|i|i'm|i have|do i|did i)\b", message, re.IGNORECASE)
            and not re.search(r"\b(my team|direct reports?)\b", message, re.IGNORECASE)
        ):
            intent = "org_attendance"
            mode = "deterministic"
        # Keep HR leave-utilization / attendance-analytics threads on workforce_analytics.
        inherited_ctx = state.get("record_context") or {}
        if (
            user.role == "hr_admin"
            and inherited_ctx.get("subject") in {"workforce_analytics", "org_attendance"}
            and continuation_intent(message, inherited_ctx) == "workforce_analytics"
            and intent
            in {
                "org_leave",
                "org_attendance",
                "hr_metrics",
                "employee_directory",
                "general_hr",
                "out_of_scope",
                "workforce_analytics",
                "team_analytics",
                "policy_question",
            }
        ):
            intent = "workforce_analytics"
            mode = "deterministic"
        # HR Admin org-wide attendance analytics must never fall through to direct-report team_analytics.
        if (
            user.role == "hr_admin"
            and intent == "team_analytics"
            and not re.search(r"\b(my team|direct reports?|my reports?)\b", message, re.IGNORECASE)
            and re.search(
                r"\b(unusual|anomal|pattern|investigation|absence|attendance|department|"
                r"recommend|office|location|compare|months?)\b",
                message,
                re.IGNORECASE,
            )
        ):
            intent = "workforce_analytics"
            mode = "deterministic"
        # Standalone HR analytics phrasing without a prior thread still stays organization-wide.
        if (
            user.role == "hr_admin"
            and intent in {"general_hr", "out_of_scope", "policy_question"}
            and re.search(
                r"\b(unusual attendance|attendance patterns?|further investigation|"
                r"recommend(?:ations?)? to hr|actions? would you recommend|"
                r"explain the data behind|data behind (?:your |the )?recommend|"
                r"largest increase|compare (?:the )?(?:last|past|previous) (?:\w+ )?months?|"
                r"highest absence rates?)\b",
                message,
                re.IGNORECASE,
            )
            and not re.search(r"\b(my team|direct reports?)\b", message, re.IGNORECASE)
        ):
            intent = "workforce_analytics"
            mode = "deterministic"
        # Preserve a prior directory hire-date filter when HR asks about tenure of that cohort.
        if (
            user.role == "hr_admin"
            and same_actor
            and inherited_ctx.get("subject") == "employee_directory"
            and inherited_ctx.get("hire_filter")
            and re.search(r"\b(tenure|average tenure|mean tenure)\b", message, re.IGNORECASE)
            and re.search(
                r"\b(those|them|that group|that cohort|among those|of those|same employees)\b",
                message,
                re.IGNORECASE,
            )
            and intent in {"hr_metrics", "general_hr", "out_of_scope", "employee_directory"}
        ):
            intent = "employee_directory"
            mode = "deterministic"
        # Keep concrete approval decisions from being remapped to org HR actions or analytics.
        if approval_ask and not violation:
            intent = "approval_decision"
            mode = "deterministic"
    pending = state.get("pending_clarification")
    if pending and (
        pending.get("user_id") not in {None, user.id} or pending.get("intent") not in _REQUEST_INTENTS
    ):
        pending = None
    reply = is_clarification_reply(message)
    if reply and pending is None and not violation and ruled is None:
        intent = "general_hr"
        mode = "deterministic"
    days = parse_business_days(message)
    clarify = False
    clarification = ""
    new_pending = None
    if (
        pending
        and not violation
        and reply
        and rule_intent(message, mentioned_other=bool(mentioned), kinds=kinds, bulk=bulk) is None
    ):
        intent = pending.get("intent") or "general_hr"
        mode = "deterministic"
        text = message.strip().lower().rstrip(".!")
        if text in {"no", "nope"}:
            clarify = True
            clarification = "I will not plan leave from that reply. No leave request was created."
            days = None
        elif "business_days" in (pending.get("missing") or []) and days is None:
            clarify = True
            clarification = "How many business days should I plan?"
            new_pending = {
                "intent": intent,
                "missing": ["business_days"],
                "question": clarification,
                "user_id": user.id,
            }
        else:
            clarify = False
    elif (
        not violation
        and intent == "confidential_lookup"
        and not bulk
        and not mentioned
        and re.search(r"\b(my|i|me)\b", message.lower())
        and "salary" in kinds
        and not re.search(r"\b(manager|colleague|someone|everyone|department)\b", message, re.IGNORECASE)
    ):
        intent = "employee_profile"
    context = state.get("record_context") or {}
    span = parse_leave_dates(message)
    if (
        days is None
        and intent in _REQUEST_INTENTS
        and context.get("subject") == "leave_eligibility"
        and context.get("requested_days")
        and context.get("user_id") in {None, user.id}
        and re.search(r"\b(prepare|the request|it)\b", message.lower())
    ):
        days = int(context["requested_days"])
    # Inherit a previously planned leave window so prepare/review turns do not re-ask duration.
    inherited_plan = context if context.get("subject") == "leave_plan" else {}
    if (
        days is None
        and intent in _REQUEST_INTENTS
        and inherited_plan.get("business_days")
        and inherited_plan.get("user_id") in {None, user.id}
    ):
        days = int(inherited_plan["business_days"])
    has_inherited_window = bool(
        inherited_plan.get("start")
        and inherited_plan.get("end")
        and inherited_plan.get("user_id") in {None, user.id}
    )
    if not (pending and reply and not violation and new_pending is not None) and not (
        pending and reply and not violation and rule_intent(message, mentioned_other=bool(mentioned), kinds=kinds, bulk=bulk) is None
    ):
        if (
            intent in _REQUEST_INTENTS
            and days is None
            and not (span and span.get("start"))
            and not has_inherited_window
            and leave_focus not in {"draft_review", "confirmation_gate", "prepare", "business_days"}
            and not violation
        ):
            clarify = True
            clarification = "How many business days should I plan?"
            new_pending = {
                "intent": intent,
                "missing": ["business_days"],
                "question": clarification,
                "user_id": user.id,
            }
        elif intent == "general_hr" and len(message.split()) < 2 and not clarify:
            clarify = True
            clarification = "What do you want to do: plan leave, check a policy, review your team, or look at absence trends?"
        elif (
            not violation
            and not clarify
            and re.search(
                r"\b(those people|these people|for those people|for them)\b|"
                r"\bfor those\b(?!\s+(?:deductions?|records?|days?|hours?|balances?|requests?|dates?|periods?|windows?))",
                message,
                re.IGNORECASE,
            )
            and not (
                context.get("subject")
                in {"team_analytics", "team_roster", "employee_directory", "attendance", "leave_plan"}
            )
            and not mentioned
            # Manager team-ops phrasing that already names the cohort (remote + pending) is not a missing antecedent.
            and not (user.role == "manager" and manager_team_ops_question(message))
            and not re.search(
                r"\b(with remote|remote days|pending leave|direct reports?|my team|sick leave|deductions?|"
                r"leave request|those dates|that period)\b",
                message,
                re.IGNORECASE,
            )
        ):
            clarify = True
            clarification = (
                "Which people or group do you mean? Name the employees, team, department, or prior result you are referring to."
            )
            intent = "general_hr"
            new_pending = {
                "intent": "general_hr",
                "missing": ["referent"],
                "question": clarification,
                "user_id": user.id,
            }
        elif (
            not violation
            and not clarify
            and user.role == "manager"
            and ambiguous_team_or_org_scope(message)
        ):
            clarify = True
            clarification = (
                "Do you want absence trends for your direct reports only, or for the whole organization? "
                "Organization-wide absence trends require HR Admin access."
            )
            intent = "team_analytics"
            new_pending = {
                "intent": "team_analytics",
                "missing": ["scope"],
                "question": clarification,
                "user_id": user.id,
            }
    result = {
        "role": user.role,
        "intent": intent,
        "orchestration_mode": mode,
        "injection_flag": bool(injections),
        "business_days": days if days else (0 if clarify else 10),
        "target_employee_ids": [person.id for person in mentioned],
        "sensitive_kinds": sorted(kinds),
        "clarify": clarify,
        "clarification": clarification,
        "pending_clarification": new_pending,
        "leave_focus": leave_focus,
        "authorized": True,
        "security_decision": security_audit,
        "workflow": [_step("classify_intent", "Intent classification", "complete", intent)],
        "tool_trace": [_tool("classify_intent", "ok", intent)],
        "analysis": None,
        "citations": [],
        "retrieval": {},
        "analytics": {},
    }
    text = message.lower()
    continuing = intent in {
        "attendance_query",
        "holiday_query",
        "leave_eligibility",
        "leave_balance",
        "leave_request_status",
        "leave_planning",
        "leave_request",
        "policy_question",
        "employee_directory",
        "team_roster",
        "team_analytics",
        "team_conflict",
        "approval_decision",
        "org_attendance",
        "org_leave",
        "compensation_query",
        "hr_metrics",
        "workforce_analytics",
    } or (
        intent == "leave_balance" and re.search(r"\b(that|those|affect|impact)\b", text)
    ) or (
        intent in _REQUEST_INTENTS and re.search(r"\b(prepare|the request|it)\b", text)
    ) or (
        intent == "employee_profile" and re.search(r"\bapprov", text)
    ) or (
        (context.get("subject") == "leave_plan" or leave_focus)
        and intent in {"holiday_query", "leave_planning", "leave_request", "leave_balance"}
    ) or (
        context.get("subject") == "approval_decision"
        and re.search(r"\b(approve|reject|status|why|reason|it|that request)\b", text)
    )
    if not continuing:
        result["record_context"] = None
    return result


def authorize(state: HRState) -> dict:
    ctx = get_ctx()
    user, employee, _department = load_actor(ctx.db, ctx.user_id)
    intent = state.get("intent")
    targets = state.get("target_employee_ids") or []
    kinds = set(state.get("sensitive_kinds") or [])
    reason = ""
    allowed = True
    message = state.get("message", "")
    if state.get("clarify"):
        return {
            "authorized": True,
            "workflow": [_step("authorize", "Security and authorization", "complete", "clarification required before data access")],
        }
    # Defense in depth: the authorization node re-runs the security layer on the message itself,
    # so a denial never depends on what classify_intent or the LLM produced. Role exemptions are
    # inside the layer (HR Admin keeps organization scope) and nothing exempts bypass, secrets,
    # destructive SQL, cross-user memory, or bulk personal exports.
    security = assess_security(message, role=user.role, mentioned_other=bool(targets))
    security_audit = security.as_audit() if security.denied else None
    if security.denied:
        allowed = False
        reason = security.reason
        intent = security.intent
    elif intent == "unsupported_mutation":
        allowed = False
        reason = (
            "Destructive changes such as deleting attendance, leave, payroll, or employee records are not available "
            "through this assistant. No database mutation was performed. Use the authorized HR correction process instead."
        )
    elif (
        hard_security_violation(message)
        or out_of_hierarchy_request(message)
        or other_employee_private_request(message)
        or intent == "unauthorized_access"
    ) and not (
        user.role == "hr_admin"
        and intent
        in {
            "org_leave",
            "org_attendance",
            "employee_directory",
            "compensation_query",
            "hr_metrics",
            "workforce_analytics",
        }
        and not hard_security_violation(message)
        and not unsupported_destructive_request(message)
    ):
        # Jailbreak / out-of-hierarchy / private-other probes are denied before retrieval.
        allowed = False
        if out_of_hierarchy_request(message) or outside_authorized_scope_request(message):
            reason = (
                "You do not have permission to access attendance or employee records outside your authorized "
                "reporting scope. The assistant will not substitute unrelated personal data for that request."
            )
        elif other_employee_private_request(message):
            reason = (
                "You are not authorized to view another employee's medical, sick leave, leave balance, "
                "or other private HR details. Access to another employee's leave records is limited to HR "
                "administrators. This is not allowed. The assistant will not substitute policy text or your "
                "own records for that request."
            )
        else:
            reason = (
                "This request asks to bypass access rules or to reveal other employees' records, prompts, or credentials. "
                "This is not allowed. The assistant will not do that."
            )
    elif intent == "confidential_lookup":
        if wants_bulk_export(state.get("message", "")):
            allowed = False
            reason = "Bulk compensation and performance exports are blocked in the assistant, including for HR administrators."
        elif re.search(r"\bemployee(?:\s*id|_id)?\s*[:=#]?\s*\d+", state.get("message", ""), re.IGNORECASE):
            allowed = False
            reason = "This assistant uses the signed-in employee record. It does not accept an employee id from the message."
        elif re.search(r"\bmanager", state.get("message", ""), re.IGNORECASE) and "salary" in kinds:
            allowed = False
            reason = "You are not allowed to view your manager's salary."
        elif not targets:
            allowed = False
            reason = "Name the employee whose compensation or performance record you want. This is not a bulk export, and the assistant does not guess the person."
        else:
            for target_id in targets:
                subject = ctx.db.get(type(employee), target_id)
                from app.access import assert_sensitive_access

                try:
                    assert_sensitive_access(ctx.db, user, employee, subject, kinds or {"salary", "performance"})
                except Exception as exc:  # AuthorizationError
                    allowed = False
                    reason = str(exc)
                    break
    elif intent == "approval_decision" and user.role not in {"manager", "hr_admin", "employee"}:
        allowed = False
        reason = "Only signed-in employees, managers, and HR administrators can attempt approval decisions."
    elif intent == "team_conflict" and user.role not in {"manager", "hr_admin"}:
        allowed = False
        reason = "Team staffing analysis is limited to managers and HR administrators."
    elif intent == "team_roster" and user.role not in {"manager", "hr_admin"}:
        allowed = False
        reason = "Direct-report and team roster questions are limited to managers and HR administrators."
    elif intent == "team_analytics" and user.role not in {"manager", "hr_admin"}:
        allowed = False
        reason = "Team attendance and leave analytics are limited to managers and HR administrators."
    elif intent in {"team_analytics", "team_roster"} and targets and user.role == "manager" and any(
        (ctx.db.get(type(employee), target_id) is None or ctx.db.get(type(employee), target_id).manager_id != employee.id)
        for target_id in targets
    ):
        # A named person in a team question must be one of the manager's own direct reports.
        # Otherwise the answer would either leak that person's record or quietly substitute the team's.
        allowed = False
        reason = (
            "You do not have permission to access attendance or employee records outside your authorized "
            "reporting scope. The named employee is not one of your direct reports, and the assistant will not "
            "substitute your team's data for that request."
        )
    elif intent in _REQUEST_INTENTS and targets and user.role != "hr_admin":
        allowed = False
        reason = "Leave planning uses your own employee record from the signed-in session."
    elif intent in {
        "attendance_query",
        "holiday_query",
        "leave_balance",
        "leave_eligibility",
        "leave_policy",
        "leave_request_status",
        "employee_profile",
        "leave_cancellation",
    } and (
        targets or re.search(r"\bemployee(?:\s*id|_id)?\s*[:=#]?\s*\d+", state.get("message", ""), re.IGNORECASE)
    ):
        allowed = False
        reason = "This assistant uses the signed-in employee record. It does not accept another person or an employee id from the message."
    elif intent == "org_transfer" and targets and user.role == "employee":
        allowed_ids = {employee.manager_id, employee.approval_manager_id}
        if any(target_id not in allowed_ids for target_id in targets):
            allowed = False
            reason = "You can investigate your own reporting route, not another employee's."
    elif intent in {
        "employee_directory",
        "org_attendance",
        "org_leave",
        "compensation_query",
        "hr_metrics",
        "hr_action",
    } and user.role != "hr_admin":
        allowed = False
        reason = "Organization-wide employee, attendance, leave, and compensation questions are limited to HR administrators."
    if not allowed:
        write_audit(
            ctx.db,
            actor_user_id=user.id,
            action="assistant_access",
            resource_type="assistant",
            resource_id=intent,
            outcome="denied",
            details={
                "intent": intent,
                "reason": reason,
                "target_employee_ids": targets,
                "fields": sorted(kinds),
                "injection": bool(state.get("injection_flag")),
                "security_categories": (security_audit or {}).get("categories", []),
            },
            ip_address=ctx.ip,
        )
        denied_detail = reason
        if security_audit:
            denied_detail = f"{reason} [categories: {', '.join(security_audit['categories'])}]"
        return {
            "authorized": False,
            "intent": intent,
            "denial_reason": reason,
            "security_decision": security_audit,
            "analysis": None,
            "citations": [],
            "retrieval": {},
            "analytics": {},
            "workflow": [_step("authorize", "Security and authorization", "denied", denied_detail[:300])],
            "tool_trace": [_tool("authorize", "denied", reason[:300])],
        }
    if intent == "workforce_analytics" and user.role == "employee":
        write_audit(
            ctx.db,
            actor_user_id=user.id,
            action="assistant_access",
            resource_type="analytics",
            resource_id="self",
            outcome="allowed",
            details={"scope": "self", "note": "organization-wide analytics refused"},
            ip_address=ctx.ip,
            commit=True,
        )
    return {
        "authorized": True,
        "denial_reason": "",
        "workflow": [_step("authorize", "Security and authorization", "complete", intent or "")],
        "tool_trace": [_tool("authorize", "allowed", intent or "")],
    }


def employee_context(state: HRState) -> dict:
    ctx = get_ctx()
    user, employee, _department = load_actor(ctx.db, ctx.user_id)
    target = employee
    if state.get("intent") == "confidential_lookup" and state.get("target_employee_ids"):
        loaded = ctx.db.get(type(employee), state["target_employee_ids"][0])
        if loaded is not None:
            target = loaded
    private = state.get("intent") in {"self_service", "org_transfer", "leave_planning", "leave_request", "confidential_lookup"}
    profile = profile_for(ctx.db, user, employee, target, private=private)
    if state.get("intent") == "confidential_lookup":
        kinds = set(state.get("sensitive_kinds") or [])
        if "salary" not in kinds:
            profile.pop("salary_annual", None)
        if "performance" not in kinds:
            profile.pop("performance_rating", None)
        write_audit(
            ctx.db,
            actor_user_id=user.id,
            action="sensitive_read",
            resource_type="employee",
            resource_id=str(target.id),
            outcome="success",
            details={"fields": state.get("sensitive_kinds") or []},
            ip_address=ctx.ip,
        )
    extra: dict[str, Any] = {}
    if state.get("intent") == "self_service":
        extra["analysis"] = {"balance": balance_payload(get_balance(ctx.db, employee.id))}
    return {
        "employee_context": profile,
        **extra,
        "workflow": [_step("employee_context", "Employee context", "complete", profile.get("full_name", ""))],
        "tool_trace": [_tool("get_employee_context", "ok", profile.get("full_name", ""))],
    }


def approval_decision(state: HRState) -> dict:
    """Execute leave/approval decisions through the same service as POST /api/approvals/{id}/decide."""
    ctx = get_ctx()
    user, _employee, _department = load_actor(ctx.db, ctx.user_id)
    message = state.get("message") or ""
    parsed = parse_approval_decision(message) or {}
    decision = parsed.get("decision")
    request_id = parsed.get("request_id")
    reason = parsed.get("reason") or ""
    inherited = state.get("record_context") or {}
    if request_id is None and inherited.get("subject") == "approval_decision" and inherited.get("request_id"):
        request_id = int(inherited["request_id"])
    if decision is None and inherited.get("subject") == "approval_decision" and inherited.get("decision"):
        decision = inherited["decision"]
    if decision == "reject" and not reason:
        analysis = {
            "kind": "approval_decision",
            "status": "needs_reason",
            "decision": decision,
            "request_id": request_id,
            "message": "A rejection reason is required. Example: reject request 12 because coverage is insufficient.",
        }
        return {
            "analysis": analysis,
            "record_context": {
                "subject": "approval_decision",
                "decision": decision,
                "request_id": request_id,
                "user_id": user.id,
                "employee_id": user.employee_id,
            },
            "workflow": [_step("approval_decision", "Approval decision", "complete", "reason required")],
            "tool_trace": [_tool("decide_approval", "clarify", "rejection reason required")],
        }
    if decision is None or request_id is None:
        analysis = {
            "kind": "approval_decision",
            "status": "needs_clarification",
            "message": "Specify the leave request id and whether to approve or reject it. Example: approve leave request 12.",
        }
        return {
            "analysis": analysis,
            "workflow": [_step("approval_decision", "Approval decision", "complete", "clarification")],
            "tool_trace": [_tool("decide_approval", "clarify", "missing request id or decision")],
        }
    # Resolve leave request id → pending approval id (same authorization as REST).
    leave = ctx.db.get(LeaveRequest, int(request_id))
    approval = None
    if leave is not None:
        approval = (
            ctx.db.query(Approval)
            .filter(
                Approval.leave_request_id == leave.id,
                Approval.status == "pending",
            )
            .order_by(Approval.stage.asc(), Approval.id.asc())
            .first()
        )
    if approval is None:
        approval = ctx.db.get(Approval, int(request_id))
    if approval is None:
        analysis = {
            "kind": "approval_decision",
            "status": "not_found",
            "request_id": request_id,
            "decision": decision,
            "message": f"No pending approval was found for request {request_id}.",
        }
        return {
            "analysis": analysis,
            "record_context": {
                "subject": "approval_decision",
                "request_id": request_id,
                "decision": decision,
                "user_id": user.id,
                "employee_id": user.employee_id,
            },
            "workflow": [_step("approval_decision", "Approval decision", "complete", "not found")],
            "tool_trace": [_tool("decide_approval", "denied", "approval not found")],
        }
    try:
        result = decide_approval(
            ctx.db,
            user,
            approval.id,
            decision,
            reason,
            ip_address=getattr(ctx, "ip", None),
        )
        analysis = {
            "kind": "approval_decision",
            "status": result.get("status"),
            "decision": decision,
            "request_id": result.get("leave_request_id") or request_id,
            "approval_id": approval.id,
            "message": result.get("message") or f"Decision recorded as {result.get('status')}.",
            "reason": reason,
        }
        tool_status = "ok"
        summary = analysis["message"][:180]
    except AuthorizationError as exc:
        analysis = {
            "kind": "approval_decision",
            "status": "denied",
            "decision": decision,
            "request_id": request_id,
            "approval_id": approval.id,
            "message": str(exc),
            "reason": reason,
        }
        tool_status = "denied"
        summary = str(exc)[:180]
    return {
        "analysis": analysis,
        "record_context": {
            "subject": "approval_decision",
            "request_id": analysis.get("request_id"),
            "approval_id": analysis.get("approval_id"),
            "decision": decision,
            "status": analysis.get("status"),
            "user_id": user.id,
            "employee_id": user.employee_id,
        },
        "workflow": [_step("approval_decision", "Approval decision", "complete", analysis.get("status") or "")],
        "tool_trace": [_tool("decide_approval", tool_status, summary)],
    }


def leave_analysis(state: HRState) -> dict:
    ctx = get_ctx()
    user, employee, _department = load_actor(ctx.db, ctx.user_id)
    if state.get("intent") == "team_conflict":
        analysis = analyze_team(ctx.db, employee)
        summary = f"{len(analysis.get('requests') or [])} requests in {analysis.get('window', {}).get('label', 'the review window')}"
        tool = "analyze_team_conflicts"
        return {
            "analysis": analysis,
            "record_context": {
                "subject": "team_conflict",
                "user_id": user.id,
                "employee_id": employee.id,
                "focus": "overlap",
            },
            "workflow": [_step("leave_analysis", "Leave and attendance analysis", "complete", summary)],
            "tool_trace": [_tool(tool, "ok", summary)],
        }

    message = state.get("message") or ""
    parsed = parse_leave_dates(message)
    context = state.get("record_context") or {}
    focus = state.get("leave_focus") or ""
    leave_type = (parsed or {}).get("leave_type") or context.get("leave_type") or "annual"
    stated = parse_business_days(message)
    # "How many business days should I plan?" is a question about the current window, not a new duration.
    if focus == "business_days" or re.search(r"\bhow many business days should i plan\b", message.lower()):
        stated = None
    if stated is None and context.get("subject") == "leave_eligibility" and context.get("requested_days"):
        stated = int(context["requested_days"])
    if stated is None and context.get("subject") == "leave_plan" and context.get("business_days"):
        stated = int(context["business_days"])
    days = stated or int(state.get("business_days") or 10)
    start = parsed.get("start") if parsed else None
    end = parsed.get("end") if parsed else None
    if start is None and context.get("subject") == "leave_plan" and context.get("start"):
        start = date.fromisoformat(context["start"])
    if end is None and context.get("subject") == "leave_plan" and context.get("end") and stated is None:
        end = date.fromisoformat(context["end"])
    if start is not None and end is None:
        closed = holidays_for_employee(ctx.db, employee, start, start + timedelta(days=60)).dates
        end = add_business_days(start, max(days, 1), closed)
    # Duration-only change: keep start, recompute end from the new business-day count.
    if (
        stated is not None
        and start is not None
        and parsed is None
        and context.get("subject") == "leave_plan"
        and not re.search(r"\b(from|starting|until|through|to)\b", message.lower())
    ):
        closed = holidays_for_employee(ctx.db, employee, start, start + timedelta(days=60)).dates
        end = add_business_days(start, max(stated, 1), closed)
        days = stated
    analysis = plan_leave(
        ctx.db,
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
    summary = requested.get("label") or "leave plan"
    record_context = {
        "subject": "leave_plan",
        "intent": state.get("intent") or "leave_planning",
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
        "analysis": analysis,
        "record_context": record_context,
        "leave_focus": focus or "plan",
        "workflow": [_step("leave_analysis", "Leave and attendance analysis", "complete", summary)],
        "tool_trace": [_tool("plan_leave", "ok", summary)],
    }


def workforce_analytics(state: HRState) -> dict:
    ctx = get_ctx()
    user, employee, _department = load_actor(ctx.db, ctx.user_id)
    message = (state.get("message") or "").lower()
    months = 3
    inherited = _usable_context(state, user.id, employee.id) or {}
    if "six month" in message:
        months = 6
    elif inherited.get("months"):
        months = int(inherited["months"])
    payload = absence_analytics(ctx.db, user, months=months)
    prior_focus = inherited.get("focus") or ""
    if re.search(r"\b(break(?:\s+it)?\s+down|by department)\b", message):
        focus = "department_breakdown"
    elif re.search(r"\b(remaining leave|highest remaining|leave balances?)\b", message):
        focus = "remaining_balances"
        payload = {**payload, "department_remaining": _department_remaining_leave(ctx.db)}
    elif re.search(r"\b(data integrity|integrity issues?|these calculations)\b", message):
        focus = "data_integrity"
    elif re.search(r"\b(confidently report|which figures|can hr)\b", message):
        focus = "reporting_confidence"
    elif re.search(
        r"\b(compare|comparison|contrast).{0,40}\b(last|past|previous).{0,20}\bmonths?\b|"
        r"\b(last|past|previous)\s+(?:\w+\s+)?months?\b.{0,20}\b(compar|contrast)",
        message,
    ) or (
        re.search(r"\b(compare|comparison|contrast)\b", message)
        and re.search(r"\b(last|past|previous)\s+(?:\w+\s+)?(?:complete\s+)?(?:calendar\s+)?months?\b", message)
    ) or re.search(
        r"\b(last|past|previous)\s+(?:two|2)\s+(?:complete\s+)?(?:calendar\s+)?months?\b",
        message,
    ):
        focus = "month_comparison"
    elif re.search(
        r"\b(office|location|site).{0,40}\b(increase|change|rose|largest)|"
        r"\blargest increase\b",
        message,
    ):
        # Office increase requires a prior attendance comparison context; otherwise clarify.
        if prior_focus in {
            "month_comparison",
            "office_comparison",
            "absence",
            "anomaly",
            "investigation",
            "recommendations",
            "evidence",
            "department_breakdown",
            "",
        } or inherited.get("subject") in {"workforce_analytics", "org_attendance"}:
            focus = "office_comparison"
        else:
            focus = "office_clarification"
    elif re.search(r"\b(unusual|anomal(?:y|ies)|attendance patterns?)\b", message):
        focus = "anomaly"
    elif re.search(r"\b(further )?investigation|need(?:s)? (?:further )?investigation|might need\b", message):
        focus = "investigation"
    elif re.search(
        r"\b(explain the data|data behind|figures? behind|supporting (?:data|figures|evidence))\b",
        message,
    ):
        focus = "evidence"
    elif re.search(r"\b(recommend|recommendation|actions? (?:for|to) hr|what actions)\b", message):
        focus = "recommendations"
    elif "utilization" in message or prior_focus in {
        "leave_utilization",
        "department_breakdown",
        "remaining_balances",
        "data_integrity",
        "reporting_confidence",
    }:
        focus = "leave_utilization"
    else:
        focus = "absence"
    payload = {**payload, "followup_focus": focus}
    summary = payload["insights"][0] if payload["insights"] else "analytics complete"
    context_focus = (
        "leave_utilization"
        if focus
        in {
            "leave_utilization",
            "department_breakdown",
            "remaining_balances",
            "data_integrity",
            "reporting_confidence",
        }
        or "utilization" in message
        or prior_focus == "leave_utilization"
        else focus
    )
    month_cmp = payload.get("month_comparison") or {}
    office_cmp = payload.get("office_comparison") or {}
    return {
        "analytics": payload,
        "analysis": {
            "kind": "workforce_analytics",
            "focus": focus,
            "period": payload.get("period"),
            "scope": payload.get("scope"),
            "method": payload.get("method"),
            "leave_utilization": payload.get("leave_utilization"),
            "department_remaining": payload.get("department_remaining"),
            "departments": payload.get("departments"),
            "anomalies": payload.get("anomalies"),
            "month_comparison": month_cmp,
            "office_comparison": office_cmp,
            "investigation": payload.get("investigation"),
            "recommendations": payload.get("recommendations"),
            "limitations": payload.get("limitations"),
        },
        "record_context": {
            "subject": "workforce_analytics",
            "focus": context_focus,
            "months": months,
            "period": payload.get("period"),
            "metric": "unplanned_absence_rate",
            "comparison_previous": (month_cmp.get("previous") or {}).get("label"),
            "comparison_latest": (month_cmp.get("latest") or {}).get("label"),
            "user_id": user.id,
            "employee_id": employee.id,
        },
        "workflow": [_step("workforce_analytics", "Workforce analytics", "complete", payload["scope"])],
        "tool_trace": [_tool("compute_absence_analytics", "ok", summary[:180])],
    }


def _department_remaining_leave(db) -> list[dict]:
    """Department totals of remaining annual leave for utilization follow-ups."""
    from collections import defaultdict

    from app.models import LeaveBalance

    year = org_today().year
    employees = db.query(Employee).filter(Employee.employment_status == "active").all()
    dept_names = {row.id: row.name for row in db.query(Department).all()}
    emp_dept = {row.id: dept_names.get(row.department_id, "") for row in employees}
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"entitled": 0.0, "used": 0.0, "pending": 0.0, "remaining": 0.0, "employees": 0}
    )
    balances = (
        db.query(LeaveBalance)
        .filter(LeaveBalance.leave_type == "annual", LeaveBalance.year == year)
        .all()
    )
    for row in balances:
        department = emp_dept.get(row.employee_id)
        if not department:
            continue
        remaining = float(row.entitled_days) - float(row.used_days) - float(row.pending_days)
        bucket = totals[department]
        bucket["entitled"] += float(row.entitled_days)
        bucket["used"] += float(row.used_days)
        bucket["pending"] += float(row.pending_days)
        bucket["remaining"] += remaining
        bucket["employees"] += 1
    ranked = [{"department": name, **values} for name, values in totals.items()]
    ranked.sort(key=lambda item: (-item["remaining"], item["department"]))
    return ranked


def _usable_context(state: HRState, user_id: int, employee_id: int) -> dict | None:
    context = state.get("record_context")
    if not context:
        return None
    if context.get("user_id") not in {None, user_id}:
        return None
    if context.get("employee_id") not in {None, employee_id}:
        return None
    return context


def lookup_records(state: HRState) -> dict:
    """Load one record type for the authenticated employee. Message text never selects the id."""
    ctx = get_ctx()
    user, employee, _department = load_actor(ctx.db, ctx.user_id)
    intent = state.get("intent")
    message = state.get("message") or ""
    inherited = _usable_context(state, user.id, employee.id)
    context = inherited
    try:
        if intent == "team_roster":
            payload = team_roster_report(ctx.db, employee, message)
            summary = f"{payload.get('people_count', 0)} direct reports"
            context = {
                "subject": "team_roster",
                "intent": "team_roster",
                "user_id": user.id,
                "employee_id": employee.id,
                "focus": "direct_reports",
            }
            return {
                "analysis": payload,
                "record_context": context,
                "workflow": [_step("lookup_records", "Team roster", "complete", summary)],
                "tool_trace": [_tool("team_roster", "ok", summary)],
            }
        if intent == "team_analytics":
            payload = team_analytics_report(ctx.db, employee, message, inherited)
            summary = f"{payload.get('focus')}: {payload.get('people_count', 0)} people"
            # Anchor the top-ranked person when a single status ranking was produced
            # so follow-ups ("their present days") keep person + month.
            anchor_employee_id = None
            anchor_name = None
            people = payload.get("people") or []
            asked_status = payload.get("asked_status")
            if asked_status and people:
                top = people[0]
                # Resolve employee id from number when payload only has number/name.
                number = top.get("employee_number")
                if number:
                    from app.models import Employee as EmpModel

                    hit = (
                        ctx.db.query(EmpModel)
                        .filter(EmpModel.employee_number == number, EmpModel.manager_id == employee.id)
                        .first()
                    )
                    if hit is not None:
                        anchor_employee_id = hit.id
                        anchor_name = hit.full_name
            if inherited and inherited.get("anchor_employee_id") and not anchor_employee_id:
                anchor_employee_id = inherited.get("anchor_employee_id")
                anchor_name = inherited.get("anchor_name")
            context = {
                "subject": "team_analytics",
                "intent": "team_analytics",
                "user_id": user.id,
                "employee_id": employee.id,
                "focus": payload.get("focus") or "",
                "start": payload.get("start"),
                "end": payload.get("end"),
                "label": payload.get("label"),
                "asked_status": asked_status,
                "anchor_employee_id": anchor_employee_id,
                "anchor_name": anchor_name,
            }
            return {
                "analysis": payload,
                "record_context": context,
                "workflow": [_step("lookup_records", "Team analytics", "complete", summary)],
                "tool_trace": [_tool("team_analytics", "ok", summary)],
            }
        if intent in {"employee_directory", "org_attendance", "org_leave", "compensation_query", "hr_metrics", "hr_action"}:
            payload, org_context = run_hr_report(
                ctx.db,
                intent,
                message,
                inherited,
                list(state.get("target_employee_ids") or []),
            )
            if org_context is not None:
                org_context["user_id"] = user.id
                org_context["employee_id"] = employee.id
                org_context["focus"] = payload.get("focus") or ""
                targets = list(state.get("target_employee_ids") or [])
                if targets:
                    org_context["subject_employee_id"] = targets[0]
                elif inherited and inherited.get("subject_employee_id"):
                    org_context["subject_employee_id"] = inherited["subject_employee_id"]
                context = org_context
            summary = payload.get("statement") or payload.get("note") or intent
            return {
                "analysis": payload,
                "record_context": context,
                "workflow": [_step("lookup_records", "HR records", "complete", str(summary)[:180])],
                "tool_trace": [_tool(intent, "ok", str(summary)[:180])],
            }
        if intent == "attendance_query":
            payload = attendance_report(ctx.db, employee.id, message, inherited)
            if payload.get("kind") == "attendance_comparison":
                summary = f"{len(payload.get('periods') or [])} attendance periods"
                latest = (payload.get("periods") or [None])[-1] or {}
                context = {
                    "subject": "attendance",
                    "intent": "attendance_query",
                    "user_id": user.id,
                    "employee_id": employee.id,
                    "start": latest.get("start"),
                    "end": latest.get("end"),
                    "label": latest.get("range_label"),
                    "counts": latest.get("counts") or {},
                    "focus": "",
                }
            elif payload.get("kind") in {"reconciliation", "leave_overlap", "year_coverage"}:
                summary = payload.get("kind")
                context = {
                    "subject": "reconciliation" if payload.get("kind") == "reconciliation" else "attendance",
                    "intent": "attendance_query",
                    "user_id": user.id,
                    "employee_id": employee.id,
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "label": payload.get("range_label"),
                    "focus": payload.get("focus") or "",
                    "counts": payload.get("counts") or {},
                }
            elif payload.get("focus") == "unsupported_metric" or payload.get("unsupported"):
                summary = payload.get("note") or "Unsupported attendance metric"
                context = {
                    "subject": "attendance",
                    "intent": "attendance_query",
                    "user_id": user.id,
                    "employee_id": employee.id,
                    "focus": "unsupported_metric",
                    "unsupported": payload.get("unsupported") or "",
                    "counts": {},
                }
            else:
                summary = f"{payload.get('matched_count', payload.get('recorded_days', 0))} attendance rows"
                context = {
                    "subject": "attendance",
                    "intent": "attendance_query",
                    "user_id": user.id,
                    "employee_id": employee.id,
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "label": payload.get("range_label"),
                    "counts": payload.get("counts") or {},
                    "focus": payload.get("focus") or "",
                }
                # Keep the previous month window when the user switches periods so a later
                # "compare those two" can recover both antecedents.
                if (
                    inherited
                    and inherited.get("subject") == "attendance"
                    and inherited.get("start")
                    and inherited.get("end")
                    and (
                        inherited.get("start") != payload.get("start")
                        or inherited.get("end") != payload.get("end")
                    )
                ):
                    context["prior_window"] = {
                        "start": inherited.get("start"),
                        "end": inherited.get("end"),
                        "label": inherited.get("label") or "",
                        "counts": inherited.get("counts") or {},
                        "focus": inherited.get("focus") or "",
                    }
                elif inherited and inherited.get("prior_window"):
                    context["prior_window"] = inherited.get("prior_window")
        elif intent == "holiday_query":
            payload = holiday_report(ctx.db, employee, message, inherited)
            summary = f"{len(payload['holidays'])} holidays"
            # Preserve an active leave-plan thread so later "those dates" / project asks still resolve.
            if inherited and inherited.get("subject") == "leave_plan":
                context = {
                    **inherited,
                    "user_id": user.id,
                    "employee_id": employee.id,
                    "focus": "holidays",
                    "holiday_start": payload["start"],
                    "holiday_end": payload["end"],
                    "holiday_label": payload["range_label"],
                }
            else:
                context = {
                    "subject": "holiday",
                    "intent": "holiday_query",
                    "user_id": user.id,
                    "employee_id": employee.id,
                    "start": payload["start"],
                    "end": payload["end"],
                    "label": payload["range_label"],
                }
        elif intent == "leave_balance":
            payload = leave_balance_report(ctx.db, employee.id, message, inherited)
            if payload.get("kind") == "sick_taken":
                summary = f"{payload.get('attendance_sick_days', 0)} sick attendance rows"
                leave_types = ["sick"]
            else:
                summary = ", ".join(item["leave_type"] for item in payload["balances"])
                leave_types = [item["leave_type"] for item in payload["balances"]]
            if not (inherited and inherited.get("subject") == "attendance" and re.search(r"\b(affect|impact|deduct|reduce)\b", message, re.IGNORECASE)):
                context = {
                    "subject": "leave_balance",
                    "intent": "leave_balance",
                    "user_id": user.id,
                    "employee_id": employee.id,
                    "leave_types": leave_types,
                }
        elif intent == "leave_cancellation":
            payload = cancellation_report(ctx.db, employee.id)
            summary = f"{len(payload['requests'])} open requests"
            context = None
        elif intent == "leave_request_status":
            payload = leave_status_report(ctx.db, employee.id, message, inherited)
            summary = f"{len(payload['requests'])} leave requests"
            kept_request = (inherited or {}).get("request_id") if inherited else None
            if payload.get("filter") == "latest" and payload.get("requests"):
                kept_request = payload["requests"][0]["id"]
            context = {
                "subject": "leave_history",
                "intent": "leave_request_status",
                "user_id": user.id,
                "employee_id": employee.id,
                "request_id": kept_request,
                "filter": payload.get("filter"),
            }
        else:
            kinds = set(state.get("sensitive_kinds") or [])
            from app.access import can_view_personal

            payload = profile_report(
                ctx.db,
                employee,
                include_salary="salary" in kinds,
                include_performance="performance" in kinds,
                # Existing privacy policy: an employee may view their own phone; HR may view any.
                include_phone=can_view_personal(user, employee),
            )
            summary = payload["full_name"]
            if not (inherited and inherited.get("subject") == "leave_eligibility"):
                context = None
    except WindowError as exc:
        payload = {"kind": intent, "employee_id": employee.id, "error": str(exc)}
        summary = str(exc)
    return {
        "analysis": payload,
        "record_context": context,
        "workflow": [_step("lookup_records", "Record lookup", "complete", summary)],
        "tool_trace": [_tool("lookup_records", "ok", summary)],
    }


def _policy_paragraphs(db, slug: str, keywords: tuple[str, ...]) -> str:
    document = db.query(PolicyDocument).filter(PolicyDocument.slug == slug).one_or_none()
    if document is None:
        return ""
    chunks = (
        db.query(PolicyChunk)
        .filter(PolicyChunk.policy_id == document.id)
        .order_by(PolicyChunk.chunk_index)
        .all()
    )
    text = "\n\n".join(chunk.content for chunk in chunks) or (document.body or "")
    chosen = []
    for paragraph in re.split(r"\n\s*\n", text):
        compact = " ".join(paragraph.split())
        if compact and any(word in compact.lower() for word in keywords):
            chosen.append(compact)
    return "\n\n".join(chosen)[:1200]


def _focus_citations(db, citations: list[dict], keywords: tuple[str, ...]) -> list[dict]:
    focused = []
    for citation in citations:
        passage = _policy_paragraphs(db, citation.get("policy_id") or "", keywords)
        if passage:
            citation = dict(citation)
            citation["excerpt"] = passage
        focused.append(citation)
    return focused


def _policy_topic(message: str, inherited: dict | None) -> str:
    text = message.lower()
    inherited = inherited or {}
    if post_approval_question(text):
        return "post_approval"
    if re.search(r"\breset\b", text):
        return "leave_reset"
    if re.search(r"\b(carry forward|carry over|carryover|carry into)\b", text):
        return "leave_carryover"
    if re.search(r"\bovertime\b", text):
        return "overtime"
    if inherited.get("policy_topic") == "overtime" and re.search(
        r"\b(premium|start|authorize|authorise|approval|approve|hours|when|who)\b", text
    ) and not re.search(r"\b(leave|sick|remote|balance|attendance)\b", text):
        return "overtime"
    if re.search(r"\b(maternity|paternity|parental|transfer|resign|probation)\b", text):
        return ""
    if remote_policy_question(text):
        return "remote"
    if inherited.get("policy_topic") == "remote" and re.search(
        r"\b(approval|approve|manager|days|tomorrow|policy|remote|from home)\b", text
    ):
        if not re.search(r"\b(leave|balance|sick)\b", text):
            return "remote"
    return ""


def _retrieval_query(state: HRState) -> str:
    intent = state.get("intent")
    message = state.get("message") or ""
    text = message.lower()
    topic = _policy_topic(message, state.get("record_context"))
    if intent in _REQUEST_INTENTS:
        return "annual leave entitlement notice period approval manager critical project deadline People Operations co-approval"
    if intent == "leave_policy":
        return message + " annual leave notice period duration business days approval"
    if intent == "team_conflict":
        return "leave approval policy manager decision People Operations co-approval"
    if intent == "workforce_analytics":
        return "attendance policy unplanned absence core hours"
    if intent == "org_transfer":
        return "employee transfer approval routing approval manager previous manager"
    if intent == "policy_question":
        if topic == "overtime" or re.search(r"\bovertime\b", text):
            if re.search(r"\bovertime\b", text) and not re.search(r"\bapprov", text):
                return message
            return message + " overtime premium hours per week manager approval authorization"
        if topic == "post_approval":
            return (
                "leave approval policy the approval manager decides first "
                "People Operations co-approval before the request is confirmed"
            )
        if topic == "leave_reset":
            return "annual leave entitlement each calendar year recorded in the leave balance carryover expires 31 March"
        if topic == "leave_carryover" or re.search(r"\b(carry forward|carry over|carryover|carry into)\b", text):
            return "annual leave carryover unused leave up to 5 days expires 31 March"
        if topic == "remote" or re.search(r"work[-\s]from[-\s]home|every friday|remote", text):
            return "remote work policy employees may work up to 3 remote days each week with agreement from the line manager"
        if re.search(r"working hours|core hours", text) and "overtime" not in text:
            return "attendance policy core hours"
        if re.search(r"working days|weekend", text):
            return "attendance policy core hours weekends published holidays are not deducted from leave"
        if "instead" in text:
            return message + " sick leave is not part of the annual leave balance"
        if "transfer" in text:
            return "employee transfer policy approval"
        if re.search(r"absent without|without prior approval|unplanned absence", text):
            return "attendance policy weekday unplanned absence prior approval"
        if "insufficient" in text and "balance" in text:
            return "annual leave insufficient balance a request is not accepted when the employee does not have sufficient annual leave balance"
        if "consecutive" in text and "leave" in text:
            return "annual leave more than 15 consecutive business days written exception"
        if re.search(r"\bresign", text) and re.search(r"\b(unused|leave|encash|settlement|terminat)\b", text):
            # Keep the user wording. Retrieval expands the search text separately.
            # Adding unused topic words here would force every required-topic group to match.
            return message
        if re.search(r"carry forward|carry over|carryover|unused annual", text) and not re.search(r"\bresign", text):
            return "annual leave carryover unused leave"
        if re.search(r"approv", text):
            return "leave approval policy the approval manager must approve the request before it is confirmed"
        if "sick" in text or "medical" in text:
            return message + " sick leave entitlement medical note consecutive business days"
        if re.search(r"maternity|paternity|parental", text):
            return "parental leave maternity paternity primary caregiver secondary caregiver"
        if re.search(r"compensat", text):
            return "compensatory leave time off in lieu"
        if re.search(r"public holiday", text):
            return "public holiday work policy"
        if re.search(r"resign", text) and re.search(r"notice", text):
            return "resignation notice period"
        if "probation" in text:
            return "probation leave eligibility"
        if "annual leave" in text and "sick" not in text and not re.search(r"\bresign", text):
            return message + " annual leave entitlement notice approval"
    return message or "HR policy"


def _preferred_citations(message: str, citations: list[dict]) -> list[dict]:
    """Keep the policy that the question names when retrieval also returned neighbors."""
    text = message.lower()
    if re.search(r"\bovertime\b", text):
        return citations

    def matching(predicate) -> list[dict] | None:
        chosen = [item for item in citations if predicate(item)]
        return chosen or None

    choice = None
    if re.search(r"\bresign", text) and re.search(r"\b(unused|leave|encash|settlement)\b", text):
        resignation = matching(
            lambda item: re.search(
                r"resign|terminat|settlement|encash|forfeit",
                f"{item.get('title', '')} {item.get('excerpt', '')}",
                re.IGNORECASE,
            )
        )
        # Annual leave entitlement or carryover text is not a resignation answer.
        return resignation or []
    if re.search(r"absent without|without prior approval|unplanned absence", text):
        choice = matching(lambda item: "attendance" in str(item.get("policy_id", "")))
    elif post_approval_question(text):
        choice = matching(lambda item: "leave-approval" in str(item.get("policy_id", "")))
    elif re.search(r"\breset\b", text):
        choice = matching(
            lambda item: str(item.get("policy_id", "")).startswith("annual-leave")
            and "expired" not in str(item.get("title", "")).lower()
        )
    elif "insufficient" in text and "balance" in text:
        choice = matching(
            lambda item: str(item.get("policy_id", "")).startswith("annual-leave")
            and "expired" not in str(item.get("title", "")).lower()
        )
    elif "consecutive" in text and "leave" in text and "sick" not in text and "medical" not in text:
        choice = matching(
            lambda item: str(item.get("policy_id", "")).startswith("annual-leave")
            and "expired" not in str(item.get("title", "")).lower()
        )
    elif "instead" not in text and "annual leave" in text and "sick" not in text and not re.search(r"\bresign", text):
        choice = matching(
            lambda item: str(item.get("policy_id", "")).startswith("annual-leave")
            and "expired" not in str(item.get("title", "")).lower()
        )
    elif "sick" in text or "medical" in text:
        choice = matching(lambda item: "sick" in str(item.get("policy_id", "")))
    elif re.search(r"work from home|remote work|\bremote\b", text):
        choice = matching(lambda item: "remote" in str(item.get("policy_id", "")))
    elif re.search(r"working hours|core hours", text) and "weekend" not in text and "overtime" not in text:
        choice = matching(lambda item: "attendance" in str(item.get("policy_id", "")))
    elif re.search(r"weekend|official working", text) and "overtime" not in text:
        choice = matching(
            lambda item: "attendance" in str(item.get("policy_id", ""))
            or (
                str(item.get("policy_id", "")).startswith("annual-leave")
                and "expired" not in str(item.get("title", "")).lower()
            )
        )
    elif "transfer" in text:
        choice = matching(lambda item: "transfer" in str(item.get("policy_id", "")))
    elif re.search(r"maternity|paternity|parental", text):
        choice = matching(lambda item: "parental" in str(item.get("policy_id", "")))
    elif re.search(r"approv", text):
        choice = matching(
            lambda item: "leave-approval" in str(item.get("policy_id", ""))
            or str(item.get("policy_id", "")).startswith("annual-leave")
        )
    return choice or citations


def retrieve_policy(state: HRState) -> dict:
    ctx = get_ctx()
    user, employee, department = load_actor(ctx.db, ctx.user_id)
    # Attendance-analytics answers must not be replaced by unrelated policy excerpts.
    analytics_focus = ((state.get("analysis") or {}).get("focus") or "").lower()
    if state.get("intent") == "workforce_analytics" and analytics_focus in {
        "month_comparison",
        "office_comparison",
        "office_clarification",
        "anomaly",
        "investigation",
        "recommendations",
        "evidence",
        "absence",
        "department_breakdown",
    }:
        empty = {
            "citations": [],
            "confidence": "none",
            "retrieval_mode": "skipped_for_analytics",
        }
        return {
            "retrieval": empty,
            "citations": [],
            "retrieval_confidence": "none",
            "policy_note": "",
            "workflow": [_step("retrieve_policy", "HR policy retrieval", "complete", "skipped")],
            "tool_trace": [_tool("search_policies", "skipped", "analytics focus — no policy retrieval")],
        }
    result = retrieve(
        ctx.db,
        _retrieval_query(state),
        role=user.role,
        department=department.name if department else None,
        location=employee.location,
    )
    note = ""
    if state.get("intent") in _REQUEST_INTENTS and "extended-leave-approval" in conflict_groups_in_force(ctx.db):
        note = (
            "Operational workflows follow the global Leave Approval Policy. "
            "The Engineering Leave Addendum is also current and conflicts with it, so the looser addendum is not applied."
        )
    confidence = result["confidence"]
    citations = result["citations"]
    if state.get("intent") == "leave_policy":
        citations = _focus_citations(ctx.db, citations, ("notice",))
        result = dict(result)
        result["citations"] = citations
    elif state.get("intent") == "policy_question":
        citations = _preferred_citations(state.get("message") or "", citations)
        message = state.get("message") or ""
        lowered = message.lower()
        if "insufficient" in lowered and "balance" in lowered:
            citations = _focus_citations(ctx.db, citations, ("sufficient", "not accepted"))
        elif re.search(r"carry forward|carry over|carryover|carry into|unused annual", lowered):
            citations = _focus_citations(ctx.db, citations, ("carry", "unused", "5", "march"))
        elif "consecutive" in lowered and "leave" in lowered:
            citations = _focus_citations(ctx.db, citations, ("consecutive",))
        elif re.search(r"absent without|without prior approval", lowered):
            citations = _focus_citations(ctx.db, citations, ("unplanned", "notifies"))
        elif _policy_topic(message, state.get("record_context")) == "leave_reset":
            citations = _focus_citations(ctx.db, citations, ("calendar year", "carry", "unused", "march"))
        elif _policy_topic(message, state.get("record_context")) == "post_approval":
            citations = _focus_citations(ctx.db, citations, ("co-approval", "confirmed", "manager", "decides"))
        elif _policy_topic(message, state.get("record_context")) == "remote":
            citations = _focus_citations(ctx.db, citations, ("remote", "manager", "week", "country"))
        if re.search(r"official working|working days|working hours|core hours", message, re.IGNORECASE):
            passage = _policy_paragraphs(ctx.db, "attendance-v2", ("core hours", "weekday"))
            document = ctx.db.query(PolicyDocument).filter(PolicyDocument.slug == "attendance-v2").one_or_none()
            if passage and document is not None and not any(item.get("policy_id") == "attendance-v2" for item in citations):
                citations = [
                    {
                        "policy_id": document.slug,
                        "title": document.title,
                        "version": document.version,
                        "effective_date": document.effective_date.isoformat(),
                        "excerpt": passage,
                    },
                    *citations,
                ]
        result = dict(result)
        result["citations"] = citations
    stored = {
        "retrieval": result,
        "citations": citations,
        "retrieval_confidence": confidence,
        "policy_note": note,
        "workflow": [_step("retrieve_policy", "HR policy retrieval", "complete", confidence)],
        "tool_trace": [_tool("search_policies", result.get("retrieval_mode") or confidence, f"{len(result['citations'])} citations")],
    }
    if state.get("intent") == "policy_question":
        topic = _policy_topic(state.get("message") or "", state.get("record_context"))
        asked = (state.get("message") or "").lower()
        if topic == "remote":
            stored["analysis"] = {
                "kind": "remote_eligibility",
                "specific_date": bool(re.search(r"\b(tomorrow|today|next week)\b", asked)),
            }
        elif topic == "post_approval":
            stored["analysis"] = post_approval_facts()
        elif topic == "leave_reset":
            stored["analysis"] = {"kind": "leave_reset"}
        elif topic == "leave_carryover":
            balances = leave_balance_report(ctx.db, employee.id, "annual leave balance")
            annual = next((item for item in balances.get("balances") or [] if item.get("leave_type") == "annual"), None)
            stored["analysis"] = {
                "kind": "leave_carryover",
                "hypothetical_remaining": hypothetical_remaining_days(asked),
                "actual_balance": annual,
                "policy_max_carryover": 5,
                "carryover_expires": "31 March",
            }
        stored["record_context"] = {
            "subject": "policy",
            "intent": "policy_question",
            "policy_topic": topic,
            "user_id": user.id,
            "employee_id": employee.id,
        }
    return stored


def assess_eligibility(state: HRState) -> dict:
    """Compare the signed-in balance with the annual leave policy. Do not open a request."""
    ctx = get_ctx()
    _user, employee, department = load_actor(ctx.db, ctx.user_id)
    message = state.get("message") or ""
    balance = leave_balance_report(ctx.db, employee.id, message)
    annual = next((item for item in balance["balances"] if item["leave_type"] == "annual"), None)
    result = retrieve(
        ctx.db,
        message + " annual leave notice period approval consecutive limit insufficient balance",
        role=_user.role,
        department=department.name if department else None,
        location=employee.location,
    )
    citations = _focus_citations(ctx.db, result["citations"], ("notice", "approv", "sufficient", "consecutive"))
    result = dict(result)
    result["citations"] = citations
    requested = requested_leave_days(message)
    return {
        "analysis": {
            "kind": "leave_eligibility",
            "employee_id": employee.id,
            "balance": annual,
            "requested_days": requested,
            "hypothetical_remaining": hypothetical_remaining_days(message),
        },
        "record_context": {
            "subject": "leave_eligibility",
            "user_id": _user.id,
            "employee_id": employee.id,
            "requested_days": requested,
        },
        "retrieval": result,
        "citations": citations,
        "retrieval_confidence": result["confidence"],
        "pending_action": None,
        "workflow": [_step("assess_eligibility", "Leave eligibility", "complete", "balance and policy")],
        "tool_trace": [_tool("assess_leave_eligibility", result.get("retrieval_mode") or "ok", "no request created")],
    }


def prepare_workflow(state: HRState) -> dict:
    intent = state.get("intent")
    pending = None
    detail = "no consequential action"
    analysis = state.get("analysis") or {}
    focus = state.get("leave_focus") or analysis.get("focus") or ""
    # Review / "submit only after I confirm" never authorizes submission by themselves.
    defer_pending = focus in {"draft_review", "confirmation_gate", "holidays", "projects", "approvals", "business_days"}
    if intent in _REQUEST_INTENTS:
        analysis = state.get("analysis") or {}
        requested = analysis.get("requested_window") or {}
        if requested and not requested.get("hard_blocks") and not defer_pending:
            options = {
                "primary": {
                    "id": "primary",
                    "label": f"Submit {requested['label']} (the window you asked about)",
                    "start": requested["start"],
                    "end": requested["end"],
                    "business_days": requested["business_days"],
                    "risk_level": requested["risk_level"],
                    "hr_required": requested["hr_required"],
                    "leave_type": analysis.get("leave_type") or "annual",
                }
            }
            recommended = analysis.get("recommended_window") or {}
            if recommended and not recommended.get("hard_blocks") and recommended.get("start") != requested.get("start"):
                options["alternative"] = {
                    "id": "alternative",
                    "label": f"Submit {recommended['label']} (lower conflict)",
                    "start": recommended["start"],
                    "end": recommended["end"],
                    "business_days": recommended["business_days"],
                    "risk_level": recommended["risk_level"],
                    "hr_required": recommended["hr_required"],
                    "leave_type": analysis.get("leave_type") or "annual",
                }
            pending = {
                "type": "submit_leave",
                "confirmation_prompt": "Confirm which leave request I should file. I will not approve it.",
                "options": options,
                "default_option": "primary",
                "workflow_state": "AWAITING_CONFIRMATION",
            }
            detail = "awaiting confirmation"
        elif requested and requested.get("hard_blocks"):
            detail = "blocked; draft not eligible"
        elif defer_pending:
            detail = f"leave focus={focus}; no submission"
    elif intent == "org_transfer":
        analysis = state.get("analysis") or {}
        if not analysis:
            ctx = get_ctx()
            _user, employee, _department = load_actor(ctx.db, ctx.user_id)
            analysis = investigate_routing(ctx.db, employee)
        if analysis.get("correction"):
            pending = {
                "type": "file_routing_correction",
                "confirmation_prompt": "File this routing correction for People Operations? Your record will not change until HR approves it.",
                "options": {"primary": {"id": "primary", "label": "File the HR correction"}},
                "default_option": "primary",
                "correction": analysis["correction"],
            }
            detail = "awaiting HR filing confirmation"
    return {
        "analysis": state.get("analysis") or (analysis if intent == "org_transfer" else state.get("analysis")),
        "pending_action": pending,
        "workflow": [_step("prepare_workflow", "Workflow preparation", "complete", detail)],
        "tool_trace": [_tool("prepare_action", "ok" if pending else "skipped", detail)],
    }


def compose_response(state: HRState) -> dict:
    text, summary = narrate(state)
    mode = state.get("orchestration_mode") or "deterministic"
    conflicts = (state.get("retrieval") or {}).get("conflicts") or []
    structured = (state.get("analysis") or {}).get("kind") in {
        "post_approval",
        "remote_eligibility",
        "leave_reset",
        "leave_carryover",
    }
    polishable = state.get("authorized", True) and not state.get("clarify") and not conflicts and not structured
    polishable = polishable and state.get("intent") not in {
        "confidential_lookup",
        "attendance_query",
        "holiday_query",
        "leave_balance",
        "leave_cancellation",
        "leave_eligibility",
        "leave_policy",
        "leave_request_status",
        "employee_profile",
        "out_of_scope",
    }
    if polishable and state.get("intent") != "general_hr":
        polished, used = polish(text, text)
        if used == "llm":
            text = polished
            mode = "llm"
    elif (
        not structured
        and state.get("intent") == "policy_question"
        and state.get("retrieval_confidence") == "strong"
    ):
        polished, used = polish(text, text)
        if used == "llm":
            text = polished
            mode = "llm"
    return {
        "response": text,
        "decision_summary": summary,
        "orchestration_mode": mode,
        "workflow": [_step("compose_response", "Response generation", "complete", mode)],
    }


def human_gate(state: HRState) -> dict:
    pending = state.get("pending_action")
    if not pending:
        return {
            "confirmed": None,
            "workflow": [_step("human_gate", "Human approval", "skipped", "no consequential action")],
        }
    decision = interrupt(
        {
            "type": "confirmation",
            "pending_action": pending,
            "decision_summary": state.get("decision_summary"),
        }
    )
    if not isinstance(decision, dict):
        decision = {"confirmed": bool(decision)}
    confirmed = bool(decision.get("confirmed"))
    return {
        "confirmed": confirmed,
        "selected_option": decision.get("option_id") or pending.get("default_option") or "primary",
        "workflow": [_step("human_gate", "Human approval", "complete", "confirmed" if confirmed else "declined")],
    }


def execute_workflow(state: HRState) -> dict:
    ctx = get_ctx()
    user, _employee, _department = load_actor(ctx.db, ctx.user_id)
    pending = state.get("pending_action") or {}
    action_type = pending.get("type")
    if action_type == "submit_leave":
        option = (pending.get("options") or {}).get(state.get("selected_option") or "primary")
        if not option:
            option = (pending.get("options") or {}).get("primary")
        leave_type = option.get("leave_type") or "annual"
        result = submit_leave(
            ctx.db,
            user,
            start=parse_day(option["start"]),
            end=parse_day(option["end"]),
            leave_type=leave_type,
            reason="Planned sick leave prepared with Helios" if leave_type == "sick" else "Planned annual leave prepared with Helios",
            ip_address=ctx.ip,
        )
    elif action_type == "file_routing_correction":
        result = file_routing_correction(ctx.db, user, ip_address=ctx.ip)
    else:
        result = {"status": "error", "message": "That action is not available.", "summary": "No record was changed."}
    return {
        "execution_result": result,
        "pending_action": None,
        "workflow": [_step("execute_workflow", "Workflow execution", result.get("status", "error"), result.get("summary", ""))],
        "tool_trace": [_tool("execute_authorized_action", result.get("status", "error"), result.get("summary", "")[:180])],
    }


def cancel_workflow(state: HRState) -> dict:
    ctx = get_ctx()
    write_audit(
        ctx.db,
        actor_user_id=ctx.user_id,
        action="workflow_cancelled",
        resource_type="assistant",
        resource_id=str((state.get("pending_action") or {}).get("type") or ""),
        outcome="cancelled",
        details={"intent": state.get("intent")},
        ip_address=ctx.ip,
    )
    result = {
        "status": "cancelled",
        "message": "I did not submit anything. Official records are unchanged.",
        "summary": "Confirmation was declined.",
    }
    return {
        "execution_result": result,
        "pending_action": None,
        "workflow": [_step("cancel_workflow", "Human approval", "cancelled", "no write")],
    }


def compose_result(state: HRState) -> dict:
    text, summary = result_message(state.get("execution_result") or {})
    return {
        "response": text,
        "decision_summary": summary,
        "workflow": [_step("compose_result", "Response generation", "complete", "after human decision")],
    }


def route_after_auth(state: HRState) -> str:
    if not state.get("authorized", False) or state.get("clarify"):
        return "compose_response"
    return {
        "leave_planning": "employee_context",
        "leave_request": "employee_context",
        "leave_eligibility": "assess_eligibility",
        "org_transfer": "employee_context",
        "self_service": "employee_context",
        "employee_profile": "lookup_records",
        "employee_directory": "lookup_records",
        "team_roster": "lookup_records",
        "team_analytics": "lookup_records",
        "org_attendance": "lookup_records",
        "org_leave": "lookup_records",
        "compensation_query": "lookup_records",
        "hr_metrics": "lookup_records",
        "hr_action": "lookup_records",
        "attendance_query": "lookup_records",
        "holiday_query": "lookup_records",
        "leave_balance": "lookup_records",
        "leave_request_status": "lookup_records",
        "leave_cancellation": "lookup_records",
        "leave_policy": "retrieve_policy",
        "out_of_scope": "compose_response",
        "unsupported_mutation": "compose_response",
        "data_semantics": "compose_response",
        "confidential_lookup": "employee_context",
        "team_conflict": "leave_analysis",
        "approval_decision": "approval_decision",
        "workforce_analytics": "workforce_analytics",
        "policy_question": "retrieve_policy",
        "general_hr": "retrieve_policy",
    }.get(state.get("intent", ""), "retrieve_policy")


def route_after_context(state: HRState) -> str:
    intent = state.get("intent")
    if intent in _REQUEST_INTENTS:
        return "leave_analysis"
    if intent == "org_transfer":
        return "investigate_then_retrieve"
    return "compose_response"


def investigate_then_retrieve(state: HRState) -> dict:
    """Employee-context path for transfers still needs the routing tool, then policies."""
    ctx = get_ctx()
    _user, employee, _department = load_actor(ctx.db, ctx.user_id)
    analysis = investigate_routing(ctx.db, employee)
    return {
        "analysis": analysis,
        "workflow": [_step("investigate_routing", "Employee context", "complete", "routing compared")],
        "tool_trace": [_tool("investigate_approval_routing", "ok", "consistent" if analysis.get("consistent") else "mismatch")],
    }


def route_after_leave(state: HRState) -> str:
    return "retrieve_policy"


def route_after_analytics(state: HRState) -> str:
    return "retrieve_policy"


def route_after_retrieval(state: HRState) -> str:
    if state.get("intent") in {"leave_planning", "leave_request", "org_transfer"}:
        return "prepare_workflow"
    return "compose_response"


def route_after_gate(state: HRState) -> str:
    if state.get("confirmed") is True:
        return "execute_workflow"
    if state.get("confirmed") is False:
        return "cancel_workflow"
    return END


def build_graph():
    graph = StateGraph(HRState)
    graph.add_node("classify_intent", classify_intent, retry_policy=TRANSIENT)
    graph.add_node("authorize", authorize)
    graph.add_node("employee_context", employee_context)
    graph.add_node("lookup_records", lookup_records)
    graph.add_node("investigate_then_retrieve", investigate_then_retrieve)
    graph.add_node("leave_analysis", leave_analysis)
    graph.add_node("approval_decision", approval_decision)
    graph.add_node("workforce_analytics", workforce_analytics)
    graph.add_node("assess_eligibility", assess_eligibility)
    graph.add_node("retrieve_policy", retrieve_policy)
    graph.add_node("prepare_workflow", prepare_workflow)
    graph.add_node("compose_response", compose_response, retry_policy=TRANSIENT)
    graph.add_node("human_gate", human_gate)
    graph.add_node("execute_workflow", execute_workflow)
    graph.add_node("cancel_workflow", cancel_workflow)
    graph.add_node("compose_result", compose_result)

    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "authorize")
    graph.add_conditional_edges(
        "authorize",
        route_after_auth,
        {
            "compose_response": "compose_response",
            "employee_context": "employee_context",
            "lookup_records": "lookup_records",
            "leave_analysis": "leave_analysis",
            "approval_decision": "approval_decision",
            "workforce_analytics": "workforce_analytics",
            "retrieve_policy": "retrieve_policy",
            "assess_eligibility": "assess_eligibility",
        },
    )
    graph.add_edge("assess_eligibility", "compose_response")
    graph.add_edge("approval_decision", "compose_response")
    graph.add_conditional_edges(
        "employee_context",
        route_after_context,
        {
            "leave_analysis": "leave_analysis",
            "investigate_then_retrieve": "investigate_then_retrieve",
            "compose_response": "compose_response",
        },
    )
    graph.add_edge("lookup_records", "compose_response")
    graph.add_edge("investigate_then_retrieve", "retrieve_policy")
    graph.add_edge("leave_analysis", "retrieve_policy")
    graph.add_edge("workforce_analytics", "retrieve_policy")
    graph.add_conditional_edges(
        "retrieve_policy",
        route_after_retrieval,
        {"prepare_workflow": "prepare_workflow", "compose_response": "compose_response"},
    )
    graph.add_edge("prepare_workflow", "compose_response")
    graph.add_edge("compose_response", "human_gate")
    graph.add_conditional_edges(
        "human_gate",
        route_after_gate,
        {"execute_workflow": "execute_workflow", "cancel_workflow": "cancel_workflow", END: END},
    )
    graph.add_edge("execute_workflow", "compose_result")
    graph.add_edge("cancel_workflow", "compose_result")
    graph.add_edge("compose_result", END)
    return graph
