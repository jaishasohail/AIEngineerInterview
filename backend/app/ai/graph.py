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
    _explicit_leave_request,
    advance_leave_eligibility_question,
    ambiguous_team_or_org_scope,
    approval_inbox_followup_focus,
    approval_inbox_question,
    classify_with_model,
    continuation_intent,
    delayed_approval_advice_question,
    subject_employee_pronoun_followup,
    employee_self_approval_question,
    explicit_routing_correction_request,
    find_mentioned,
    get_ctx,
    hypothetical_remaining_days,
    is_clarification_reply,
    leave_date_recommendation_question,
    leave_history_record_context,
    leave_plan_followup,
    leave_plan_thread_active,
    leave_planning_continuation_message,
    load_actor,
    manager_team_ops_question,
    manager_leave_decision_guidance_question,
    managed_leave_request_status_id,
    named_employee_identity_question,
    org_transfer_followup_focus,
    organization_intent,
    org_attendance_month_comparison_question,
    org_attendance_named_month_rate_question,
    hr_workforce_risk_analytics_question,
    parse_approval_decision,
    parse_business_days,
    personal_leave_status_question,
    post_approval_question,
    profile_for,
    hr_corrective_action_question,
    record_consistency_question,
    workforce_attendance_corrective_question,
    workforce_corrective_thread_prepare,
    workforce_corrective_prepare_question,
    remote_policy_question,
    unauthorized_leave_consequence_question,
    requested_leave_days,
    routing_discrepancy_question,
    rule_intent,
    team_leave_overlap_question,
    workforce_analytics_focus,
    workforce_followup_question,
)
from app.ai.guard import (
    broad_private_data_request,
    detect_injection,
    hard_security_violation,
    manager_requested_organization_scope,
    organization_wide_request,
    other_employee_private_request,
    out_of_hierarchy_request,
    outside_authorized_scope_request,
    sensitive_kinds,
    unsupported_destructive_request,
    wants_bulk_export,
)
from app.ai.llm import (
    KNOWLEDGE_INTENTS,
    TOOL_EXPLAIN_INTENTS,
    generate_grounded_answer,
    polish,
)
from app.ai.narrate import narrate, result_message
from app.ai.query_spec import intent_from_spec, interpret_and_validate
from app.ai.security import assess_security
from app.ai.retrieval import conflict_groups_in_force, retrieve
from app.audit import write_audit
from app.access import AuthorizationError
from app.services.actions import decide_approval, file_routing_correction, submit_leave
from app.services.confirmation import (
    cancel_pending,
    execute_pending,
    get_pending,
    prepare_decision_pending,
    prepare_routing_pending,
    prepare_submit_leave_pending,
    prepare_workforce_corrective_pending,
)
from app.services.analytics_engine import absence_analytics, workforce_corrective_proposal
from app.services.hr_ops import run_hr_report
from app.services.leave_engine import (
    analyze_team,
    balance_payload,
    get_balance,
    investigate_routing,
    investigate_routing_discrepancies,
    plan_leave,
    post_approval_facts,
    recommend_leave_dates,
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
_TOOL_ORCHESTRATED_INTENTS = {
    "leave_balance",
    "attendance_query",
    "holiday_query",
    "team_conflict",
    "org_attendance",
    # workforce_analytics uses the dedicated graph node so follow-up focuses differ.
    "org_transfer",
    "policy_question",
    "leave_policy",
    "general_hr",
}
from app.clock import add_business_days, org_today, parse_day
from app.services.holidays import holidays_for_employee

TRANSIENT = RetryPolicy(max_attempts=2)


class HRState(TypedDict, total=False):
    message: str
    raw_message: str
    role: str
    intent: str
    orchestration_mode: str
    injection_flag: bool
    requested_org_wide: bool
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
    conversation_history: list
    memory: dict
    tool_plan: dict
    tool_results: list
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
    memory = state.get("memory") or {}
    # Prefer the memory-rewritten standalone question for routing, while security
    # still inspects the raw user text when present.
    raw_message = state.get("raw_message") or state.get("message", "")
    message = state.get("message") or raw_message
    security_text = raw_message or message
    injections = detect_injection(security_text)
    kinds = sensitive_kinds(security_text)
    mentioned = find_mentioned(ctx.db, security_text, employee.id)
    inherited_for_targets = state.get("record_context") or {}
    if (
        not mentioned
        and subject_employee_pronoun_followup(message)
        and inherited_for_targets.get("subject_employee_id") is not None
        and inherited_for_targets.get("user_id") in {None, user.id}
    ):
        from app.models import Employee as EmployeeModel

        subject_row = ctx.db.get(EmployeeModel, int(inherited_for_targets["subject_employee_id"]))
        if subject_row is not None and subject_row.id != employee.id:
            mentioned = [subject_row]
    bulk = wants_bulk_export(security_text)
    ruled = rule_intent(message, mentioned_other=bool(mentioned), kinds=kinds, bulk=bulk)
    mode = "deterministic"
    leave_focus = memory.get("leave_focus")
    # The compositional security layer runs on the whole message before any intent is chosen,
    # so a malicious instruction embedded in an otherwise ordinary question denies the whole
    # message instead of letting the benign half pick the intent. It never grants access.
    security = assess_security(security_text, role=user.role, mentioned_other=bool(mentioned))
    security_audit = security.as_audit() if security.denied else None
    requested_org_wide = user.role == "manager" and manager_requested_organization_scope(
        security_text,
        message,
    )
    # Hard security probes always win. Broad org analytics are role-gated: HR Admin may ask
    # organization-wide questions; other roles are denied before retrieval.
    hr_employee_delete = bool(
        user.role == "hr_admin"
        and re.search(r"\b(delete|remove)\b", security_text, re.IGNORECASE)
        and re.search(r"\bemployee record\b", security_text, re.IGNORECASE)
    )
    if security.denied:
        intent = security.intent
        violation = True
    elif requested_org_wide:
        # Bind authorization to the original requested scope before intent routing can
        # remap a workforce request to team analytics.
        intent = "unauthorized_access"
        violation = True
    elif user.role == "employee" and re.search(
        r"\b(organization[- ]wide|organisation[- ]wide|company[- ]wide|org[- ]wide|"
        r"across (?:the )?(?:organization|organisation|company)|every employee|all employees)\b",
        security_text,
        re.IGNORECASE,
    ) and re.search(
        r"\b(absenteeism|absence|attendance|workforce|leave balance|payroll|compensation)\b",
        security_text,
        re.IGNORECASE,
    ):
        intent = "unauthorized_access"
        violation = True
    elif unsupported_destructive_request(security_text) and not hr_employee_delete:
        intent = "unsupported_mutation"
        violation = True
    elif out_of_hierarchy_request(security_text) and user.role != "hr_admin":
        intent = "unauthorized_access"
        violation = True
    elif hard_security_violation(security_text):
        intent = "unauthorized_access"
        violation = True
    elif other_employee_private_request(security_text) and user.role != "hr_admin":
        intent = "unauthorized_access"
        violation = True
    elif mentioned and user.role == "employee" and re.search(
        r"\b(medical|sick leave|leave history|leave balance|salary|compensation|phone)\b",
        security_text,
        re.IGNORECASE,
    ):
        intent = "unauthorized_access"
        violation = True
    elif broad_private_data_request(security_text) and user.role != "hr_admin":
        intent = "unauthorized_access"
        violation = True
    elif user.role == "manager" and (
        workforce_corrective_prepare_question(security_text)
        or workforce_attendance_corrective_question(security_text)
    ):
        intent = "unauthorized_access"
        violation = True
    else:
        violation = False
        intent = ruled
        # Concrete leave-request approve/reject uses the same decide_approval service as REST.
        approval_ask = parse_approval_decision(message)
        guidance_ask = user.role in {"manager", "hr_admin"} and manager_leave_decision_guidance_question(message)
        if user.role == "employee" and employee_self_approval_question(message):
            intent = "approval_decision"
            mode = "deterministic"
        elif approval_ask and not guidance_ask:
            intent = "approval_decision"
            mode = "deterministic"
        if user.role == "manager" and re.search(
            r"\bprepare (?:approval|rejection)\b", message, re.IGNORECASE
        ) and re.search(r"\b(pending|direct report|leave request)\b", message, re.IGNORECASE):
            intent = "approval_decision"
            mode = "deterministic"
        if (
            user.role in {"manager", "hr_admin"}
            and approval_inbox_question(message)
            and not personal_leave_status_question(message)
        ):
            intent = "team_analytics"
            mode = "deterministic"
        if user.role in {"manager", "hr_admin"} and managed_leave_request_status_id(message):
            intent = "team_analytics"
            mode = "deterministic"
        if user.role in {"manager", "hr_admin"} and team_leave_overlap_question(message):
            intent = "team_conflict"
            mode = "deterministic"
        if (
            user.role == "employee"
            and advance_leave_eligibility_question(message)
            and not _explicit_leave_request(message)
            and intent
            in {
                "leave_planning",
                "leave_request",
                "policy_question",
                "leave_policy",
                "general_hr",
                "out_of_scope",
            }
        ):
            intent = "leave_eligibility"
            mode = "deterministic"
        if leave_date_recommendation_question(message):
            intent = "leave_planning"
            leave_focus = "recommend_dates"
            mode = "deterministic"
        if delayed_approval_advice_question(message) and not employee_self_approval_question(message):
            intent = "leave_request_status"
            mode = "deterministic"
        # Employees never need manager permissions for their own pending/status leave.
        if (
            user.role == "employee"
            and personal_leave_status_question(message)
            and not employee_self_approval_question(message)
        ):
            intent = "leave_request_status"
            mode = "deterministic"
        # Personal reporting-manager identity is profile data, not an org routing audit.
        if (
            user.role == "employee"
            and not record_consistency_question(message)
            and re.search(
                r"\b(who is my|who's my|who is listed as my|what is my)\b.{0,40}\b"
                r"(reporting[-\s]?manager|line manager|manager)\b",
                message,
                re.IGNORECASE,
            )
            and not re.search(
                r"\b(inconsist\w*|mismatch|discrepan\w*|previous|old|stale|wrong|transfer|changed)\b",
                message,
                re.IGNORECASE,
            )
        ):
            intent = "employee_profile"
            mode = "deterministic"
        if user.role == "employee" and record_consistency_question(message):
            intent = "org_transfer"
            mode = "deterministic"
        inherited = state.get("record_context") or {}
        if user.role == "manager" and (
            workforce_corrective_prepare_question(message)
            or workforce_attendance_corrective_question(message)
        ):
            intent = "unauthorized_access"
            violation = True
        # Attendance-grounded corrective guidance stays on workforce analytics; routing remediation on org_transfer.
        if user.role == "hr_admin" and (
            workforce_attendance_corrective_question(message)
            or (
                inherited.get("subject") in {"org_attendance", "workforce_analytics"}
                and workforce_corrective_thread_prepare(message)
            )
        ):
            intent = "workforce_analytics"
            mode = "deterministic"
        elif user.role == "hr_admin" and (
            hr_corrective_action_question(message) or org_transfer_followup_focus(message) == "remediation"
        ):
            intent = "org_transfer"
            mode = "deterministic"
        same_actor = inherited.get("user_id") in {None, user.id} and inherited.get("employee_id") in {None, employee.id}
        # Keep delayed-approval threads from being remapped into manager team analytics.
        if (
            same_actor
            and inherited.get("subject") == "leave_history"
            and inherited.get("focus") in {"delayed_approval", "pending", "latest"}
            and (
                delayed_approval_advice_question(message)
                or re.search(
                    r"\b(approval manager|assigned approver|pending request|that request|"
                    r"follow up|next steps?|how long|still|respond|which request|"
                    r"who needs to approve|can i follow up)\b",
                    message,
                    re.IGNORECASE,
                )
            )
            and user.role == "employee"
        ):
            if re.search(
                r"\b(who (?:is|was) my|who's my|who is listed as my)\b.{0,30}\bmanager\b",
                message,
                re.IGNORECASE,
            ) and not re.search(r"\b(approve|pending leave|leave request)\b", message, re.IGNORECASE):
                intent = "employee_profile"
            else:
                intent = "leave_request_status"
            mode = "deterministic"
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
            if query_spec.metric == "policy_conflict" and approval_inbox_question(message):
                intent = "team_analytics"
            elif query_spec.metric == "policy_conflict":
                intent = "policy_question"
            else:
                intent = hinted
            mode = "deterministic"
        elif intent == "policy_question" and query_spec.metric == "policy_conflict" and not approval_inbox_question(
            message
        ):
            mode = "deterministic"
        elif hinted and intent == "hr_metrics" and query_spec.intent_hint == "hr_metrics":
            mode = "deterministic"
        # Leave-plan multi-turn follow-ups keep the planned window even when a weak
        # policy/holiday rule would otherwise steal the turn.
        history = state.get("conversation_history") or []
        prior_intent = inherited.get("intent") or memory.get("workflow", {}).get("prior_intent")
        if user.role in {"manager", "hr_admin"} and manager_leave_decision_guidance_question(message):
            intent = "team_analytics"
            mode = "deterministic"
        if unauthorized_leave_consequence_question(message):
            intent = "policy_question"
            mode = "deterministic"
            leave_focus = None
        if subject_employee_pronoun_followup(message) and inherited.get("subject_employee_id"):
            intent = "confidential_lookup"
            mode = "deterministic"
            leave_follow = None
        leave_follow = leave_plan_followup(message, inherited) if same_actor else None
        if unauthorized_leave_consequence_question(message):
            leave_follow = None
        if leave_follow and user.role in {"manager", "hr_admin"} and manager_leave_decision_guidance_question(message):
            leave_follow = None
        if leave_follow and not violation:
            intent, leave_focus = leave_follow
            mode = "deterministic"
        elif (
            not violation
            and user.role == "employee"
            and not unauthorized_leave_consequence_question(message)
            and intent != "leave_request_status"
            and leave_plan_thread_active(
                message,
                inherited,
                prior_intent=prior_intent,
                conversation_history=history,
            )
            and leave_planning_continuation_message(message)
            and intent not in {"leave_planning", "leave_request"}
        ):
            follow = leave_plan_followup(message, inherited) or ("leave_planning", "plan")
            intent, leave_focus = follow
            mode = "deterministic"
        # Pending-approval follow-ups keep the inbox thread even when a weak policy/transfer
        # rule matched the surface wording ("what happens if I approve", "why urgent").
        approval_follow = approval_inbox_followup_focus(message) if same_actor else None
        if (
            not violation
            and leave_follow is None
            and same_actor
            and inherited.get("subject") in {"approval_inbox", "team_analytics", "approval_decision"}
            and (
                approval_follow is not None
                or (
                    str(inherited.get("focus") or "").startswith("awaiting_approval")
                    and re.search(r"\b(urgent|approv|reject|confirm|request|it|that|this)\b", message, re.I)
                )
            )
        ):
            if approval_follow == "prepare_approve" or (
                parse_approval_decision(message) and not manager_leave_decision_guidance_question(message)
            ):
                intent = "approval_decision"
            else:
                intent = "team_analytics"
            mode = "deterministic"
        # Explicit organization-wide phrasing must not inherit a prior self/team subject.
        if (
            leave_follow is None
            and not ruled
            and same_actor
            and not mentioned
            and not organization_wide_request(message)
            and approval_follow is None
        ):
            continued = continuation_intent(message, inherited)
            if continued:
                intent = continued
        # Short period/metric switches keep the prior attendance/team/directory subject even when
        # a weak rule/LLM would pivot to policy.
        if (
            same_actor
            and inherited.get("subject") in {
                "attendance",
                "team_analytics",
                "team_roster",
                "employee_directory",
                "leave_balance",
                "policy",
            }
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
                "policy": "policy_question",
            }.get(str(inherited.get("subject")))
            if continued:
                intent = continued
                mode = "deterministic"
        if (
            not violation
            and same_actor
            and inherited.get("subject") == "policy"
            and re.search(
                r"\b(policy|version|effective|expired|contradict|conflict|entitlement|document|covered|applicable)\b",
                message,
                re.IGNORECASE,
            )
            and intent
            in {
                "workforce_analytics",
                "org_attendance",
                "general_hr",
                "out_of_scope",
                "unauthorized_access",
                "leave_policy",
            }
        ):
            intent = "policy_question"
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
        # Leave-overlap among employees is always team_conflict for managers (never workforce analytics).
        if user.role in {"manager", "hr_admin"} and team_leave_overlap_question(message):
            intent = "team_conflict"
            mode = "deterministic"
        # Managers get team-scoped analytics for team ops phrasing; HR keeps org_* intents.
        # Do NOT remap team_conflict — overlap/coverage analysis must stay on analyze_team.
        # Explicit organization-wide asks stay on org/workforce intents so authorize can deny.
        if (
            user.role == "manager"
            and manager_team_ops_question(message)
            and not organization_wide_request(message)
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
            and not team_leave_overlap_question(message)
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
            and requested_org_wide
            and intent
            in {
                "team_analytics",
                "team_roster",
                "attendance_query",
                "general_hr",
                "out_of_scope",
                "workforce_analytics",
                "org_attendance",
                "hr_metrics",
                "employee_directory",
            }
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
                r"\b(overtime|premium|extra pay|policy|policies|authorize|authorise|approval rule|"
                r"working hours|core hours|official working|weekly hours)\b",
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
            and not routing_discrepancy_question(message)
            and org_transfer_followup_focus(message) is None
        ):
            intent = "workforce_analytics"
            mode = "deterministic"
        # Keep manager pending-approval threads on team_analytics / approval_decision.
        # Do not let generic urgency phrasing steal the conversation into org_transfer.
        approval_follow = approval_inbox_followup_focus(message) if same_actor else None
        if (
            same_actor
            and inherited_ctx.get("subject") in {"approval_inbox", "team_analytics", "approval_decision"}
            and inherited_ctx.get("focus")
            in {
                "approval_inbox",
                "awaiting_approval",
                "awaiting_approval_risk",
                "awaiting_approval_why_urgent",
                "awaiting_approval_impact",
                "awaiting_approval_reject_impact",
                "awaiting_approval_approve_guidance",
                "awaiting_approval_reject_guidance",
                "pending_leave",
                "",
                None,
            }
            and (
                approval_follow is not None
                or approval_inbox_question(message)
                or continuation_intent(message, inherited_ctx) in {"team_analytics", "approval_decision"}
            )
            and intent
            in {
                "org_transfer",
                "org_leave",
                "org_attendance",
                "hr_metrics",
                "employee_directory",
                "general_hr",
                "out_of_scope",
                "workforce_analytics",
                "team_analytics",
                "policy_question",
                "leave_request_status",
                "leave_request",
                "leave_planning",
                "approval_decision",
            }
        ):
            if approval_follow == "prepare_approve" or (
                parse_approval_decision(message) and not manager_leave_decision_guidance_question(message)
            ):
                intent = "approval_decision"
            else:
                intent = "team_analytics"
            mode = "deterministic"
        # Keep routing-audit investigation threads on org_transfer for urgency/importance/steps follow-ups.
        if (
            same_actor
            and inherited_ctx.get("subject") == "workforce_analytics"
            and workforce_corrective_prepare_question(message)
        ):
            intent = "workforce_analytics"
            mode = "deterministic"
        if (
            same_actor
            and inherited_ctx.get("subject") in {"routing_audit", "routing_investigation", "org_transfer"}
            and not workforce_corrective_prepare_question(message)
            and (
                org_transfer_followup_focus(message) is not None
                or hr_corrective_action_question(message)
                or continuation_intent(message, inherited_ctx) == "org_transfer"
                or routing_discrepancy_question(message)
            )
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
                "org_transfer",
                "leave_request_status",
            }
        ):
            intent = "org_transfer"
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
        # Standalone / follow-up HR analytics phrasing stays on workforce_analytics.
        # Includes employee_directory so "which employees should HR investigate" is not a roster dump.
        # Do not steal org_attendance month-comparison follow-ups ("which department contributed most?").
        # Check raw_message too: memory may rewrite "contributed most" into a workforce ranking phrase.
        comparison_followup = bool(
            re.search(
                r"\b(contributed most|department contributed|how much did it (?:increase|decrease))\b",
                f"{raw_message} {message}",
                re.I,
            )
        )
        if (
            user.role == "hr_admin"
            and intent
            in {
                "general_hr",
                "out_of_scope",
                "policy_question",
                "org_attendance",
                "hr_metrics",
                "team_analytics",
                "employee_directory",
                "workforce_analytics",
            }
            and workforce_followup_question(message)
            and not re.search(r"\b(my team|direct reports?)\b", message, re.IGNORECASE)
            and not comparison_followup
            and not org_attendance_month_comparison_question(message)
            and not org_attendance_named_month_rate_question(message)
        ):
            intent = "workforce_analytics"
            mode = "deterministic"
        if (
            user.role == "hr_admin"
            and same_actor
            and comparison_followup
            and (
                inherited_ctx.get("subject") in {"org_attendance", "workforce_analytics"}
                or intent in {"workforce_analytics", "org_attendance", "hr_metrics"}
            )
            and re.search(
                r"\b(contributed most|department contributed|how much did it (?:increase|decrease)|what about|october|september)\b",
                f"{raw_message} {(memory.get('rewritten_message') or '')}",
                re.I,
            )
        ):
            # Month-over-month contribution/increase asks stay on org_attendance comparison, not ranking analytics.
            intent = "org_attendance"
            mode = "deterministic"
        elif (
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
        if user.role == "hr_admin" and intent == "workforce_analytics" and (
            org_attendance_month_comparison_question(message)
            or org_attendance_named_month_rate_question(message)
        ):
            intent = "org_attendance"
            mode = "deterministic"
        if (
            user.role == "hr_admin"
            and intent in {"policy_question", "general_hr", "out_of_scope", None, ""}
            and hr_workforce_risk_analytics_question(message)
        ):
            intent = "workforce_analytics"
            mode = "deterministic"
        if (
            user.role == "hr_admin"
            and same_actor
            and inherited_ctx.get("subject") == "workforce_analytics"
            and inherited_ctx.get("focus") == "workforce_risk_plan"
            and re.search(
                r"\b(why|reason|explain|prioriti[sz]e|that department|those departments|action plan)\b",
                message,
                re.IGNORECASE,
            )
            and intent
            in {
                "policy_question",
                "general_hr",
                "out_of_scope",
                "org_attendance",
                "hr_metrics",
            }
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
        # HR named-person identity asks (e.g. "who is Liam Chen") are directory lookups, not policy.
        if (
            not violation
            and user.role == "hr_admin"
            and mentioned
            and not kinds
            and named_employee_identity_question(message)
        ):
            intent = "employee_directory"
            mode = "deterministic"
        # Employee delayed-approval follow-ups must never become manager team analytics.
        if (
            not violation
            and user.role == "employee"
            and same_actor
            and inherited_ctx.get("subject") == "leave_history"
            and inherited_ctx.get("focus") in {"delayed_approval", "pending", "latest"}
            and re.search(
                r"\b(approval manager|assigned approver|pending request|that request|follow up|next steps?|what should i do)\b",
                message,
                re.IGNORECASE,
            )
            and intent
            in {
                "team_analytics",
                "team_roster",
                "org_leave",
                "org_attendance",
                "workforce_analytics",
                "general_hr",
                "out_of_scope",
                "leave_request",
            }
        ):
            if re.search(r"\b(who|approval manager|assigned approver|my manager)\b", message, re.IGNORECASE):
                intent = "employee_profile"
            else:
                intent = "leave_request_status"
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
            and leave_focus
            not in {
                "draft_review",
                "confirmation_gate",
                "prepare",
                "business_days",
                "recommend_dates",
            }
            and not leave_date_recommendation_question(message)
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
        # Defense-in-depth remaps for acceptance-critical paraphrases.
        if not violation and not clarify:
            if user.role == "employee" and personal_leave_status_question(message):
                intent = "leave_request_status"
                mode = "deterministic"
            if (
                user.role in {"manager", "hr_admin"}
                and team_leave_overlap_question(message)
                and intent
                in {
                    "leave_planning",
                    "leave_request",
                    "policy_question",
                    "leave_policy",
                    "team_analytics",
                    "workforce_analytics",
                    "general_hr",
                }
            ):
                intent = "team_conflict"
                mode = "deterministic"
            if (
                user.role in {"manager", "hr_admin"}
                and approval_inbox_question(message)
                and not personal_leave_status_question(message)
                and intent
                in {
                    "leave_request_status",
                    "leave_request",
                    "general_hr",
                    "out_of_scope",
                    "team_roster",
                    "org_leave",
                    "policy_question",
                    "leave_policy",
                    "workforce_analytics",
                    "attendance_query",
                    "leave_planning",
                }
            ):
                intent = "team_analytics"
                mode = "deterministic"
            if (
                intent
                in {
                    "approval_decision",
                    "leave_request_status",
                    "leave_planning",
                    "general_hr",
                    "out_of_scope",
                    "leave_eligibility",
                }
                and _explicit_leave_request(message)
                and not delayed_approval_advice_question(message)
                and not parse_approval_decision(message)
                and not leave_date_recommendation_question(message)
            ):
                intent = "leave_request"
                mode = "deterministic"
                if re.search(r"\b(prepare|draft|submit|file)\b", message, re.IGNORECASE):
                    leave_focus = leave_focus or "prepare"
            if (
                intent in {"leave_request_status", "holiday_query", "general_hr"}
                and re.search(r"\b(notice|rulebook|leave policy)\b", message, re.IGNORECASE)
                and not re.search(r"\b(status of my|my leave request|cancel|withdraw)\b", message, re.IGNORECASE)
            ):
                intent = "leave_policy" if re.search(r"\bnotice\b", message, re.IGNORECASE) else "policy_question"
                mode = "deterministic"
            if (
                intent in {"attendance_query", "compensation_query", "general_hr", "confidential_lookup"}
                and re.search(
                    r"\b(overtime|extra pay|premium|work past .{0,20}hours|weekly hours)\b",
                    message,
                    re.IGNORECASE,
                )
                and not re.search(r"\b(calculate|compute|incomplete|missing rate|my hours|attendance)\b", message, re.IGNORECASE)
            ):
                intent = "policy_question"
                mode = "deterministic"
            if (
                intent == "confidential_lookup"
                and re.search(
                    r"\b(compensation\s+bands?|pay\s+bands?|salary\s+bands?|compensation\s+(?:policy|notes?|document)|"
                    r"confidential\s+compensation)\b",
                    message,
                    re.IGNORECASE,
                )
            ):
                intent = "policy_question"
                mode = "deterministic"
        if (
            not violation
            and user.role == "employee"
            and not unauthorized_leave_consequence_question(message)
            and context.get("subject") == "leave_plan"
            and intent in {"policy_question", "general_hr", "employee_profile", "leave_eligibility"}
            and (leave_planning_continuation_message(message) or leave_plan_followup(message, context))
        ):
            follow = leave_plan_followup(message, context) or ("leave_planning", "plan")
            intent, leave_focus = follow
            mode = "deterministic"
    result = {
        "role": user.role,
        "intent": intent,
        "requested_org_wide": requested_org_wide,
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
        "employee_directory",
        "confidential_lookup",
        "employee_profile",
    } or (
        context.get("subject_employee_id") is not None
        and intent in {"confidential_lookup", "employee_directory", "employee_profile", "policy_question"}
    ) or (
        intent == "leave_balance" and re.search(r"\b(that|those|affect|impact)\b", text)
    ) or (
        intent in _REQUEST_INTENTS and re.search(r"\b(prepare|the request|it)\b", text)
    ) or (
        intent == "employee_profile" and re.search(r"\bapprov", text)
    ) or (
        (context.get("subject") == "leave_plan" or leave_focus)
        and intent in {"holiday_query", "leave_planning", "leave_request", "leave_balance"}
    ) or (
        context.get("subject") in {"approval_decision", "approval_inbox", "team_analytics"}
        and (
            approval_inbox_followup_focus(message) is not None
            or re.search(
                r"\b(approve|reject|status|why|reason|it|that request|urgent|confirmation|prepare|pending)\b",
                text,
            )
        )
    )
    if intent == "leave_request_status" and context.get("subject") == "leave_plan":
        result["record_context"] = None
    elif not continuing and context.get("subject") == "leave_plan":
        result["record_context"] = context
    elif (
        not continuing
        and intent != "leave_request_status"
        and leave_plan_thread_active(
            message,
            context,
            prior_intent=state.get("memory", {}).get("workflow", {}).get("prior_intent"),
            conversation_history=state.get("conversation_history") or [],
        )
    ):
        result["record_context"] = context or None
    elif not continuing:
        result["record_context"] = None
    elif context:
        result["record_context"] = context
    if unauthorized_leave_consequence_question(message):
        result["intent"] = "policy_question"
        result["leave_focus"] = None
        result["record_context"] = {
            "subject": "policy",
            "intent": "policy_question",
            "policy_topic": "unauthorized_leave",
            "user_id": user.id,
            "employee_id": employee.id,
        }
    return result


def authorize(state: HRState) -> dict:
    ctx = get_ctx()
    user, employee, _department = load_actor(ctx.db, ctx.user_id)
    intent = state.get("intent")
    targets = state.get("target_employee_ids") or []
    kinds = set(state.get("sensitive_kinds") or [])
    reason = ""
    allowed = True
    auth_extras: dict = {}
    message = state.get("message", "")
    requested_org_wide = user.role == "manager" and (
        state.get("requested_org_wide")
        or manager_requested_organization_scope(state.get("raw_message") or "", message)
    )
    if state.get("clarify") and not requested_org_wide:
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
    elif requested_org_wide:
        allowed = False
        intent = "unauthorized_access"
        reason = "Organization-wide workforce analytics require HR Admin access. Managers may access analytics for their direct reports."
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
    elif user.role == "employee" and (
        employee_self_approval_question(message)
        or (intent == "leave_request_status" and parse_approval_decision(message))
    ):
        allowed = False
        intent = "unauthorized_access"
        reason = (
            "Employees cannot approve or reject leave requests, including their own. "
            "Only your assigned manager or HR can approve or reject pending leave. "
            "You may ask about the status of your requests, but you cannot act as the approver."
        )
    elif intent == "approval_decision":
        # Primary security control: deny before tool selection / pending preparation.
        from app.access import approval_decision_access
        from app.services.approval_resolution import nl_approval_resolution

        resolution, approval_ask = nl_approval_resolution(ctx.db, user, employee, message, targets)
        request_id = approval_ask.get("request_id")
        approval_id = approval_ask.get("approval_id")
        if resolution:
            if resolution["status"] == "resolved":
                request_id = resolution.get("request_id")
                auth_extras["resolved_request_id"] = request_id
            elif resolution["status"] == "clarify":
                auth_extras["approval_resolution"] = resolution
            elif resolution["status"] in {"not_found", "unauthorized"}:
                allowed = False
                intent = "unauthorized_access"
                reason = resolution["message"]
        if allowed and resolution and resolution.get("status") == "clarify":
            pass
        elif allowed and request_id is not None:
            ok, gate_reason = approval_decision_access(
                ctx.db,
                user,
                request_id=int(request_id),
                approval_id=int(approval_id) if approval_id is not None else None,
            )
            if not ok:
                allowed = False
                intent = "unauthorized_access"
                reason = gate_reason
        elif allowed and approval_ask and approval_ask.get("request_id") is not None:
            ok, gate_reason = approval_decision_access(
                ctx.db,
                user,
                request_id=int(approval_ask["request_id"]),
            )
            if not ok:
                allowed = False
                intent = "unauthorized_access"
                reason = gate_reason
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
        intent = "unauthorized_access"
        reason = (
            "You are not authorized to access another employee's attendance or private employee record. "
            "This assistant uses the signed-in employee record and does not accept another person or an "
            "employee id from the message."
        )
    elif intent == "org_transfer" and targets and user.role == "employee":
        allowed_ids = {employee.manager_id, employee.approval_manager_id}
        if any(target_id not in allowed_ids for target_id in targets):
            allowed = False
            intent = "unauthorized_access"
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
        reason = (
            "Organization-level aggregate analytics are limited to HR administrators. "
            "Individual attendance and employee records outside your authorized scope are not disclosed."
        )
        intent = "unauthorized_access"
    elif intent == "workforce_analytics" and user.role == "employee":
        # Employees may not run department/org absence analytics; managers stay on team scope.
        allowed = False
        reason = (
            "Organization and department attendance analytics are limited to HR administrators. "
            "Your own attendance remains available through personal attendance questions."
        )
        intent = "unauthorized_access"
    elif intent == "workforce_analytics" and user.role == "manager" and requested_org_wide:
        # Explicit org-wide wording is denied; managers never receive organization attendance extracts.
        allowed = False
        reason = (
            "Organization-level aggregate analytics are limited to HR administrators. "
            "Managers may review attendance patterns for their authorized direct reports only."
        )
        intent = "unauthorized_access"
    elif intent == "org_transfer" and user.role not in {"employee", "manager", "hr_admin"}:
        allowed = False
        reason = "Routing investigations are limited to signed-in employees, managers, and HR administrators."
        intent = "unauthorized_access"
    elif (
        intent == "out_of_scope"
        and user.role != "hr_admin"
        and (
            other_employee_private_request(message)
            or out_of_hierarchy_request(message)
            or outside_authorized_scope_request(message)
            or re.search(
                r"\b(another employee|other employees?|someone else|'s)\b.{0,40}\b"
                r"(attendance|employee record|leave balance|salary|medical|sick)\b",
                message,
                re.IGNORECASE,
            )
        )
    ):
        # Private employee-record probes must hard-deny, never soft-classify as out_of_scope.
        allowed = False
        intent = "unauthorized_access"
        reason = (
            "You are not authorized to access that employee's attendance or private employee record. "
            "This is an authorization denial, not an out-of-scope topic."
        )
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
            **auth_extras,
        }
    return {
        "authorized": True,
        "denial_reason": "",
        "workflow": [_step("authorize", "Security and authorization", "complete", intent or "")],
        "tool_trace": [_tool("authorize", "allowed", intent or "")],
        **auth_extras,
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
    """Prepare approve/reject for human confirmation; execution uses the same decide_approval service as REST."""
    from app.services.actions import _can_decide
    from app.services.approval_resolution import nl_approval_resolution

    ctx = get_ctx()
    user, employee, _department = load_actor(ctx.db, ctx.user_id)
    message = state.get("message") or ""
    parsed = parse_approval_decision(message) or {}
    decision = parsed.get("decision")
    request_id = state.get("resolved_request_id") or parsed.get("request_id")
    reason = parsed.get("reason") or ""
    pending_resolution = state.get("approval_resolution")
    if pending_resolution and pending_resolution.get("status") == "clarify" and request_id is None:
        analysis = {
            "kind": "approval_decision",
            "status": "needs_clarification",
            "message": pending_resolution.get("message") or "",
            "candidates": pending_resolution.get("candidates") or [],
            "confirmation_required": False,
        }
        return {
            "analysis": analysis,
            "pending_action": None,
            "record_context": {
                "subject": "approval_decision",
                "decision": decision,
                "user_id": user.id,
                "employee_id": user.employee_id,
                "subject_employee_id": pending_resolution.get("subject_employee_id"),
            },
            "workflow": [_step("approval_decision", "Approval decision", "complete", "clarify")],
            "tool_trace": [_tool("prepare_approval_decision", "clarify", "multiple pending requests")],
        }
    prepare_flow = approval_inbox_followup_focus(message) in {"prepare_approve", "prepare_reject"}
    resolution = None
    if request_id is None:
        resolution, _parsed = nl_approval_resolution(
            ctx.db,
            user,
            employee,
            message,
            list(state.get("target_employee_ids") or []),
        )
        if resolution and resolution.get("status") == "clarify" and not prepare_flow:
            analysis = {
                "kind": "approval_decision",
                "status": "needs_clarification",
                "message": resolution.get("message") or "",
                "candidates": resolution.get("candidates") or [],
                "confirmation_required": False,
            }
            return {
                "analysis": analysis,
                "pending_action": None,
                "record_context": {
                    "subject": "approval_decision",
                    "decision": decision,
                    "user_id": user.id,
                    "employee_id": user.employee_id,
                    "subject_employee_id": resolution.get("subject_employee_id"),
                },
                "workflow": [_step("approval_decision", "Approval decision", "complete", "clarify")],
                "tool_trace": [_tool("prepare_approval_decision", "clarify", "multiple pending requests")],
            }
        if resolution and resolution.get("status") == "resolved":
            request_id = resolution.get("request_id")
        if resolution and resolution.get("status") == "not_found" and prepare_flow:
            analysis = {
                "kind": "approval_decision",
                "status": "not_found",
                "message": resolution.get("message") or "",
                "confirmation_required": False,
            }
            return {
                "analysis": analysis,
                "pending_action": None,
                "record_context": {
                    "subject": "approval_decision",
                    "decision": decision,
                    "user_id": user.id,
                    "employee_id": user.employee_id,
                },
                "workflow": [_step("approval_decision", "Approval decision", "complete", "not found")],
                "tool_trace": [_tool("prepare_approval_decision", "clarify", "no direct report pending")],
            }
    inherited = state.get("record_context") or {}
    if request_id is None:
        for key in ("selected_request_id", "request_id"):
            if inherited.get(key) is not None and inherited.get("subject") in {
                "approval_decision",
                "approval_inbox",
                "team_analytics",
            }:
                request_id = int(inherited[key])
                break
    if decision is None and inherited.get("subject") == "approval_decision" and inherited.get("decision"):
        decision = inherited["decision"]
    approval_follow = approval_inbox_followup_focus(message)
    if decision is None and approval_follow == "prepare_approve":
        decision = "approve"
    if decision is None and approval_follow == "prepare_reject":
        decision = "reject"
    if request_id is None and (approval_follow in {"prepare_approve", "prepare_reject"} or prepare_flow):
        from app.services.approval_resolution import (
            first_decidable_direct_report_leave_request,
            first_decidable_leave_request,
        )

        auto_id = first_decidable_direct_report_leave_request(ctx.db, user, employee)
        if auto_id is None:
            auto_id = first_decidable_leave_request(ctx.db, user, employee)
        if auto_id is not None:
            request_id = auto_id
    if decision == "reject" and not reason:
        if approval_follow == "prepare_reject":
            reason = "Operational coverage concern (manager draft)."
        else:
            analysis = {
                "kind": "approval_decision",
                "status": "needs_reason",
                "decision": decision,
                "request_id": request_id,
                "message": "A rejection reason is required. Example: reject request 12 because coverage is insufficient.",
                "confirmation_required": True,
            }
            return {
                "analysis": analysis,
                "pending_action": None,
                "record_context": {
                    "subject": "approval_decision",
                    "decision": decision,
                    "request_id": request_id,
                    "user_id": user.id,
                    "employee_id": user.employee_id,
                },
                "workflow": [_step("approval_decision", "Approval decision", "complete", "reason required")],
                "tool_trace": [_tool("prepare_approval_decision", "clarify", "rejection reason required")],
            }
    if decision is None or request_id is None:
        analysis = {
            "kind": "approval_decision",
            "status": "needs_clarification",
            "message": "Specify the leave request id and whether to approve or reject it. Example: approve leave request 12.",
            "confirmation_required": True,
        }
        return {
            "analysis": analysis,
            "pending_action": None,
            "workflow": [_step("approval_decision", "Approval decision", "complete", "clarification")],
            "tool_trace": [_tool("prepare_approval_decision", "clarify", "missing request id or decision")],
        }
    from app.access import approval_decision_access, resolve_leave_approval_reference

    ok, gate_reason = approval_decision_access(ctx.db, user, request_id=int(request_id))
    if not ok:
        analysis = {
            "kind": "approval_decision",
            "status": "denied",
            "request_id": request_id,
            "decision": decision,
            "message": gate_reason,
            "confirmation_required": False,
        }
        return {
            "analysis": analysis,
            "pending_action": None,
            "workflow": [_step("approval_decision", "Approval decision", "denied", gate_reason[:180])],
            "tool_trace": [_tool("prepare_approval_decision", "denied", gate_reason[:120])],
        }
    approval = resolve_leave_approval_reference(ctx.db, request_id=int(request_id))
    leave = ctx.db.get(LeaveRequest, int(request_id)) if request_id is not None else None
    if approval is None:
        analysis = {
            "kind": "approval_decision",
            "status": "not_found",
            "request_id": request_id,
            "decision": decision,
            "message": f"No pending approval was found for request {request_id}.",
            "confirmation_required": True,
        }
        return {
            "analysis": analysis,
            "pending_action": None,
            "record_context": {
                "subject": "approval_decision",
                "request_id": request_id,
                "decision": decision,
                "user_id": user.id,
                "employee_id": user.employee_id,
            },
            "workflow": [_step("approval_decision", "Approval decision", "complete", "not found")],
            "tool_trace": [_tool("prepare_approval_decision", "denied", "approval not found")],
        }
    if approval.status != "pending":
        analysis = {
            "kind": "approval_decision",
            "status": "duplicate",
            "request_id": request_id,
            "approval_id": approval.id,
            "decision": decision,
            "message": f"Approval {approval.id} is already {approval.status}. It cannot be decided again.",
            "confirmation_required": True,
        }
        return {
            "analysis": analysis,
            "pending_action": None,
            "workflow": [_step("approval_decision", "Approval decision", "complete", "already decided")],
            "tool_trace": [_tool("prepare_approval_decision", "denied", "already decided")],
        }
    if not _can_decide(user, approval):
        analysis = {
            "kind": "approval_decision",
            "status": "denied",
            "request_id": request_id,
            "approval_id": approval.id,
            "decision": decision,
            "message": "You are not the assigned approver for this request.",
            "confirmation_required": True,
        }
        return {
            "analysis": analysis,
            "pending_action": None,
            "workflow": [_step("approval_decision", "Approval decision", "complete", "denied")],
            "tool_trace": [_tool("prepare_approval_decision", "denied", "not assigned approver")],
        }

    from app.services.confirmation import ConfirmationError

    try:
        pending = prepare_decision_pending(
            ctx.db,
            user,
            conversation_id=getattr(ctx, "conversation_id", None),
            approval_id=approval.id,
            request_id=leave.id if leave is not None else request_id,
            decision=decision,
            reason=reason,
            thread_id=getattr(ctx, "thread_id", None),
            ip=getattr(ctx, "ip", None),
        )
    except ConfirmationError as exc:
        analysis = {
            "kind": "approval_decision",
            "status": "denied",
            "request_id": request_id,
            "approval_id": approval.id,
            "decision": decision,
            "message": exc.message,
            "confirmation_required": False,
        }
        return {
            "analysis": analysis,
            "pending_action": None,
            "workflow": [_step("approval_decision", "Approval decision", "denied", exc.message[:180])],
            "tool_trace": [_tool("prepare_approval_decision", "denied", exc.message[:120])],
        }
    if pending.get("already_executed"):
        prior = pending.get("prior_result") or {}
        analysis = {
            "kind": "approval_decision",
            "status": "duplicate",
            "decision": decision,
            "request_id": request_id,
            "approval_id": approval.id,
            "message": prior.get("message")
            or f"This {decision} was already executed earlier. Helios will not repeat it.",
            "confirmation_required": False,
        }
        return {
            "analysis": analysis,
            "pending_action": None,
            "execution_result": prior,
            "workflow": [_step("approval_decision", "Approval decision", "complete", "duplicate")],
            "tool_trace": [_tool("prepare_approval_decision", "ok", "already executed")],
        }

    analysis = {
        "kind": "approval_decision",
        "status": "awaiting_confirmation",
        "decision": decision,
        "request_id": leave.id if leave is not None else request_id,
        "approval_id": approval.id,
        "reason": reason,
        "confirmation_required": True,
        "message": (
            f"I prepared a {decision} for leave request {leave.id if leave is not None else request_id} "
            f"(approval {approval.id}). Confirm to apply it through the same service as the Approvals API. "
            "No record has been changed yet."
        ),
    }
    return {
        "analysis": analysis,
        "pending_action": pending,
        "record_context": {
            "subject": "approval_decision",
            "request_id": analysis.get("request_id"),
            "approval_id": approval.id,
            "decision": decision,
            "status": "awaiting_confirmation",
            "user_id": user.id,
            "employee_id": user.employee_id,
        },
        "workflow": [_step("approval_decision", "Approval decision", "complete", "awaiting confirmation")],
        "tool_trace": [_tool("prepare_approval_decision", "ok", f"pending:{pending.get('pending_action_id')}")],
    }


def leave_analysis(state: HRState) -> dict:
    ctx = get_ctx()
    user, employee, _department = load_actor(ctx.db, ctx.user_id)
    message = state.get("message") or ""
    if state.get("intent") == "team_conflict":
        from app.services.leave_engine import resolve_team_overlap_window

        window_start, window_end, window_label, window_err = resolve_team_overlap_window(message)
        if window_err:
            analysis = {
                "kind": "team_conflict",
                "status": window_err.get("status", "needs_clarification"),
                "message": window_err.get("message", "Specify one calendar month to review."),
            }
            return {
                "analysis": analysis,
                "record_context": {
                    "subject": "team_conflict",
                    "user_id": user.id,
                    "employee_id": employee.id,
                    "focus": "overlap",
                },
                "workflow": [_step("leave_analysis", "Leave and attendance analysis", "clarify", window_err.get("message", ""))],
                "tool_trace": [_tool("analyze_team_conflicts", "clarify", "overlap window")],
            }
        analysis = analyze_team(
            ctx.db,
            employee,
            filter_start=window_start,
            filter_end=window_end,
            filter_label=window_label or None,
        )
        pair_count = len(analysis.get("overlap_pairs") or [])
        request_count = len(analysis.get("requests") or [])
        if pair_count:
            summary = f"{pair_count} overlapping leave pair(s); densest week has {request_count} request(s)"
        else:
            summary = f"{request_count} requests in {analysis.get('window', {}).get('label', 'the review window')}"
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

    parsed = parse_leave_dates(message)
    context = state.get("record_context") or {}
    focus = state.get("leave_focus") or ""
    if leave_date_recommendation_question(message):
        focus = "recommend_dates"
    leave_type = (parsed or {}).get("leave_type") or context.get("leave_type") or "annual"
    stated = parse_business_days(message)
    # "How many business days should I plan?" is a question about the current window, not a new duration.
    if focus == "business_days" or re.search(r"\bhow many business days should i plan\b", message.lower()):
        stated = None
    if stated is None and context.get("subject") == "leave_eligibility" and context.get("requested_days"):
        stated = int(context["requested_days"])
    if stated is None and context.get("subject") == "leave_plan" and context.get("business_days"):
        stated = int(context["business_days"])
    if focus == "recommend_dates":
        analysis = recommend_leave_dates(
            ctx.db,
            employee,
            business_days=stated,
            leave_type=leave_type,
        )
        requested = analysis.get("requested_window") or {}
        summary = (
            f"Recommended leave window: {requested.get('label')}"
            if requested
            else "Leave date recommendation"
        )
        record_context = {
            "subject": "leave_plan",
            "intent": "leave_planning",
            "user_id": user.id,
            "employee_id": employee.id,
            "leave_type": leave_type,
            "start": requested.get("start"),
            "end": requested.get("end"),
            "label": requested.get("label"),
            "business_days": analysis.get("assumed_business_days") or requested.get("business_days"),
            "remaining_before": (analysis.get("balance") or {}).get("remaining"),
            "workflow_state": "RECOMMENDATION",
            "hard_blocks": requested.get("hard_blocks") or [],
            "focus": "recommend_dates",
            "confirmation_required": False,
        }
        return {
            "analysis": analysis,
            "record_context": record_context,
            "leave_focus": "recommend_dates",
            "pending_action": None,
            "workflow": [_step("leave_analysis", "Leave date recommendation", "complete", summary)],
            "tool_trace": [_tool("recommend_leave_dates", "ok", summary)],
        }
    start = parsed.get("start") if parsed else None
    end = parsed.get("end") if parsed else None
    if start is None and context.get("subject") == "leave_plan" and context.get("start"):
        start = date.fromisoformat(context["start"])
    if end is None and context.get("subject") == "leave_plan" and context.get("end") and stated is None:
        end = date.fromisoformat(context["end"])
    # Duration resolution: explicit count wins; a single named date means one day;
    # an inherited plan window keeps its length; only then fall back to classify default.
    days = stated
    if days is None and start is not None and end is None:
        days = 1
    if days is None and context.get("subject") == "leave_plan" and context.get("business_days"):
        days = int(context["business_days"])
    if days is None:
        days = int(state.get("business_days") or 10)
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
    raw = state.get("raw_message") or state.get("message") or ""
    if user.role == "manager" and (
        state.get("requested_org_wide")
        or manager_requested_organization_scope(raw, state.get("message") or "")
    ):
        reason = (
            "Organization-level aggregate analytics are limited to HR administrators. "
            "Managers may review attendance patterns for their authorized direct reports only."
        )
        return {
            "intent": "unauthorized_access",
            "authorized": False,
            "denial_reason": reason,
            "analysis": None,
            "analytics": {},
            "workflow": [_step("workforce_analytics", "Workforce analytics", "denied", "organization-wide scope blocked")],
            "tool_trace": [],
        }
    message = (state.get("message") or "").lower()
    months = 3
    inherited = _usable_context(state, user.id, employee.id) or {}
    if "six month" in message:
        months = 6
    elif inherited.get("months"):
        months = int(inherited["months"])
    # Scope is enforced inside absence_analytics before any attendance rows are loaded.
    payload = absence_analytics(ctx.db, user, months=months)
    prior_focus = inherited.get("focus") or ""
    focus = workforce_analytics_focus(
        message,
        prior_focus=prior_focus,
        subject=str(inherited.get("subject") or ""),
    )
    if focus == "remaining_balances":
        payload = {
            **payload,
            "department_remaining": _department_remaining_leave(
                ctx.db,
                employee_ids=payload.get("employee_ids"),
                scope=payload.get("scope"),
            ),
        }
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
        "intent": "workforce_analytics",
        "analytics": payload,
        "analysis": {
            "kind": "workforce_analytics",
            "focus": focus,
            "period": payload.get("period"),
            "scope": payload.get("scope"),
            "scope_label": payload.get("scope_label"),
            "employee_ids": payload.get("employee_ids") or [],
            "method": payload.get("method"),
            "leave_utilization": payload.get("leave_utilization"),
            "department_remaining": payload.get("department_remaining"),
            "departments": payload.get("departments"),
            "anomalies": payload.get("anomalies"),
            "month_comparison": month_cmp,
            "office_comparison": office_cmp,
            "investigation": payload.get("investigation"),
            "priority_employees": payload.get("priority_employees") or [],
            "recommendations": payload.get("recommendations"),
            "workforce_risk_findings": payload.get("workforce_risk_findings"),
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
        "workflow": [_step("workforce_analytics", "Workforce analytics", "complete", f"{payload['scope']}:{focus}")],
        "tool_trace": [_tool("compute_absence_analytics", "ok", f"{focus}: {summary[:160]}")],
    }


def _department_remaining_leave(
    db,
    *,
    employee_ids: list[int] | None = None,
    scope: str | None = None,
) -> list[dict]:
    """Department totals of remaining annual leave for utilization follow-ups.

    When scope is not organization, only the pre-authorized employee_ids are included.
    """
    from collections import defaultdict

    from app.models import LeaveBalance

    year = org_today().year
    employee_query = db.query(Employee).filter(Employee.employment_status == "active")
    if scope != "organization":
        ids = list(employee_ids or [])
        if not ids:
            return []
        employee_query = employee_query.filter(Employee.id.in_(ids))
    employees = employee_query.all()
    dept_names = {row.id: row.name for row in db.query(Department).all()}
    emp_dept = {row.id: dept_names.get(row.department_id, "") for row in employees}
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"entitled": 0.0, "used": 0.0, "pending": 0.0, "remaining": 0.0, "employees": 0}
    )
    balance_query = db.query(LeaveBalance).filter(
        LeaveBalance.leave_type == "annual",
        LeaveBalance.year == year,
    )
    if scope != "organization":
        balance_query = balance_query.filter(LeaveBalance.employee_id.in_(list(emp_dept.keys()) or [-1]))
    balances = balance_query.all()
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
            selected_request_id = payload.get("selected_request_id")
            if selected_request_id is None and (payload.get("highest") or {}).get("request_id"):
                selected_request_id = payload["highest"]["request_id"]
            if selected_request_id is None and inherited and inherited.get("selected_request_id"):
                selected_request_id = inherited.get("selected_request_id")
            context = {
                "subject": "team_analytics"
                if not str(payload.get("focus") or "").startswith("awaiting_approval")
                else "approval_inbox",
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
                "selected_request_id": selected_request_id,
                "request_id": selected_request_id,
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
            if payload.get("filter") in {"latest", "delayed_approval"} and payload.get("requests"):
                kept_request = payload["requests"][0]["id"]
            context = {
                "subject": "leave_history",
                "intent": "leave_request_status",
                "user_id": user.id,
                "employee_id": employee.id,
                "request_id": kept_request,
                "filter": payload.get("filter"),
                "focus": payload.get("focus") or payload.get("filter"),
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
    if unauthorized_leave_consequence_question(text):
        return "unauthorized_leave"
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
    if intent == "team_analytics":
        focus = ((state.get("analysis") or {}).get("focus") or "").lower()
        if focus in {"awaiting_approval_approve_guidance", "awaiting_approval_reject_guidance"}:
            return "leave approval policy manager coverage balance co-approval rejection consequences"
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
        if unauthorized_leave_consequence_question(message) or re.search(
            r"absent without|without prior approval|unplanned absence", text
        ):
            return (
                "attendance policy unplanned absence notify manager leave approved in advance "
                "leave approval confirmed HRIS pending not approved"
            )
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
        "consequences",
        "highest_department",
        "evidence",
        "absence",
        "department_breakdown",
        "workforce_risk_plan",
        "corrective_prepare",
        "proposed_action",
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
    unauthorized_leave = unauthorized_leave_consequence_question(state.get("message") or "")
    if unauthorized_leave:
        result = retrieve(
            ctx.db,
            "attendance policy unplanned absence weekday notify manager approved in advance not confirmed HRIS",
            role=user.role,
            department=department.name if department else None,
            location=employee.location,
        )
        result = dict(result)
        result["conflicts"] = []
        if result.get("confidence") == "conflict":
            result["confidence"] = "adequate"
    else:
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
        elif re.search(r"absent without|without prior approval", lowered) or unauthorized_leave_consequence_question(
            message
        ):
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
        elif topic == "unauthorized_leave":
            stored["analysis"] = {"kind": "unauthorized_leave_consequence"}
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
    from app.clock import add_business_days, notice_business_days, org_today
    from app.services.holidays import holidays_for_employee
    from app.services.leave_engine import assess_window, notice_required
    from app.services.records import parse_relative_leave_start

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
    proposed_start = parse_relative_leave_start(message)
    proposed_window: dict | None = None
    if proposed_start is not None:
        today = org_today()
        horizon_end = proposed_start + timedelta(days=120)
        closed = holidays_for_employee(ctx.db, employee, today, horizon_end).dates
        notice = notice_business_days(today, proposed_start, closed)
        if requested:
            leave_end = add_business_days(proposed_start, int(requested), closed)
            proposed_window = assess_window(
                ctx.db, employee, proposed_start, leave_end, leave_type="annual"
            )
            proposed_window = {
                **proposed_window,
                "proposed_start": proposed_start.isoformat(),
                "proposed_end": leave_end.isoformat(),
                "requested_days": requested,
            }
        else:
            required = notice_required(5)
            proposed_window = {
                "proposed_start": proposed_start.isoformat(),
                "notice_business_days": notice,
                "notice_required_example": required,
                "notice_ok_for_example": notice >= required,
                "duration_specified": False,
            }
    return {
        "analysis": {
            "kind": "leave_eligibility",
            "employee_id": employee.id,
            "balance": annual,
            "requested_days": requested,
            "hypothetical_remaining": hypothetical_remaining_days(message),
            "proposed_window": proposed_window,
        },
        "record_context": {
            "subject": "leave_eligibility",
            "user_id": _user.id,
            "employee_id": employee.id,
            "requested_days": requested,
            "start": (proposed_window or {}).get("proposed_start") or (proposed_window or {}).get("start"),
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
    pending = state.get("pending_action")
    detail = "no consequential action"
    analysis = state.get("analysis") or {}
    focus = state.get("leave_focus") or analysis.get("focus") or ""
    ctx = get_ctx()
    user, employee, _department = load_actor(ctx.db, ctx.user_id)
    analytics_payload = state.get("analytics")
    # Review / recommendations / coverage checks never authorize submission alone.
    defer_pending = focus in {
        "draft_review",
        "confirmation_gate",
        "holidays",
        "projects",
        "approvals",
        "business_days",
        "recommend_dates",
        "coverage",
        "prerequisites",
        "balance_check",
        "window",
        "duration",
    } or analysis.get("focus") in {
        "recommend_dates",
        "coverage",
    } or analysis.get("workflow_state") == "RECOMMENDATION" or analysis.get(
        "confirmation_required"
    ) is False
    if pending and pending.get("pending_action_id"):
        detail = f"reuse pending:{pending.get('pending_action_id')}"
    elif intent in _REQUEST_INTENTS:
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
            pending = prepare_submit_leave_pending(
                ctx.db,
                user,
                conversation_id=getattr(ctx, "conversation_id", None),
                options=options,
                default_option="primary",
                thread_id=getattr(ctx, "thread_id", None),
                ip=getattr(ctx, "ip", None),
            )
            if pending.get("already_executed"):
                prior = pending.get("prior_result") or {}
                detail = "duplicate; prior submission exists"
                window_label = (options.get("primary") or {}).get("label") or "that leave window"
                return {
                    "analysis": {
                        **analysis,
                        "status": "duplicate",
                        "message": prior.get("message")
                        or f"{window_label} was already submitted. Helios will not create a duplicate.",
                    },
                    "pending_action": None,
                    "execution_result": prior,
                    "workflow": [_step("prepare_workflow", "Workflow preparation", "complete", detail)],
                    "tool_trace": [_tool("prepare_action", "ok", detail)],
                }
            detail = "awaiting confirmation"
        elif requested and requested.get("hard_blocks"):
            detail = "blocked; draft not eligible"
        elif defer_pending:
            detail = f"leave focus={focus}; no submission"
    elif intent == "org_transfer":
        analysis = state.get("analysis") or {}
        if not analysis:
            analysis = investigate_routing(ctx.db, employee)
        # Read-only investigation / recommendation / urgency questions never prepare a mutation.
        # Only an explicit "file/apply/correct" request may create a pending routing correction.
        if analysis.get("correction") and explicit_routing_correction_request(state.get("message") or ""):
            pending = prepare_routing_pending(
                ctx.db,
                user,
                conversation_id=getattr(ctx, "conversation_id", None),
                correction=analysis["correction"],
                thread_id=getattr(ctx, "thread_id", None),
                ip=getattr(ctx, "ip", None),
            )
            detail = "awaiting HR filing confirmation"
        else:
            detail = "read-only routing investigation; no pending action"
    elif intent == "workforce_analytics" and workforce_corrective_prepare_question(state.get("message") or ""):
        analytics_payload = state.get("analytics") or {}
        if not analytics_payload.get("departments"):
            months = int((state.get("record_context") or {}).get("months") or 3)
            analytics_payload = absence_analytics(ctx.db, user, months=months)
        proposal = workforce_corrective_proposal(analytics_payload)
        if proposal and user.role == "hr_admin":
            pending = prepare_workforce_corrective_pending(
                ctx.db,
                user,
                conversation_id=getattr(ctx, "conversation_id", None),
                proposal=proposal,
                thread_id=getattr(ctx, "thread_id", None),
                ip=getattr(ctx, "ip", None),
            )
            detail = "awaiting HR corrective confirmation"
            analysis = {
                **(state.get("analysis") or {}),
                "kind": "workforce_analytics",
                "focus": "corrective_prepare",
                "proposal": proposal,
            }
        else:
            detail = "corrective prepare blocked or no analytics"
    elif intent == "approval_decision" and (state.get("pending_action") or {}).get("pending_action_id"):
        pending = state.get("pending_action")
        detail = f"approval pending:{pending.get('pending_action_id')}"
    if intent == "org_transfer" or (
        intent == "workforce_analytics"
        and workforce_corrective_prepare_question(state.get("message") or "")
    ):
        patch_analysis = analysis
    else:
        patch_analysis = state.get("analysis") or analysis

    record_context = dict(state.get("record_context") or {})
    if intent == "workforce_analytics" and pending:
        proposal = (patch_analysis or {}).get("proposal") or (pending or {}).get("proposal") or {}
        record_context = {
            **record_context,
            "subject": "workforce_analytics",
            "focus": "corrective_prepare",
            "user_id": user.id,
            "employee_id": employee.id,
            "months": proposal.get("months") or record_context.get("months") or 3,
        }
    patch: dict = {
        "analysis": patch_analysis,
        "pending_action": pending,
        "record_context": record_context,
        "workflow": [_step("prepare_workflow", "Workflow preparation", "complete", detail)],
        "tool_trace": [_tool("prepare_action", "ok" if pending else "skipped", detail)],
    }
    if (
        intent == "workforce_analytics"
        and workforce_corrective_prepare_question(state.get("message") or "")
        and analytics_payload
    ):
        patch["analytics"] = analytics_payload
    return patch


def compose_response(state: HRState) -> dict:
    """Compose the user-facing answer. Tool results stay authoritative; LLM may explain."""
    intent = state.get("intent") or ""
    mode = state.get("orchestration_mode") or "deterministic"
    enriched = dict(state)
    if not enriched.get("role") or not enriched.get("employee_context"):
        try:
            ctx = get_ctx()
            user, employee, department = load_actor(ctx.db, ctx.user_id)
            enriched["role"] = user.role
            if not enriched.get("employee_context"):
                enriched["employee_context"] = {
                    "full_name": employee.full_name,
                    "job_title": employee.job_title,
                    "department": department.name if department else None,
                    "location": employee.location,
                    "employee_number": employee.employee_number,
                    "employment_status": employee.employment_status,
                }
        except Exception:
            pass

    denials = (
        not enriched.get("authorized", True)
        or intent in {"unauthorized_access", "unsupported_mutation", "out_of_scope"}
        or enriched.get("clarify")
    )
    analysis_focus = (enriched.get("analysis") or {}).get("focus") or ""
    analysis_kind = (enriched.get("analysis") or {}).get("kind") or ""
    force_deterministic = analysis_focus in {
        "absent_employees",
        "period_change",
        "period_switch",
        "department_contribution",
    } or analysis_kind in {
        "routing_audit",
        "routing_investigation",
        "approval_decision",
        "approval_inbox",
        "tool_clarification",
        # Keep leave-overlap pairs, request IDs, and coverage risk wording intact.
        "team_conflict",
    } or intent == "team_conflict" or analysis_focus in {
        "awaiting_approval",
        "awaiting_approval_risk",
        "awaiting_approval_why_urgent",
        "awaiting_approval_impact",
        "awaiting_approval_reject_impact",
        "awaiting_approval_approve_guidance",
        "awaiting_approval_reject_guidance",
        "approval_inbox",
        "recommend_dates",
        "delayed_approval",
    } or analysis_kind == "unauthorized_leave_consequence" or (enriched.get("analysis") or {}).get("filter") == "delayed_approval" or (
        enriched.get("leave_focus") == "recommend_dates"
    ) or (enriched.get("pending_action") or {}).get("pending_action_id") or (
        # Keep org absence counts from tools deterministic so the LLM cannot drop months.
        analysis_kind in {"org_attendance", "attendance"}
        and analysis_focus in {"absent_employees", "absent_rows", "summary", "department", ""}
    )

    text, summary = narrate(enriched)
    use_tool_llm = intent in TOOL_EXPLAIN_INTENTS or intent in KNOWLEDGE_INTENTS or mode.startswith("tool_")
    if use_tool_llm and not force_deterministic and not denials:
        fact_state = dict(enriched)
        fact_state["decision_summary"] = summary
        fact_state["analysis"] = {
            **(fact_state.get("analysis") or {}),
            "deterministic_briefing": text,
            "deterministic_summary": summary,
        }
        grounded = generate_grounded_answer(fact_state, require_llm=False)
        if grounded.mode == "llm":
            return {
                "response": grounded.answer,
                "decision_summary": grounded.summary or summary,
                "orchestration_mode": "llm" if not mode.startswith("tool_") else mode,
                "workflow": [_step("compose_response", "Response generation", "complete", "llm")],
                "tool_trace": [_tool("openrouter_grounded_answer", "ok", "explained tool results")],
            }
    elif not denials and intent in KNOWLEDGE_INTENTS and not force_deterministic:
        grounded = generate_grounded_answer(enriched, require_llm=False)
        if grounded.mode == "llm":
            return {
                "response": grounded.answer,
                "decision_summary": grounded.summary or summary,
                "orchestration_mode": "llm",
                "workflow": [_step("compose_response", "Response generation", "complete", "llm")],
                "tool_trace": [_tool("openrouter_grounded_answer", "ok", grounded.summary[:180] if grounded.summary else "ok")],
            }
    return {
        "response": text,
        "decision_summary": summary,
        "orchestration_mode": mode if mode != "deterministic" else "deterministic",
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
    user, employee, _department = load_actor(ctx.db, ctx.user_id)
    pending = state.get("pending_action") or {}
    pending_id = pending.get("pending_action_id")
    row = get_pending(ctx.db, pending_id)
    if row is None and pending.get("type") in {
        "submit_leave",
        "decide_approval",
        "file_routing_correction",
        "record_hr_corrective_recommendation",
    }:
        # Legacy in-memory pending without durable id — refuse rather than write blindly.
        result = {
            "status": "error",
            "message": "The pending action could not be loaded for revalidation. No record was changed.",
            "summary": "Missing durable pending action.",
        }
    elif row is None:
        result = {"status": "error", "message": "That action is not available.", "summary": "No record was changed."}
    else:
        result = execute_pending(
            ctx.db,
            user,
            row,
            option_id=state.get("selected_option"),
            ip=ctx.ip,
        )
    status = result.get("status", "error")
    # Never report failure as success.
    if status in {"error", "failed", "denied", "blocked", "expired", "duplicate", "needs_reason", "not_found"}:
        tool_status = "error" if status in {"error", "failed"} else status
    else:
        tool_status = "ok"
    patch: dict = {
        "execution_result": result,
        "pending_action": None,
        "analysis": {
            **(state.get("analysis") or {}),
            "kind": (state.get("analysis") or {}).get("kind") or pending.get("type") or "execution",
            "status": status,
            "message": result.get("message"),
        },
        "workflow": [_step("execute_workflow", "Workflow execution", status, (result.get("summary") or "")[:120])],
        "tool_trace": [_tool("execute_authorized_action", tool_status, (result.get("summary") or result.get("message") or "")[:180])],
    }
    if status in {"submitted", "pending_manager", "pending_hr", "approved", "rejected"}:
        request_id = result.get("leave_request_id") or (result.get("result") or {}).get("leave_request_id")
        if request_id:
            patch["record_context"] = leave_history_record_context(
                user_id=user.id,
                employee_id=employee.id,
                request_id=int(request_id),
                report_filter="latest",
            )
            patch["intent"] = "leave_request_status"
    return patch


def cancel_workflow(state: HRState) -> dict:
    ctx = get_ctx()
    user, _employee, _department = load_actor(ctx.db, ctx.user_id)
    pending = state.get("pending_action") or {}
    row = get_pending(ctx.db, pending.get("pending_action_id"))
    if row is not None:
        result = cancel_pending(ctx.db, row, user=user, ip=ctx.ip, reason="user_declined")
    else:
        write_audit(
            ctx.db,
            actor_user_id=ctx.user_id,
            action="workflow_cancelled",
            resource_type="assistant",
            resource_id=str(pending.get("type") or ""),
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


def tool_orchestrate(state: HRState) -> dict:
    """LLM/deterministic tool selection → validate → authorize → execute HR services."""
    # Lazy import avoids services ↔ graph ↔ orchestrate circular import at module load.
    from app.ai.orchestrate import run_tool_orchestration

    ctx = get_ctx()
    user, employee, department = load_actor(ctx.db, ctx.user_id)
    message = state.get("message") or ""
    intent = state.get("intent")
    profile = state.get("employee_context") or profile_for(ctx.db, user, employee, employee, private=False)
    if department is not None and not profile.get("department"):
        profile = {**profile, "department": department.name}
    inherited = state.get("record_context")
    if inherited and inherited.get("user_id") not in {None, user.id}:
        inherited = None
    if inherited and inherited.get("employee_id") not in {None, employee.id}:
        inherited = None

    # Prefer deterministic tool plans for numeric org analytics / structured approvals so
    # the LLM cannot invent filters that change authoritative counts.
    prefer_llm = intent not in {
        "org_attendance",
        "approval_decision",
        "org_transfer",
        "leave_balance",
        "attendance_query",
        "team_conflict",
    }
    # Keep pending-approval listing/urgency on the deterministic inbox planner.
    if intent == "team_analytics" and (
        approval_inbox_question(message) or approval_inbox_followup_focus(message) is not None
    ):
        prefer_llm = False
    outcome = run_tool_orchestration(
        ctx.db,
        user,
        message=message,
        intent=intent,
        employee_context=profile,
        conversation_history=state.get("conversation_history") or [],
        citations=state.get("citations") or [],
        record_context=inherited,
        leave_focus=state.get("leave_focus"),
        ip=getattr(ctx, "ip", None),
        prefer_llm=prefer_llm,
    )
    patch = dict(outcome.get("state_patch") or {})
    selected = outcome.get("selected_tools") or []
    detail = ",".join(selected) if selected else "no_tools"
    if "intent" not in patch:
        patch["intent"] = intent
    if not patch.get("employee_context"):
        patch["employee_context"] = profile
    if patch.get("clarify") and not patch.get("analysis"):
        patch["analysis"] = {
            "kind": "tool_clarification",
            "message": patch.get("clarification") or "I need more information to run the HR tool.",
            "missing_args": [
                arg
                for item in (outcome.get("results") or [])
                for arg in (getattr(item, "missing_args", None) or [])
            ],
        }
    traces = list(patch.pop("tool_trace", None) or [])
    return {
        **patch,
        "workflow": [
            _step("tool_orchestrate", "Authorized context", "complete", profile.get("full_name") or user.role),
            _step("tool_orchestrate", "Tool selection", "complete", (patch.get("tool_plan") or {}).get("mode") or "planner"),
            _step("tool_orchestrate", "Tool execution", "complete", detail),
        ],
        "tool_trace": traces,
    }


def route_after_auth(state: HRState) -> str:
    if not state.get("authorized", False) or state.get("clarify"):
        return "compose_response"
    if state.get("role") == "manager" and state.get("requested_org_wide"):
        return "compose_response"
    intent = state.get("intent", "")
    message = state.get("message") or ""
    # Focused org-wide absence analytics (highest department, consequences, investigation)
    # must use the workforce_analytics node — not the generic last-7-days attendance tool.
    # Month-over-month "contributed most / how much did it increase" stay on org_attendance tools.
    raw = state.get("raw_message") or message
    comparison_followup = bool(
        re.search(
            r"\b(contributed most|department contributed|how much did it (?:increase|decrease))\b",
            f"{raw} {message}",
            re.I,
        )
    )
    if intent == "workforce_analytics" and not comparison_followup:
        return "workforce_analytics"
    if (
        intent == "org_attendance"
        and workforce_followup_question(message)
        and not comparison_followup
        and not org_attendance_month_comparison_question(message)
        and not org_attendance_named_month_rate_question(message)
    ):
        return "workforce_analytics"
    if intent == "workforce_analytics" and comparison_followup:
        # Re-anchor comparison asks onto the org_attendance tool path.
        return "tool_orchestrate"
    # Employees investigate their own routing via the dedicated Postgres path — the HR
    # investigate_employee_discrepancy tool is org-wide and denied for employees.
    if intent == "org_transfer" and state.get("role") == "employee":
        return "employee_context"
    if intent == "approval_decision":
        return "approval_decision"
    if intent == "leave_eligibility":
        return "assess_eligibility"
    # Approval inbox listing / urgency follow-ups are classified as team_analytics; still run tools.
    if intent in _TOOL_ORCHESTRATED_INTENTS or (
        intent == "team_analytics"
        and (
            approval_inbox_question(message)
            or manager_leave_decision_guidance_question(message)
            or approval_inbox_followup_focus(message)
            in {
                "urgency",
                "why_urgent",
                "approve_impact",
                "reject_impact",
                "approve_guidance",
                "reject_guidance",
            }
        )
    ):
        return "tool_orchestrate"
    return {
        "leave_planning": "employee_context",
        "leave_request": "employee_context",
        "self_service": "employee_context",
        "employee_profile": "lookup_records",
        "employee_directory": "lookup_records",
        "team_roster": "lookup_records",
        "team_analytics": "lookup_records",
        "org_leave": "lookup_records",
        "compensation_query": "lookup_records",
        "hr_metrics": "lookup_records",
        "hr_action": "lookup_records",
        "leave_request_status": "lookup_records",
        "leave_cancellation": "lookup_records",
        "workforce_analytics": "workforce_analytics",
        "out_of_scope": "compose_response",
        "unsupported_mutation": "compose_response",
        "data_semantics": "compose_response",
        "confidential_lookup": "employee_context",
    }.get(intent, "retrieve_policy")


def route_after_tools(state: HRState) -> str:
    intent = state.get("intent") or ""
    analysis = state.get("analysis") or {}
    focus = state.get("leave_focus") or analysis.get("focus") or ""
    if intent == "leave_request" or (
        (intent == "leave_planning" or analysis.get("kind") == "leave_plan")
        and focus in {"prepare", "draft_review", "confirmation_gate"}
    ):
        return "prepare_workflow"
    # Routing corrections require an explicit file/apply/correct request — never auto-HITL on investigate.
    if (
        intent == "org_transfer"
        and analysis.get("correction")
        and explicit_routing_correction_request(state.get("message") or "")
    ):
        return "prepare_workflow"
    if intent == "workforce_analytics" and workforce_corrective_prepare_question(state.get("message") or ""):
        return "prepare_workflow"
    return "compose_response"


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
    user, employee, _department = load_actor(ctx.db, ctx.user_id)
    targets = state.get("target_employee_ids") or []
    message = state.get("message") or ""
    focus = org_transfer_followup_focus(message) or "summary"
    # HR asking about a generic transferred employee should audit org-wide mismatches,
    # not inspect the signed-in HR admin's own (usually consistent) record.
    generic_transfer = bool(
        user.role == "hr_admin"
        and not targets
        and (
            routing_discrepancy_question(message)
            or org_transfer_followup_focus(message) is not None
            or re.search(
                r"\b(an employee|employee changed|department(?:s)? (?:and|&) (?:their )?reporting|"
                r"previous manager|stale (?:approval|routing)|approval still|"
                r"routing (?:mismatch|discrepan|inconsist)|investigate)\b",
                message,
                re.IGNORECASE,
            )
        )
    )
    if generic_transfer:
        analysis = investigate_routing_discrepancies(ctx.db)
        tool = "investigate_routing_discrepancies"
        detail = f"{analysis.get('case_count', 0)} discrepancies"
    elif targets:
        subject = ctx.db.get(type(employee), targets[0]) or employee
        analysis = investigate_routing(ctx.db, subject)
        tool = "investigate_approval_routing"
        detail = "consistent" if analysis.get("consistent") else "mismatch"
    else:
        analysis = investigate_routing(ctx.db, employee)
        tool = "investigate_approval_routing"
        detail = "consistent" if analysis.get("consistent") else "mismatch"
    analysis = {**analysis, "focus": focus}
    return {
        "analysis": analysis,
        "record_context": {
            "subject": "routing_audit" if analysis.get("kind") == "routing_audit" else "routing_investigation",
            "focus": focus,
            "user_id": user.id,
            "employee_id": employee.id,
        },
        "workflow": [_step("investigate_routing", "Employee context", "complete", detail)],
        "tool_trace": [_tool(tool, "ok", detail)],
    }


def route_after_leave(state: HRState) -> str:
    return "retrieve_policy"


def route_after_analytics(state: HRState) -> str:
    return "retrieve_policy"


def route_after_retrieval(state: HRState) -> str:
    focus = state.get("leave_focus") or (state.get("analysis") or {}).get("focus") or ""
    # Date recommendations are guidance only — skip the submission confirmation gate.
    if focus == "recommend_dates" or leave_date_recommendation_question(state.get("message") or ""):
        return "compose_response"
    if state.get("intent") == "leave_request":
        return "prepare_workflow"
    if state.get("intent") == "leave_planning" and focus in {
        "prepare",
        "draft_review",
        "confirmation_gate",
    }:
        return "prepare_workflow"
    if state.get("intent") == "org_transfer" and explicit_routing_correction_request(
        state.get("message") or ""
    ):
        return "prepare_workflow"
    if state.get("intent") == "workforce_analytics" and workforce_corrective_prepare_question(
        state.get("message") or ""
    ):
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
    graph.add_node("tool_orchestrate", tool_orchestrate, retry_policy=TRANSIENT)
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
            "tool_orchestrate": "tool_orchestrate",
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
        "tool_orchestrate",
        route_after_tools,
        {"prepare_workflow": "prepare_workflow", "compose_response": "compose_response"},
    )
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
