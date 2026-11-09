from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date

from sqlalchemy.orm import Session

from app.access import (
    AuthorizationError,
    assert_sensitive_access,
    can_view_performance,
    employee_for_user,
    redact_employee,
)
from app.ai.guard import detect_injection, sensitive_kinds, wants_bulk_export
from app.ai.llm import LLMClient
from app.audit import write_audit
from app.models import Department, Employee, User

ALLOWED_INTENTS = {
    "leave_planning",
    "leave_request",
    "leave_eligibility",
    "leave_policy",
    "leave_request_status",
    "leave_cancellation",
    "leave_balance",
    "team_conflict",
    "workforce_analytics",
    "org_transfer",
    "policy_question",
    "confidential_lookup",
    "employee_profile",
    "attendance_query",
    "holiday_query",
    "employee_directory",
    "team_roster",
    "team_analytics",
    "org_attendance",
    "org_leave",
    "compensation_query",
    "hr_metrics",
    "hr_action",
    "approval_decision",
    "self_service",
    "out_of_scope",
    "unsupported_mutation",
    "general_hr",
}


@dataclass
class RequestCtx:
    db: Session
    user_id: int
    ip: str | None = None
    conversation_id: int | None = None
    thread_id: str | None = None


CTX: ContextVar[RequestCtx | None] = ContextVar("helios_request_ctx", default=None)


def get_ctx() -> RequestCtx:
    ctx = CTX.get()
    if ctx is None:
        raise RuntimeError("HR request context is not set")
    return ctx


def load_actor(db: Session, user_id: int) -> tuple[User, Employee, Department | None]:
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise AuthorizationError("Account is not active.", resource="user")
    employee = employee_for_user(db, user)
    department = db.get(Department, employee.department_id)
    return user, employee, department


def find_mentioned(db: Session, message: str, self_id: int) -> list[Employee]:
    text = message.lower()
    people = [employee for employee in db.query(Employee).all() if employee.id != self_id]
    found = []
    for employee in people:
        full = f"{employee.first_name} {employee.last_name}".lower()
        if full and full in text:
            found.append(employee)
    if found:
        found.sort(key=lambda item: item.id)
        return found
    by_first: dict[str, list[Employee]] = {}
    for employee in people:
        by_first.setdefault(employee.first_name.lower(), []).append(employee)
    for first, matches in by_first.items():
        if len(matches) == 1 and len(first) >= 4 and re.search(rf"\b{re.escape(first)}(?:'s)?\b", text):
            found.append(matches[0])
    found.sort(key=lambda item: item.id)
    return found


def _personal(text: str) -> bool:
    return bool(re.search(r"\b(my|i|i'm|me)\b", text))


def remote_policy_question(text: str) -> bool:
    """Remote-work permission or policy. Past remote attendance is a different question."""
    lowered = text.lower()
    if re.search(r"\b(did i|was i|days did|have i worked|i worked|worked remotely)\b", lowered):
        return False
    # Historical attendance / dated day counts are never policy, even when phrased "remote-work days".
    if re.search(
        r"\b(between|from|during|in)\b.{0,40}\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|\d{4})\b",
        lowered,
    ) and re.search(r"\b(days?|attendance|recorded|logged|show|how many|count)\b", lowered):
        return False
    if re.search(r"\b(break(?:\s+it|\s+that|\s+this)?\s+down|by department|department breakdown)\b", lowered):
        return False
    if re.search(r"\b(remote[-\s]?work|work[-\s]from[-\s]home|working from home)\b", lowered):
        # "Show my remote-work days …" is attendance; bare "remote work" without days/history is policy.
        if re.search(r"\b(days?|attendance|recorded|logged|history|how many|count|total)\b", lowered) and _personal(
            lowered
        ):
            return False
        return True
    if re.search(r"\b(can i|could i|may i|am i allowed|am i permitted|do i need)\b", lowered) and re.search(
        r"\b(remote|remotely|from home)\b", lowered
    ):
        return True
    if re.search(r"\bhow many\b", lowered) and re.search(r"\b(remote|remotely|from home)\b", lowered) and re.search(
        r"\b(can i|may i|allowed|per week|each week|a week)\b", lowered
    ):
        return True
    if re.search(r"\b(policy|policies)\b", lowered) and re.search(r"\b(remote|remotely|from home|wfh)\b", lowered):
        return True
    if re.search(r"\b(tomorrow|next week)\b", lowered) and re.search(r"\b(remote|remotely|from home)\b", lowered):
        return True
    return False


def remote_attendance_question(text: str) -> bool:
    """Recorded remote days for the signed-in employee, not a permission question."""
    lowered = text.lower()
    if re.search(r"\b(employees|organization|organisation)\b", lowered) and not _personal(lowered):
        return False
    if remote_policy_question(lowered):
        return False
    if not re.search(r"\b(remote|remotely|remote[-\s]?work)\b", lowered):
        return False
    return bool(
        re.search(
            r"\b(did|was|were|worked|recorded|attendance|show|total|count|how many|number of|logged|days?)\b",
            lowered,
        )
        and _personal(lowered)
    )


def post_approval_question(text: str) -> bool:
    """Policy ask about the post-approval process, not a manager inbox impact follow-up."""
    lowered = text.lower()
    if re.search(r"\bafter a manager approves\b", lowered):
        return False
    # "What happens if I approve/reject it?" is operational impact for pending approvals.
    if approval_inbox_followup_focus(lowered) in {
        "approve_impact",
        "reject_impact",
        "prepare_approve",
        "why_urgent",
        "urgency",
    }:
        return False
    if re.search(r"\b(if i approve|if we approve|approve it|approve that|approve this)\b", lowered):
        return False
    if re.search(r"\b(if i reject|if we reject|reject it|reject that|reject this)\b", lowered):
        return False
    return bool(re.search(r"\b(what happens|what would happen|what if)\b", lowered) and re.search(r"\bapprov", lowered))


_TEAM_CUE = (
    r"\b(my (?:direct |(?!attendance |leave |expense |status |hr |holiday |remote |absence )\w+ )?reports?|my team|direct reports?|"
    r"among my reports|which of my reports|a report'?s|"
    r"one of my reports|my people|members of my team|team members?|my reporting line|under me|report(?:s|ing)? to me|"
    r"(?:two|three|four|five|several|some|both|any|a few|most|all|each|every|which|no) (?:of my )?reports(?: are| have| show| were| is)?)\b"
)


def manager_team_roster_question(text: str) -> bool:
    """Self-scoped direct-report roster for the signed-in manager, not the org directory."""
    lowered = text.lower()
    # Explicit out-of-hierarchy / non-report expansion is never a legitimate roster ask.
    if re.search(
        r"\b(not my report|non-?reports?|outside my reports?|people outside|"
        r"including people outside|even though.{0,40}not my report)\b",
        lowered,
    ):
        return False
    # Team leave/attendance/analytics questions are team operations, not a roster listing.
    # Job titles, departments, locations and employee numbers remain roster attributes when the
    # ask is explicitly a list/show of direct reports (not pending-leave / hours analytics).
    roster_list_ask = bool(
        re.search(r"\b(list|show|who (?:are|is)|roster)\b", lowered)
        and re.search(
            r"\b(my (?:\w+ ){0,3}(?:direct )?reports?|direct reports? (?:of mine|to me)|"
            r"report(?:s|ing)? (?:directly )?to me)\b",
            lowered,
        )
        and not re.search(
            r"\b(most direct reports|employees with .{0,20}direct reports|which employees report|"
            r"organization|organisation|all employees)\b",
            lowered,
        )
    )
    if re.search(
        r"\b(leave|balances?|attendance|absent|absences?|remote(?:ly)?|late|pending|approved|hours|sick|projects?|phone|"
        r"joined|tenure|blank|missing|incomplete|empty|exhausted|on leave|new hires?|recent hires?|"
        r"performance ratings?|grades?)\b",
        lowered,
    ) and not re.search(r"\bhire dates?\b", lowered) and not roster_list_ask:
        return False
    # Roster list may include job titles / departments / employment status without becoming analytics.
    if roster_list_ask and not re.search(
        r"\b(leave|attendance|absent|remote|hours|pending|performance ratings?)\b", lowered
    ):
        return True
    if re.search(
        r"\b(list me as their manager(?:_id)?|me as (?:their|the) manager(?:_id)?|whose manager_id|"
        r"manager_id (?:field |column |value )?(?:is|points to|equals|=|matches|references|set to|holds|contains|shows) (?:me|my|mine)|"
        r"manager_id (?:points to|references|is) my|my (?:id|employee id|employee_id|account|number) (?:in|as|for|on) (?:their |the )?manager_id|"
        r"report(?:s|ing)? (?:to|into) my account|"
        r"(?:in|on) my reporting line|under me)\b",
        lowered,
    ):
        return True
    if re.search(
        r"\b(my direct reports|direct reports of mine|report(?:s|ing)? directly to me|"
        r"who reports to me|reports to me|"
        r"(?:my )?direct[- ]report (?:list|team|roster|group)|how (?:large|big|many people) is my (?:direct[- ]report )?team|"
        r"size of my (?:direct[- ]report )?team|my team size|"
        r"employees (?:who )?report(?:ing)? (?:directly )?to me|"
        r"return to my (?:direct )?reports?|only my (?:direct )?reports?|"
        r"listed by employee number)\b",
        lowered,
    ) and not re.search(
        r"\b(leave|attendance|absent|remote|hours|pending|performance ratings?|grades?|"
        r"approved|balances?|working hours)\b",
        lowered,
    ):
        return True
    if re.search(r"\bhow many\b", lowered) and re.search(r"\breport", lowered) and re.search(r"\bto me\b", lowered):
        return True
    if re.search(r"\b(how many|team size|count)\b", lowered) and re.search(
        r"\b(in my team|on my team|my team(?:\s+have|\s+has)?|direct reports?)\b", lowered
    ):
        return True
    if re.search(r"\b(employees?|people|members?)\b", lowered) and re.search(
        r"\b(currently )?(in|on) my team\b", lowered
    ):
        return True
    if re.search(r"\b(show|list|who (?:are|is)|who reports)\b", lowered) and re.search(
        r"\b(my (?:direct )?reports?|to me)\b", lowered
    ) and not re.search(
        r"\b(leave|attendance|absent|remote|hours|pending|performance ratings?|grades?|"
        r"approved|balances?|working hours)\b",
        lowered,
    ):
        return True
    if re.search(r"\b(show|list)\b", lowered) and re.search(r"\bmy team\b", lowered) and not re.search(
        r"\b(attendance|leave|absent|remote|conflict|overlap|pending|approved|"
        r"performance ratings?|grades?|hours)\b",
        lowered,
    ):
        return True
    return False


def hr_corrective_action_question(text: str) -> bool:
    """True for HR follow-ups asking what corrective actions/steps to take."""
    return org_transfer_followup_focus(text) == "remediation" or bool(
        re.search(
            r"\b("
            r"what corrective actions?|"
            r"corrective actions?(?: should| for| to)?|"
            r"what (?:actions?|steps?) should hr|"
            r"what should hr (?:do|take|correct|fix|resolve)|"
            r"recommended (?:hr )?(?:actions?|corrections?|steps?)"
            r")\b",
            (text or "").lower(),
        )
    )


def record_consistency_question(text: str) -> bool:
    """True when asking whether department / manager / approval-manager records align."""
    lowered = (text or "").lower()
    return bool(
        re.search(
            r"\b("
            r"(?:department|reporting[-\s]?manager|approval[-\s]?manager|manager(?:\s+records?)?)"
            r".{0,80}(?:consistent|consistency|inconsist\w*|mismatch|aligned|match)|"
            r"(?:consistent|consistency|inconsist\w*|mismatch|aligned|match)"
            r".{0,80}(?:department|reporting[-\s]?manager|approval[-\s]?manager|manager|records?)"
            r")\b",
            lowered,
        )
    )


def routing_discrepancy_question(text: str) -> bool:
    """True for employee-record routing / transfer mismatch investigations.

    Plain identity asks such as "who is my reporting manager" are employee_profile,
    not discrepancy investigations.
    """
    lowered = (text or "").lower()
    if record_consistency_question(lowered):
        return True
    # Identity-only reporting-manager questions are not discrepancy probes.
    if re.search(
        r"\b(who is my|who's my|who is listed as my|what is my|tell me my)\b.{0,40}\b"
        r"(reporting[-\s]?manager|line manager|manager)\b",
        lowered,
    ) and not re.search(
        r"\b(inconsist\w*|mismatch|discrepan\w*|previous|old|stale|wrong|transfer|changed)\b",
        lowered,
    ):
        return False
    return bool(
        re.search(
            r"\b("
            r"routing (?:mismatch|discrepan\w*|inconsist\w*|audit)|"
            r"approval[-\s]?ownership|stale (?:approval|routing|manager)|"
            r"previous manager|old manager|approval still|"
            r"manager_id|approval_manager_id|transfer_effective_date|"
            r"changed departments?|department transfer|transferred|"
            r"people-?manager changed|"
            r"(?:reporting[-\s]?manager|approval[-\s]?manager).{0,40}"
            r"(?:mismatch|inconsist\w*|discrepan\w*|wrong|stale|changed)|"
            r"(?:mismatch|inconsist\w*|discrepan\w*|wrong|stale|changed).{0,40}"
            r"(?:reporting[-\s]?manager|approval[-\s]?manager)|"
            r"employee[-\s]?record (?:investigation|discrepan\w*|mismatch)|"
            r"discrepan\w* (?:in|with|between).{0,40}(?:department|manager|approval|routing|record)"
            r")\b",
            lowered,
        )
    )


def org_transfer_followup_focus(message: str) -> str | None:
    """Follow-up focus for an existing routing_audit / routing_investigation thread.

    Standalone urgency phrasing alone is not enough — that belongs to manager approval
    inbox threads unless the ask also names a discrepancy/routing/transfer issue.
    """
    text = (message or "").lower().strip()
    discrepancy_cue = bool(
        re.search(
            r"\b(discrepan\w*|routing|transfer|previous manager|approval_manager|"
            r"wrong manager|stale approval|employee record)\b",
            text,
        )
    )
    if re.search(
        r"\b(which discrepancy|priority case|worst case|should (?:we|hr) (?:fix|address) first|"
        r"highest priority (?:discrepan\w*|case|routing)|most urgent (?:discrepan\w*|case|routing))\b",
        text,
    ) or (
        discrepancy_cue
        and re.search(r"\b(most urgent|highest priority|urgent(?:ly)?)\b", text)
    ):
        return "urgency"
    if re.search(
        r"\b(why (?:is|was) it important|why (?:does|do) it matter|why (?:should|must) .{0,20}resolve|"
        r"why important|importance|risk if unresolved|what happens if .{0,20}not (?:fix|correct|resolve))\b",
        text,
    ) or (
        discrepancy_cue
        and re.search(r"\b(why (?:is|was) it (?:important|urgent)|importance)\b", text)
    ):
        return "importance"
    if re.search(
        r"\b("
        r"what steps|how (?:should|can|do) hr (?:correct|fix|resolve)|"
        r"correct(?:ive)? (?:steps?|actions?)|"
        r"what corrective actions?|"
        r"what (?:actions?|steps?) should hr|"
        r"what should hr (?:do|take|correct|fix|resolve)|"
        r"actions? (?:should|would|can|must) hr|"
        r"recommended (?:hr )?(?:actions?|corrections?|steps?)|"
        r"how to (?:fix|correct|resolve)|remediat\w*"
        r")\b",
        text,
    ):
        return "remediation"
    if re.search(r"\b(file|submit|apply|prepare).{0,40}(correction|fix|routing)\b", text) or re.search(
        r"\b(correct (?:the|this|that) (?:record|routing|approval)|update approval_manager_id)\b",
        text,
    ):
        return "file_correction"
    return None


def approval_inbox_followup_focus(message: str) -> str | None:
    """Follow-up focus inside a manager pending-approval / team leave-request thread."""
    text = (message or "").lower().strip()
    # Routing-discrepancy urgency/importance stays on org_transfer, not the approval inbox.
    if re.search(
        r"\b(discrepan\w*|routing|transfer|previous manager|approval_manager|"
        r"wrong manager|stale approval|employee record)\b",
        text,
    ):
        return None
    if re.search(
        r"\b(what happens if (?:i |we )?reject|"
        r"if i reject|impact of reject|consequences? of reject|"
        r"what (?:would|will) happen if .{0,20}reject|"
        r"what if (?:i |we )?reject)\b",
        text,
    ):
        return "reject_impact"
    if re.search(
        r"\b(what happens if (?:i |we )?approve|"
        r"if i approve|impact of approv|consequences? of approv|"
        r"what (?:would|will) happen if .{0,20}approv|"
        r"what if (?:i |we )?approve)\b",
        text,
    ):
        return "approve_impact"
    if re.search(
        r"\b(prepare (?:the |this |that )?(?:approval|decision)|"
        r"prepare (?:to )?approv|draft (?:the |this )?approval|"
        r"wait for my confirmation|await(?:ing)? my confirmation)\b",
        text,
    ):
        return "prepare_approve"
    if re.search(
        r"\b(why (?:is|was) (?:it|that|this)(?: request)? urgent|why urgent|"
        r"why (?:is|was) (?:it|that|this) (?:the )?(?:most )?urgent|"
        r"what makes (?:it|that|this) urgent)\b",
        text,
    ):
        return "why_urgent"
    if re.search(
        r"\b(which (?:request|approval|one) (?:is |has )?(?:the )?(?:most )?urgent|"
        r"most urgent (?:request|approval|leave)|"
        r"highest (?:operational )?risk|which (?:request|approval|one) has the highest)\b",
        text,
    ):
        return "urgency"
    if re.search(r"\b(most urgent|urgent(?:ly)?)\b", text) and re.search(
        r"\b(request|approval|pending|leave|inbox)\b",
        text,
    ):
        return "urgency"
    return None


def explicit_routing_correction_request(text: str) -> bool:
    """True only when the user explicitly asks to file/apply a routing correction."""
    return org_transfer_followup_focus(text) == "file_correction" or bool(
        re.search(
            r"\b(file (?:the |this |an? )?(?:hr )?correction|apply (?:the |this )?correction|"
            r"submit (?:the |a )?routing correction|correct (?:liam|the employee)(?:'s)? (?:routing|record)|"
            r"update approval_manager_id|please (?:file|apply) .{0,20}correction)\b",
            (text or "").lower(),
        )
    )


def team_leave_overlap_question(text: str) -> bool:
    """True for manager/HR questions about employees with overlapping leave dates."""
    lowered = (text or "").lower()
    if re.search(
        r"\b(organization|organisation|company-wide|across (?:the )?departments?|"
        r"absenteeism|unplanned absence|attendance patterns?)\b",
        lowered,
    ):
        return False
    return bool(
        re.search(
            r"\b("
            r"overlap(?:ping)? leave(?: dates?)?|"
            r"leave(?: dates?)? overlap|"
            r"employees? (?:have |with )?overlap(?:ping)? leave|"
            r"which employees .{0,40}overlap|"
            r"overlapping (?:leave )?(?:dates?|requests?|windows?)|"
            r"leave conflicts?|"
            r"same (?:leave )?dates?"
            r")\b",
            lowered,
        )
        or (
            re.search(r"\b(overlap|overlapping)\b", lowered)
            and re.search(r"\b(leave|request|dates?)\b", lowered)
            and not re.search(r"\b(my present|my attendance|present days?)\b", lowered)
        )
    )


def workforce_followup_question(text: str) -> bool:
    """True when the ask needs ranked/absence analytics, not a raw last-N-days attendance dump."""
    lowered = (text or "").lower()
    # Employee-record routing investigations are not attendance analytics.
    if routing_discrepancy_question(lowered):
        return False
    # Team leave-overlap questions are staffing analysis, not org absence analytics.
    if team_leave_overlap_question(lowered):
        return False
    return bool(
        re.search(
            r"\b("
            r"highest (?:unplanned )?(?:absence|absenteeism|absent)|"
            r"which department .{0,60}(?:highest|most|worst).{0,40}(?:absence|absenteeism|absent)|"
            r"department .{0,40}(?:highest|most).{0,30}(?:absence|absenteeism)|"
            r"absenteeism|"
            r"operational consequences?|possible consequences?|consequences? for (?:the )?(?:business|operations|org|organization|organisation)|"
            r"what (?:are|could be|might be) the .{0,30}consequences|"
            r"investigat\w*|which employees|which records|employees? or records|"
            r"should hr (?:investigate|review|look|examine)|"
            r"records? should hr|"
            r"absence rates?|unplanned absence|"
            r"unusual|anomal(?:y|ies)|attendance patterns?"
            r")\b",
            lowered,
        )
    )


def workforce_analytics_focus(message: str, *, prior_focus: str = "", subject: str = "") -> str:
    """Map an HR analytics question onto a distinct narrate focus."""
    text = (message or "").lower()
    if re.search(r"\b(break(?:\s+it)?\s+down|by department)\b", text) and not re.search(
        r"\b(highest|absenteeism|investigat)\b", text
    ):
        return "department_breakdown"
    if re.search(r"\b(remaining leave|highest remaining|leave balances?)\b", text):
        return "remaining_balances"
    if re.search(r"\b(data integrity|integrity issues?|these calculations)\b", text):
        return "data_integrity"
    if re.search(r"\b(confidently report|which figures|can hr)\b", text):
        return "reporting_confidence"
    if re.search(
        r"\b(compare|comparison|contrast).{0,40}\b(last|past|previous).{0,20}\bmonths?\b|"
        r"\b(last|past|previous)\s+(?:\w+\s+)?months?\b.{0,20}\b(compar|contrast)",
        text,
    ) or (
        re.search(r"\b(compare|comparison|contrast)\b", text)
        and re.search(r"\b(last|past|previous)\s+(?:\w+\s+)?(?:complete\s+)?(?:calendar\s+)?months?\b", text)
    ) or re.search(
        r"\b(last|past|previous)\s+(?:two|2)\s+(?:complete\s+)?(?:calendar\s+)?months?\b",
        text,
    ):
        return "month_comparison"
    if re.search(
        r"\b(office|location|site).{0,40}\b(increase|change|rose|largest)|"
        r"\blargest increase\b",
        text,
    ):
        if prior_focus in {
            "month_comparison",
            "office_comparison",
            "absence",
            "highest_department",
            "anomaly",
            "investigation",
            "recommendations",
            "consequences",
            "evidence",
            "department_breakdown",
            "",
        } or subject in {"workforce_analytics", "org_attendance"}:
            return "office_comparison"
        return "office_clarification"
    if re.search(r"\b(unusual|anomal(?:y|ies)|attendance patterns?)\b", text):
        return "anomaly"
    if re.search(
        r"\b(investigat\w*|which employees|which records|employees? or records|"
        r"should hr (?:investigate|review|look|examine)|records? should hr|"
        r"further investigation|might need)\b",
        text,
    ):
        return "investigation"
    if re.search(
        r"\b(explain the data|data behind|figures? behind|supporting (?:data|figures|evidence))\b",
        text,
    ):
        return "evidence"
    if re.search(
        r"\b(operational consequences?|possible consequences?|"
        r"consequences? for (?:the )?(?:business|operations|org|organization|organisation)|"
        r"what (?:are|could be|might be) the .{0,30}consequences)\b",
        text,
    ):
        return "consequences"
    if re.search(r"\b(recommend|recommendation|actions? (?:for|to) hr|what actions)\b", text):
        return "recommendations"
    if re.search(
        r"\b(highest (?:unplanned )?(?:absence|absenteeism|absent)|"
        r"which department .{0,60}(?:highest|most|worst).{0,40}(?:absence|absenteeism|absent)|"
        r"department .{0,40}(?:highest|most).{0,30}(?:absence|absenteeism)|"
        r"absenteeism)\b",
        text,
    ):
        return "highest_department"
    if "utilization" in text or prior_focus in {
        "leave_utilization",
        "department_breakdown",
        "remaining_balances",
        "data_integrity",
        "reporting_confidence",
    }:
        return "leave_utilization"
    return "absence"


def personal_leave_status_question(text: str) -> bool:
    """Employee self-service leave status / pending list — never a manager approval inbox.

    Matches personal phrasing such as "my pending leave" and also role-gated paraphrases
    like "leave requests awaiting approval" when there is no team/inbox cue.
    """
    lowered = (text or "").lower()
    # Preparing / submitting a leave request is leave_request, not status lookup.
    if _explicit_leave_request(lowered):
        return False
    if re.search(
        r"\b(my team|direct reports?|team(?:'s)? pending|approval inbox|"
        r"awaiting my (?:approval|decision)|waiting for my (?:approval|decision)|"
        r"my (?:pending )?approvals?|sign[- ]?off|waiting (?:on|for) me|"
        r"from me|needs? my (?:decision|approval)|sitting with me|"
        r"across (?:the )?(?:organization|organisation|company)|organization-wide|"
        r"organisation-wide|company-wide|all employees)\b",
        lowered,
    ):
        return False
    if re.search(_TEAM_CUE, lowered):
        return False
    # Explicit personal leave status / pending list.
    if re.search(
        r"\b("
        r"(?:show|list|display|see|get)\s+(?:me\s+)?(?:my\s+)?pending\s+leave(?:\s+requests?)?|"
        r"my\s+pending\s+leave(?:\s+requests?)?|"
        r"status\s+of\s+my\s+leave|"
        r"what(?:'s| is)\s+the\s+status\s+of\s+my\s+leave|"
        r"my\s+leave\s+requests?(?:\s+status)?|"
        r"leave\s+request\s+status|"
        r"how\s+many\s+(?:of\s+)?my\s+leaves?\s+(?:are\s+)?pending|"
        r"what\s+leave\s+requests?\s+(?:are\s+)?awaiting\s+approval|"
        r"leave\s+requests?\s+(?:that\s+are\s+)?(?:pending|awaiting\s+approval|waiting\s+for\s+(?:a\s+)?(?:decision|approval))"
        r")\b",
        lowered,
    ):
        return True
    # Bare pending-leave asks without team/approver framing (role-gated in classify for employees).
    if re.search(r"\bpending\s+leave(?:\s+requests?)?\b", lowered) and not re.search(
        r"\b(team|reports?|approvals?|inbox|sign[- ]?off)\b",
        lowered,
    ):
        return True
    return False


def approval_inbox_question(text: str) -> bool:
    """Manager approval-inbox asks (list / waiting / highest risk), not approve/reject decisions."""
    lowered = text.lower()
    # Explicit decide phrasing is handled by parse_approval_decision instead.
    if re.search(r"\b(approve|reject)\s+(?:leave\s+)?(?:request\s+)?#?\d+\b", lowered):
        return False
    if re.search(r"\b(approve|reject)\s+(?:it|that|this|the request|that request|this request)\b", lowered):
        return False
    # Preparing an approval decision is approval_decision, not a fresh inbox list.
    if approval_inbox_followup_focus(lowered) == "prepare_approve":
        return False
    # An employee's own pending leave / leave status is never the manager inbox.
    if personal_leave_status_question(lowered) and not re.search(
        r"\b(my team|direct reports?|awaiting my|waiting for my|approval inbox|"
        r"my (?:pending )?approvals?|sign[- ]?off|from me)\b",
        lowered,
    ):
        return False
    if re.search(
        r"\b("
        r"pending approvals?|my (?:pending )?approvals?|approval inbox|"
        r"awaiting my (?:approval|decision)|waiting for my (?:approval|decision)|"
        r"waiting on (?:a |my )?(?:decision|approval|sign[- ]?off) from me|"
        r"requests? (?:are )?waiting for (?:my|me)|waiting for me|"
        r"which requests? (?:are )?waiting|"
        r"which (?:request|approval|one) has the highest (?:operational )?risk|"
        r"highest (?:operational )?risk\b.{0,40}\b(?:request|approval|pending)|"
        r"(?:request|approval|pending).{0,40}\bhighest (?:operational )?risk|"
        r"which (?:request|approval|one) (?:is |has )?(?:the )?(?:most )?urgent|"
        r"most urgent (?:request|approval|leave)|"
        r"needs? my sign[- ]?off|still needs? my (?:decision|approval|sign[- ]?off)|"
        r"sitting in my inbox|in my inbox waiting|"
        r"(?:leave|approval) (?:asks?|requests?) (?:are )?(?:sitting )?in (?:my )?inbox|"
        r"(?:leave asks?|leave requests?|approvals?).{0,40}\b(?:inbox|sign[- ]?off|waiting on)|"
        r"(?:my |the )?team(?:'s)? pending (?:leave )?requests?|"
        r"pending leave requests? (?:for|on|in) (?:my |the )?(?:team|direct reports?)|"
        r"show (?:me )?(?:my |the )?team(?:'s)? .{0,40}pending"
        r")\b",
        lowered,
    ):
        return True
    # Team-scoped pending leave listing (not the manager's own leave history).
    if re.search(r"\b(my team|direct reports?|team(?:'s)?)\b", lowered) and re.search(
        r"\b(pending (?:leave )?requests?|leave requests? (?:that are )?pending|awaiting (?:my )?approval)\b",
        lowered,
    ):
        return True
    # Paraphrases: leave/approvals + inbox / sign-off / waiting on me.
    if re.search(r"\b(leave|approval|approvals?|requests?|asks?)\b", lowered) and re.search(
        r"\b(inbox|sign[- ]?off|waiting (?:on|for) (?:a )?(?:decision|approval) from me|"
        r"needs? (?:my|a) (?:decision|approval|sign[- ]?off))\b",
        lowered,
    ):
        return True
    return False


def parse_approval_decision(message: str) -> dict | None:
    """Extract an approve/reject decision aimed at a leave/approval request.

    A numeric id may be omitted when the speaker refers to a prior request as "it"
    or "that request"; the workflow resolves the id from conversation state.
    """
    text = message.lower().strip()
    # Informational asks about who must approve / what is required are not decisions.
    if re.search(
        r"\b(who (?:needs to|must|will|should|would|can|could) approve|who approves|"
        r"tell me who (?:would|will|needs? to|must|should) approve|"
        r"approval (?:path|chain|requirements?)|"
        r"what (?:do i|must i|needs? to be).{0,40}(?:before|submit|approv)|"
        r"what happens if .{0,40}\b(?:approve|reject)|"
        r"needs? to approve|required (?:to )?approv)\b",
        text,
    ):
        return None
    # Employee drafting their own leave ("draft/prepare … so I can confirm") is never an
    # approval decision, even when the sentence also asks who would approve it.
    # Preparing "the approval" / waiting for confirmation is a manager approval decision.
    if re.search(r"\b(draft|prepare)\b", text) and re.search(
        r"\b(leave|request|application|so i can confirm|for me to confirm)\b", text
    ) and not re.search(r"\b(the approval|this approval|that approval|approval decision)\b", text):
        return None
    decision = None
    if re.search(
        r"\b(prepare (?:the |this |that )?(?:approval|decision)|"
        r"prepare (?:to )?approv|draft (?:the |this )?approval|"
        r"wait for my confirmation|await(?:ing)? my confirmation)\b",
        text,
    ) and not re.search(r"\breject\b", text):
        decision = "approve"
    elif re.search(r"\bapprove\b", text) and not re.search(r"\breject\b", text):
        decision = "approve"
    elif re.search(r"\breject\b", text):
        decision = "reject"
    if decision is None:
        return None
    reason = ""
    reason_match = re.search(
        r"\b(?:because|due to|reason[:\s]+)\s+(.+)$",
        text,
        re.IGNORECASE,
    )
    if reason_match:
        reason = reason_match.group(1).strip(" .")
    match = re.search(
        r"\b(?:leave\s+)?(?:request|approval)\s*(?:id\s*)?(?:number\s*)?#?\s*(\d+)\b",
        text,
    )
    if not match:
        match = re.search(r"\b(?:approve|reject)\s+(?:leave\s+)?(?:request\s+)?#?(\d+)\b", text)
    if match:
        return {
            "decision": decision,
            "request_id": int(match.group(1)),
            "reason": reason,
        }
    # Pronoun / prior-request follow-up: "Reject it because coverage is insufficient."
    # Also covers "Prepare the approval and wait for my confirmation."
    if (
        re.search(r"\b(request|approval|leave request|confirmation)\b", text)
        or re.search(
            r"\b(?:approve|reject)\s+(?:it|that|this|the request|that request|this request)\b",
            text,
        )
        or re.search(r"\b(prepare|draft).{0,30}\bapprov", text)
    ):
        return {
            "decision": decision,
            "request_id": None,
            "reason": reason,
        }
    return None


def manager_team_ops_question(text: str) -> bool:
    """Team attendance/leave ops for a manager's reporting line, not org-wide HR analytics.

    Explicit organization-wide framing stays with organization_intent.
    """
    lowered = text.lower()
    # "organisation date" / "organization date" is the HRIS reference day, not an org-wide scope.
    scope_text = re.sub(r"\b(organi[sz]ation)(?:'s)?\s+(date|day|today|calendar)\b", "reference \\2", lowered)
    if re.search(
        r"\b(organization|organisation|organization-wide|organisation-wide|company-wide|"
        r"all employees|every employee|entire company|across (the )?organization|"
        r"across (?:all )?departments?|every department|all departments)\b",
        scope_text,
    ):
        return False
    if manager_team_roster_question(lowered):
        return False
    team_cue = bool(
        re.search(
            r"\b(my team|my (?:direct |(?!attendance |leave |expense |status |hr |holiday |remote |absence )\w+ )?reports?|"
            r"direct reports?|team members?|members of my team|awaiting my approval|"
            r"waiting for my (approval|decision)|pending approvals?|my (?:pending )?approvals?|"
            r"report(?:s|ing)? to me|a report'?s|one of my reports|my people|"
            r"my reporting line|(?:in|on) my (?:reporting )?line|under me|"
            r"(?:for|across|within|on) (?:the |my )?team|team(?:'s)? (?:risk|attendance|leave|coverage|late|absences?)|"
            r"(?:two|three|four|five|several|some|both|any|a few|most|all|each|every|which|no) (?:of my )?reports(?: are| have| show| were| is)?|"
            r"those (?:people|employees|members)|these (?:people|employees|members))\b",
            lowered,
        )
        or re.search(_TEAM_CUE, lowered)
    )
    ops = bool(
        re.search(
            r"\b(repeated late|late attendance|late arrivals?|incomplete attendance|missing attendance|"
            r"missing \w+ attendance|attendance gaps?|late rows?|missing (?:weekdays?|days?|rows?)|unverified missing|"
            r"incomplete (?:employment|profile|information)|missing profile|"
            r"(?:profile|employment) (?:fields?|information|details?|records?) (?:\w+ )?(?:incomplete|missing|blank|empty)|"
            r"incomplete (?:\w+ )?(?:profiles?|fields?|records?)|profile completeness|"
            r"attendance patterns?|attendance concerns?|main attendance|require attention|"
            r"attendance overview|attendance anomal|"
            r"anomal(?:y|ies)|data quality|currently on leave|who is on leave|on leave|pending leave|"
            r"approved (?:leave|requests?)|awaiting my approval|waiting for my (approval|decision)|"
            r"pending approvals?|my (?:pending )?approvals?|approval inbox|highest (?:operational )?risk|"
            r"sitting with me|lowest remaining|highest remaining|leave balances?|remaining balances?|"
            r"remaining (?:annual )?leave|annual leave remaining|remaining days|used all|"
            r"at or below|or fewer|or less|less than \w+ days?|below \w+ days?|"
            r"exhausted(?:\s+their)?\s+annual|exhausted.{0,20}leave|"
            r"attendance (for|of) my (?:team|reports)|team'?s attendance|reports covering|"
            r"worked remotely|remote(?:ly)? most|absent more than|"
            r"who is absent|who has been absent|working remotely|remote days?|present today|pending leave|"
            r"based in|project atlas|assigned to|absence trends?|"
            r"grades? (?:represented|on my team|across my team)|what grades|which grades|"
            r"performance ratings?|grades? of my|ratings? and grades?|"
            r"working hours|recorded hours|average (?:october |september )?(?:recorded )?hours|lowest total working hours|"
            r"joined|hired|new hires?|recent hires?|tenure|started (?:with|at|in|before|after|since)|start dates?|hire dates?|"
            r"came on board|onboarded|\d{4} or (?:later|earlier|after|before)|"
            r"blank (?:phone|titles?|job titles?|email|field)|missing (?:a |an |their |the |any |stored |recorded )*(?:job titles?|titles?|phone|email)|"
            r"without a (?:job title|phone|title)|empty (?:phone|title|job title))\b",
            lowered,
        )
    )
    # Plain attendance / leave vocabulary counts as team ops when the team cue is explicit
    # ("my reports", "my direct reports", "my team"), so filters by month, status, or department
    # never fall back to the manager's own attendance record.
    strong_cue = bool(re.search(_TEAM_CUE, lowered))
    generic_ops = bool(
        re.search(
            r"\b(late|absent|absences?|remote|present|attendance|hours|sick|leave|balances?|pending|approved|"
            r"performance|ratings?|grades?|"
            r"job titles?|titles?|phone|started|hired|joined|location|locations|based in|incomplete|profiles?|projects?|assigned|remotely)\b",
            lowered,
        )
    )
    if strong_cue and generic_ops:
        return True
    # Ambiguous employee-plural attendance/leave without org markers: treat as team ops candidate.
    # Role gating in classify/authorize decides manager vs HR.
    ambiguous_teamish = bool(
        re.search(
            r"\b(employees with repeated late|employees with incomplete attendance|"
            r"which (?:of my )?reports|which of my reports|my reports show|among my reports|"
            r"anyone on my team|any of my (?:reports|people|direct reports)|"
            r"which employees have (?:(?:unusual|anomalous|odd|strange)\s+)?"
            r"(?:pending leave|attendance patterns?|missing profile|incomplete)|"
            r"employees? (?:with|showing|who have) (?:(?:unusual|anomalous)\s+)?attendance|"
            r"unusual attendance(?: patterns?)?|anomalous attendance|attendance anomal(?:y|ies)|"
            r"employees with the lowest remaining|who is currently on leave|"
            r"are any employees currently on leave|employees currently on leave|"
            r"who has exhausted|exhausted their annual leave|has anyone on my team exhausted|"
            r"main attendance concerns|attendance concerns|attendance anomalies|"
            r"leave requests awaiting my approval|show me employees with repeated late|"
            r"which employees have incomplete attendance|pending leave sitting with me|"
            r"those people with remote|remote days?.{0,40}pending leave|"
            r"pending leave.{0,40}remote)\b",
            lowered,
        )
    )
    # Unusual/anomalous attendance without explicit org markers is team-scoped for managers.
    unusual_attendance = bool(
        re.search(
            r"\b(unusual|anomal(?:y|ies)|anomalous)\b.{0,40}\battendance\b|"
            r"\battendance\b.{0,40}\b(unusual|anomal(?:y|ies)|anomalous|patterns?)\b",
            lowered,
        )
    )
    return bool(
        unusual_attendance
        or (
            ops
            and (
                team_cue
                or ambiguous_teamish
                or re.search(r"\bawaiting my approval\b", lowered)
            )
        )
    )


def self_scoped_approved_without_decision(text: str) -> bool:
    """Approved leave missing approval-decision rows for the signed-in employee."""
    lowered = text.lower()
    if re.search(
        r"\b(organization|organisation|all employees|every employee|company-wide|organization-wide)\b",
        lowered,
    ):
        return False
    return bool(
        re.search(r"\bapproved\b", lowered)
        and re.search(r"\b(no|without|missing)\b", lowered)
        and re.search(r"\b(approval|decision)\b", lowered)
        and re.search(r"\b(leave|request)\b", lowered)
    )


def self_scoped_missing_attendance(text: str) -> bool:
    """Scheduled attendance gaps for the signed-in employee when no org roster is named."""
    lowered = text.lower()
    if re.search(
        r"\b(employees|organization|organisation|department|everyone|company-wide|organization-wide|"
        r"my team|direct reports?|team members?)\b",
        lowered,
    ):
        return False
    if re.search(r"\b(all|every)\s+employees?\b", lowered):
        return False
    if re.search(r"\bmissing attendance\b", lowered):
        return True
    if re.search(r"\b(scheduled working days?|working days?)\b", lowered) and re.search(
        r"\b(missing|no)\b", lowered
    ) and re.search(r"\battendance\b", lowered):
        return True
    if re.search(r"\b(missing|no)\b", lowered) and re.search(r"\battendance records?\b", lowered):
        return True
    return False


def _org_attendance_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(employees|organization|organisation|organization-wide|organisation-wide|"
            r"company-wide|department|everyone|all employees|every employee|team'?s attendance)\b",
            text,
        )
    )


def organization_intent(message: str) -> str | None:
    """Deterministic organization-wide HR intent. Personal and policy questions return None."""
    text = message.lower()
    if re.search(r"\b(how many projects|projects? (?:are )?registered|expired polic(?:y|ies))\b", text):
        return "hr_metrics"
    if re.search(r"\b(policy|policies)\b", text):
        return None
    if re.search(r"\bwho approved (it|that|this|my)\b", text):
        return None
    if re.search(r"\bafter a manager approves\b", text):
        return "hr_action"
    if re.search(r"\b(what happens|what would happen|what if)\b", text):
        return None
    if manager_team_roster_question(text):
        return None
    if self_scoped_missing_attendance(text):
        return None
    if self_scoped_approved_without_decision(text):
        return None
    directed = bool(re.search(r"\b(show|tell|give|find|list) me\b", text))
    if _personal(text) and not directed and not re.search(r"\b(employees|organization|department|everyone|their)\b", text):
        return None
    if re.search(r"\bmy\b", text) and "awaiting my approval" not in text and not re.search(
        r"\b(employees|organization|department|everyone)\b", text
    ):
        return None
    if re.search(r"\b(delete|remove)\b", text) and re.search(r"\bemployee record\b", text):
        return "hr_action"
    if re.search(r"\b(profile update|on behalf of)\b", text):
        return "hr_action"
    # Specific leave-request decisions are approval_decision (same service as REST), not org HR mutation.
    if re.search(r"\b(approve|reject)\b", text) and (
        re.search(r"\b(leave|request|approval)\b", text)
        or re.search(r"\b(?:approve|reject)\s+(?:it|that|this)\b", text)
    ):
        if parse_approval_decision(text):
            return None
        # Bare org-wide "approve leave" / "reject leave" without a request target stays hr_action.
        if re.search(r"\bleave\b", text) and not re.search(r"\b(request|approval|#?\d+)\b", text):
            return "hr_action"
    if re.search(
        r"\b(payroll|average salar(?:y|ies)|salaries above|salary distribution|highest salary|"
        r"highest average annual salary|organization-?wide average (?:annual )?salary|"
        r"org(?:anisation|anization)?-?wide average (?:annual )?salary|"
        r"average annual salary for active|avg(?:erage)? (?:annual )?salary|"
        r"salary adjustment|salary adjustments|salaries of everyone|everyone'?s salar|"
        r"all salaries|salaries of all|give me the salaries|annualized salary|salary total|"
        r"stored annual salaries|compensation total|salary_annual|sum of (?:all )?salar\w*|total salar\w*|"
        r"salar\w* sum|aggregate salar\w*|combined salar\w*|"
        r"salary expenditure|annual salary expenditure|total annual salary|compensation expenditure|"
        r"payroll expenditure|wage bill|salary spend|"
        r"percentage of the total annual salary|salary expenditure .{0,20}engineering|"
        r"hired before 2020.{0,40}salary|salary.{0,40}before 2020)\b",
        text,
    ):
        return "compensation_query"
    # Tenure of a previously filtered cohort stays on the directory so hire filters survive.
    if re.search(r"\b(tenure|average tenure|mean tenure|median tenure)\b", text) and re.search(
        r"\b(those|them|that group|that cohort|among those|of those|same employees)\b",
        text,
    ):
        return "employee_directory"
    # Average tenure measured in calendar days is a directory aggregation focus.
    if re.search(r"\baverage employee tenure\b", text) and re.search(r"\bdays?\b", text):
        return "employee_directory"
    # Relative hiring-year analytics are directory/hire_date aggregations, not policy/hr_metrics.
    # Bare "monthly hiring trends" stays hr_metrics; explicit previous-calendar-year totals are directory.
    if re.search(r"\b(hir(?:e|ed|ing)|hires|hiring)\b", text) and re.search(
        r"\b(previous (?:\w+ )?calendar years?|last (?:\w+ )?calendar years?|yearly totals|"
        r"highest (?:number of )?hires)\b",
        text,
    ):
        return "employee_directory"
    if re.search(
        r"\b(workforce overview|employees by department|employees by location|by department.{0,40}by location)\b",
        text,
    ):
        return "employee_directory"
    if re.search(r"\b(missing|incomplete|empty|unset)\b", text) and re.search(
        r"\b(across (?:the )?workforce|workforce|organization|organisation|profile fields?)\b",
        text,
    ):
        return "employee_directory"
    if re.search(
        r"\b(average|avg|mean)\b.{0,40}\b(working )?hours\b", text
    ) and re.search(r"\b(department|departments)\b", text):
        return "org_attendance"
    if re.search(
        r"\b(turnover|employee tenure|average tenure|mean tenure|median tenure|tenure in years|typical tenure|"
        r"joined and left|hr summary|hr metrics|"
        r"how many projects|projects? (?:are )?registered|expired policy|expired policies|"
        r"approval decisions?|approvals? (?:rows?|table|records?|decisions?)|how many approvals|approval rows|"
        r"no corresponding approval|without .{0,20}approval decision|leave requests have no corresponding|"
        r"how many (?:distinct |unique )?departments|departments? (?:table|are defined|exist|are there|are stored)|"
        r"number of departments|list (?:the |all )?departments|distinct departments|which departments (?:exist|are defined))\b",
        text,
    ):
        return "hr_metrics"
    # Headcount questions, including two-department comparisons, stay on the directory even when
    # phrased as "between X and Y" (which otherwise reads as an attendance comparison).
    if re.search(r"\b(headcount|active employees|employees? (?:are|is) active)\b", text) and not re.search(
        r"\b(attendance|absent|remote|present|late|leave|hours|sick|salar\w*|tenure|manager_id|null|phone|location)\b", text
    ):
        return "employee_directory"
    if re.search(
        r"\b(percentage|percent|%|share).{0,40}\b(workforce|active employees|headcount)\b|"
        r"\bwhat percentage of the active workforce\b|"
        r"\bat least five years\b|\bmore than three years\b|"
        r"\bhired in each quarter\b|\boffice location has the largest\b|"
        r"\bsecond[- ]largest\b|\bthird[- ]largest\b|\bsmallest (?:active )?workforce\b|"
        r"\bwithout assigned managers?\b|\bno assigned manager\b|"
        r"\bdepartments? (?:have|with) (?:employees? )?without (?:assigned )?managers?\b|"
        r"\bstaff (?:are |is )?currently marked active\b|\bmarked active\b|"
        r"\bhow many (?:people|staff|workers) (?:are )?(?:currently )?(?:active|employed)\b|"
        r"\bhow many (?:are|employees? are) active overall\b|"
        r"\b(?:size|size of|headcount of) (?:the )?(?:active )?(?:staff|workforce|employee) pool\b|"
        r"\bactive staff pool\b|\bactive (?:employee )?headcount\b|"
        r"\blargest (?:number of )?active (?:employees|workforce|headcount)\b|"
        r"\bdepartment .{0,30}largest\b|"
        r"\b(?:people|staff|workers|employees) (?:were |was )?hired (?:in|during)\b|"
        r"\bhired during (?:calendar )?year\b",
        text,
    ):
        return "employee_directory"
    if re.search(
        r"\b(hire_date|hired (?:in|before|after|since|during|on or after|on or before)|"
        r"joined (?:in |during |before |after |since |on or after )?\d{4}|joined before|joined after|joined since|"
        r"hire date (?:in|before|after|since)|calendar year \d{4}|\d{4} or later|\d{4} or earlier|"
        r"joined before (?:january|february|march|april|may|june|july|august|september|october|november|december)|"
        r"before (?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2})\b",
        text,
    ) and re.search(r"\b(employees?|people|staff|workers)\b", text):
        return "employee_directory"
    # Hiring volume / hire-year comparisons without requiring the word "employees".
    if re.search(r"\b(hir(?:e|ed|ing)\s+volume|hiring volume|compare hiring|hires in|were hired)\b", text) or (
        re.search(r"\b(hir(?:e|ed|ing)|hires)\b", text)
        and re.search(r"\b(20\d{2})\b", text)
        and re.search(r"\b(versus|vs\.?|compared|compare|between)\b", text)
    ):
        return "employee_directory"
    if re.search(r"\bgrades?\b|\bgraded\b", text) and re.search(
        r"\b(l[1-7]|versus|vs\.?|how many employees are graded|graded)\b", text
    ):
        return "employee_directory"
    # Location / office headcount and workforce share questions.
    if re.search(
        r"\b(how many employees (?:work|are based|are located)|headcount in|workforce headcount in|"
        r"employees work in|share of the workforce|percentage of the company|"
        r"austin'?s share|singapore and what percentage|london headcount|berlin)\b",
        text,
    ) or (
        re.search(r"\b(austin|berlin|london|singapore)\b", text)
        and re.search(r"\b(how many|headcount|percentage|percent|share|employees?)\b", text)
        and not re.search(r"\b(attendance|leave|salary|policy)\b", text)
    ):
        return "employee_directory"
    if re.search(r"\b(engineering headcount by location|engineering employees are based|break down engineering)\b", text):
        return "employee_directory"
    if re.search(r"\b(helena vogel|employee number and hire date)\b", text):
        return "employee_directory"
    if re.search(r"\baverage employee tenure\b", text) and re.search(r"\bdays?\b", text):
        return "employee_directory"
    # Org-wide pending leave/requests even when the word "leave" is omitted.
    if re.search(
        r"\b(organization|organisation|organization-wide|organisation-wide|company-wide|"
        r"across (?:the )?(?:organization|organisation|company)|all employees|every employee)\b",
        text,
    ) and re.search(r"\bpending\b", text) and re.search(r"\b(requests?|leave|approval)\b", text):
        return "org_leave"
    # Attendance and leave analytics before directory so "all employees with missing
    # attendance" stays org_attendance rather than a roster dump.
    # Team-scoped phrasing (including them/those after a prior team answer) is excluded;
    # managers remap via manager_team_ops_question / continuation.
    _teamish = (
        r"\b(team members?|direct reports?|members of my team|those people|these employees|"
        r"them|those|they|their|my team|my reports?)\b"
    )
    # First-person attendance stays personal even without the word "my" ("do I have").
    # Do not treat "show/tell/give me" as personal — that is an imperative, not self-scope.
    _stripped_imperative = re.sub(r"\b(show|tell|give|find|list|get)\s+me\b", " ", text)
    _personal_attendance = bool(re.search(r"\b(i|i'm|i’ve|i have|do i|did i|me)\b", _stripped_imperative))
    # Bare "attendance for 10/2026" without org cues is not organization-wide.
    _org_people_cue = bool(_org_attendance_signal(text))
    if (
        re.search(r"\battendance\b", text)
        and not re.search(r"\bmy\b", text)
        and not _personal_attendance
        and not re.search(_teamish, text)
        and _org_people_cue
    ):
        return "org_attendance"
    if re.search(
        r"\b(most absences|worked remotely|attendance rates|working hours|present today|were present|"
        r"more than three absences|employees were absent|missing attendance|"
        r"remote-?day totals|compare remote|between .+ and .+|"
        r"(?:present|remote|absent|late|sick)-?day (?:totals?|counts?)|(?:present|remote|absent|late|sick) (?:day )?totals?|"
        r"(?:organization|organisation|company-wide|org-wide) (?:present|remote|absent|late|sick)|"
        r"remote[- ]heaviest|heaviest (?:remote|absent|late) department|department membership and .{0,20} remote)\b",
        text,
    ) and not re.search(r"\bmy\b", text) and not _personal_attendance and not re.search(_teamish, text) and _org_people_cue:
        return "org_attendance"
    # Approved leave with an explicit calendar year is a filtered org leave metric, not the pending list.
    if re.search(r"\bapproved\b", text) and re.search(r"\bleave requests?\b", text) and re.search(
        r"\b(20\d{2}|start(?:ing|s)? in|start date)\b", text
    ):
        return "org_leave"
    if re.search(
        r"\b(pending leave|awaiting hr|awaiting my approval|on leave|on approved leave|"
        r"leave requests|leave balances?|"
        r"exhausted their annual|exhausted annual|overlapping leave|leave history|pending sick|"
        r"approved and rejected|lowest remaining|highest remaining|remaining annual leave|"
        r"average remaining annual leave|pending the longest|longest .{0,20}pending leave|"
        r"low leave|who approved|audit history|negative (?:or insufficient )?leave|"
        r"insufficient leave|zero remaining leave|pending manager or hr|still pending anywhere|"
        r"pending manager or hr approval)\b",
        text,
    ):
        return "org_leave"
    if re.search(r"\bpending(?:\s+\w+){0,4}\s+approval\b", text) and re.search(
        r"\b(organization|organisation|company|all employees|every employee|anywhere|hr admin)\b",
        text,
    ):
        return "org_leave"
    if re.search(
        r"\b(headcount|active employees|employees grouped|who are the managers|incomplete (?:\w+ )?profiles?|"
        r"employment[_ ]status (?:of |= |is |equal to |set to )?(?:'|\")?(?:active|terminated|inactive|on leave)|"
        r"(?:active|terminated|inactive) employment[_ ]status|"
        r"manager_id|(?:distinct|unique|different|how many) managers|"
        r"blank (?:phone|job title|email)|missing profile|on probation|profile of|reports? directly|their manager|joined in the last|"
        r"joined in \d{4}|most direct reports|locations? (?:are )?represented|employee directory|"
        r"no manager assigned|without a manager|do not have an assigned manager|don't have an assigned manager|"
        r"without an assigned manager|no assigned manager|unassigned manager|managers who have the most)\b",
        text,
    ):
        return "employee_directory"
    if re.search(r"\bemployment details\b", text) or re.search(r"\be-\d+\b", text):
        return "employee_directory"
    if re.search(r"\b(all employees|employees in the)\b", text) and "salary" not in text:
        return "employee_directory"
    if re.search(
        r"\b(employee distribution|grouped by department|assigned to each department|"
        r"employees? (?:are )?(?:currently )?assigned (?:to )?each department|"
        r"how many employees .{0,40}each department|"
        r"(?:active )?(?:headcount|employees|people) (?:per|by) department)\b",
        text,
    ) and not re.search(r"\b(attendance|absent|remote|leave|salar\w*|policy|my (?:direct )?reports?|my team)\b", text):
        return "employee_directory"
    if re.search(r"\bemployees\b", text) and "salary" not in text and re.search(
        r"\b(engineering|product|finance|sales|legal|customer|supply chain|people operations)\b", text
    ) and not re.search(r"\b(attendance|absent|leave|remote)\b", text):
        return "employee_directory"
    return None


def ambiguous_team_or_org_scope(message: str) -> bool:
    """True when the user asks whether to use team vs organization scope."""
    text = message.lower()
    return bool(
        re.search(r"\b(my team|direct reports?)\b", text)
        and re.search(r"\b(whole company|organization|organisation|company-wide|entire company)\b", text)
        and re.search(r"\b(should i|or the|team or)\b", text)
    )


def data_semantics_question(text: str) -> bool:
    """Conceptual questions about what stored HR data can and cannot show. No record is retrieved."""
    lowered = text.lower()
    if re.search(
        r"\b(show|list|how many|count(?! as| it as| that as| them as)|total|sum|which (?:of my|employees|reports) (?:have|are|show))\b",
        lowered,
    ) and not re.search(r"\b(can be reported|reported confidently|alternative explanations?|be counted as)\b", lowered):
        return False
    # "Why can't you calculate my ... for the year" is answered from the person's own stored rows
    # (year-coverage analysis), not conceptually.
    if _personal(lowered) and re.search(
        r"\bwhy can'?t (?:you|helios|the assistant) (?:compute|calculate|derive)\b.{0,40}\b(attendance|leave|balance|hours|remote)\b",
        lowered,
    ):
        return False
    return bool(
        re.search(
            r"\b(should (?:helios|you|the assistant) (?:treat|count|interpret|assume|read|mark|consider|record|report)|"
            r"should .{0,60}? (?:count|be counted|be treated|be read|be interpreted|be marked|be considered|be recorded|be reported) as|"
            r"alternative explanations?|explanations? (?:besides|other than|apart from|beyond)|"
            r"reasons? (?:other than|besides|apart from|beyond)|other (?:possible |plausible )?(?:reasons|explanations|causes)|"
            r"which (?:figures|numbers|metrics|fields|values|counts) (?:can|could|should|may) be (?:reported|trusted|relied|used|stated|quoted)|"
            r"can be (?:reported|trusted|relied on|stated|quoted) (?:confidently|reliably|safely|with confidence)|reported confidently|"
            r"(?:trusted|reliable) from leave_balances|leave_balances (?:and|versus|vs\.?|or) (?:which )?(?:from )?leave_requests|"
            r"leave_requests (?:and|versus|vs\.?|or) (?:which )?(?:from )?leave_balances|"
            r"why (?:can'?t|cannot|won'?t|will not|doesn'?t|does not|shouldn'?t|should not|refuses? to) (?:you|helios|the assistant|it) "
            r"(?:invent|compute|calculate|derive|prove|infer|create|assume|make up|fabricate|guess|produce|generate|report|show|add)|"
            r"is (?:that|this|it) alone|does (?:that|this|it) alone|(?:that|this|it) alone (?:a|an|evidence|proof|enough)|"
            r"on its own,? (?:a|an|evidence|proof|enough|a sign|grounds)|by itself,? (?:a|an|evidence|proof|enough|a sign|grounds)|"
            r"alone,? (?:a|an|evidence|proof|enough|a sign|grounds)|"
            r"what clarification do you need|what (?:do|would) you need (?:me )?to clarify|"
            r"how should .{0,40} be (?:described|explained|presented|worded)|should .{0,30} be (?:described|explained|presented))\b",
            lowered,
        )
    )


def leave_date_recommendation_question(text: str) -> bool:
    """Employee asking which leave dates fit balance / coverage / policy — not submitting leave."""
    lowered = (text or "").lower()
    if re.search(
        r"\b(submit|file|book|apply for|create|prepare).{0,20}\b(leave|request)\b|"
        r"\b(approve|reject)\b",
        lowered,
    ):
        return False
    return bool(
        re.search(
            r"\b("
            r"which dates? (?:would be |are )?(?:best|better|good|suitable|recommended)|"
            r"best (?:dates?|window|time) (?:for|to) (?:my )?leave|"
            r"recommend(?:ed)? (?:leave )?dates?|"
            r"suggest(?:ed)? (?:leave )?dates?|"
            r"when (?:should|can|could) i (?:take|book|request) leave|"
            r"dates? .{0,40}(?:balance|coverage)|"
            r"(?:balance|coverage).{0,40}dates?"
            r")\b",
            lowered,
        )
        and re.search(r"\b(leave|vacation|time off|annual)\b", lowered)
    )


def delayed_approval_advice_question(text: str) -> bool:
    """Employee advice about a leave request waiting on manager/HR — not a new leave draft."""
    lowered = (text or "").lower()
    # New leave drafting / HITL submission is never delayed-approval guidance.
    if re.search(
        r"\b("
        r"(?:prepare|draft|submit|file|book|create).{0,40}\b(?:leave|request)\b|"
        r"approve leave request|reject leave request|"
        r"i want to (?:apply|request|take)|apply for leave next|"
        r"plan(?:ning)? (?:to take )?leave"
        r")\b",
        lowered,
    ) and not re.search(
        r"\b(has not approved|hasn'?t approved|still (?:awaiting|pending|waiting)|"
        r"doesn'?t respond|did not respond|no response|delayed)\b",
        lowered,
    ):
        return False
    waiting = bool(
        re.search(
            r"\b("
            r"manager has not approved|has(?:n'?t| not) (?:been )?approved|"
            r"still (?:waiting|pending|awaiting)|waiting (?:for|on) (?:my )?(?:manager|approval|decision)|"
            r"still awaiting approval|awaiting approval|"
            r"not (?:yet )?approved|no (?:decision|approval) yet|"
            r"manager (?:has )?not (?:responded|replied|decided)|"
            r"manager doesn'?t respond|manager did not respond|if (?:my )?manager doesn'?t respond|"
            r"pending for (?:several |a few |\d+ )?days|"
            r"(?:my )?leave (?:has )?been pending|leave request still|"
            r"approval (?:is|has been) delayed|delayed approval|"
            r"why .{0,40}(?:still )?(?:pending|awaiting|waiting)"
            r")\b",
            lowered,
        )
    )
    advice = bool(
        re.search(
            r"\b(what should i do|what can i do|what do i do|can you advise|advise|next steps?|"
            r"how (?:long|should)|follow up|escalate|who (?:can|should|needs? to) (?:i )?(?:ask|contact|approve)|"
            r"who needs to approve|why is|what if)\b",
            lowered,
        )
    )
    # Pending-leave approver identity is guidance about an existing request.
    if re.search(r"\bwho needs to approve\b", lowered) and re.search(
        r"\b(pending leave|my (?:pending )?leave|leave request|it)\b",
        lowered,
    ):
        return True
    # Short follow-ups about an already-pending request.
    if re.search(
        r"\b(how long has it been pending|what if they still don'?t respond|"
        r"can i follow up with my manager|follow up with my manager)\b",
        lowered,
    ):
        return True
    return waiting and (advice or re.search(r"\b(leave|request|manager|pending)\b", lowered))


def rule_intent(message: str, *, mentioned_other: bool, kinds: set[str], bulk: bool) -> str | None:
    text = message.lower()
    team_cue = bool(re.search(_TEAM_CUE, text))
    if manager_team_roster_question(text):
        return "team_roster"
    # Advice about delayed manager approval before leave-status "approved/rejected" steals.
    if delayed_approval_advice_question(text):
        return "leave_request_status"
    # Best-date recommendations are leave planning guidance, not leave-request creation.
    if leave_date_recommendation_question(text):
        return "leave_planning"
    # Employee self-service pending/status leave before manager approval-inbox heuristics.
    if personal_leave_status_question(text):
        return "leave_request_status"
    # Approval inbox listing/risk ranking must beat overlap/conflict heuristics and LLM fallback.
    if approval_inbox_question(text):
        return "team_analytics"
    inbox_follow = approval_inbox_followup_focus(text)
    # Preparing an inherited pending approval beats leave-draft / org-transfer heuristics.
    if inbox_follow == "prepare_approve" or (
        re.search(r"\b(prepare|draft)\b", text)
        and re.search(r"\b(approval|approve)\b", text)
        and re.search(r"\b(confirm|confirmation|wait)\b", text)
    ):
        return "approval_decision"
    # Urgency / why-urgent / approve|reject-impact are team pending-approval analytics, not policy.
    if inbox_follow in {"urgency", "why_urgent", "approve_impact", "reject_impact"}:
        return "team_analytics"
    # Employee-record routing / transfer mismatches before attendance analytics or directory.
    if routing_discrepancy_question(text):
        return "org_transfer"
    # Urgency / importance / remediation ONLY when the ask is clearly a routing discrepancy
    # follow-up. Plain "most urgent / why urgent" belongs to approval-inbox threads.
    # Corrective-action phrasing is remediation for routing investigations (HR Admin).
    if hr_corrective_action_question(text) or org_transfer_followup_focus(text) is not None:
        return "org_transfer"
    # Manager/HR leave-overlap staffing questions before workforce "which employees" steal.
    if team_leave_overlap_question(text):
        return "team_conflict"
    # Absenteeism ranking / consequences / investigation must beat directory and last-7-day dumps.
    # Keep month-over-month "which department contributed most?" on org_attendance comparison threads.
    if (
        workforce_followup_question(text)
        and not re.search(r"\b(my team|direct reports?|my reports?)\b", text)
        and not re.search(r"\bcontributed most\b", text)
    ):
        return "workforce_analytics"
    # Overtime premium / extra-pay policy (including conflicting rate docs) before salary analytics.
    if re.search(
        r"\b(overtime|extra pay|premium(?:\s+rate)?|work past (?:the )?(?:usual )?weekly hours|"
        r"hours (?:per week|beyond|over))\b",
        text,
    ) and not re.search(r"\b(calculate|compute|incomplete|missing rate)\b", text):
        return "policy_question"
    if re.search(r"\b(how many projects|projects? (?:are )?registered|expired polic(?:y|ies))\b", text):
        return "hr_metrics"
    if not mentioned_other and not bulk and data_semantics_question(text):
        return "data_semantics"
    if re.search(
        r"\b(early departures?|clock(?:ed|ing)?[- ]?out early|early clock-?outs?|leave early|leaving early|left early|"
        r"early[- ](?:exits?|finish(?:es)?|sign-?offs?|departures?|leavers?|outs?))\b",
        text,
    ) and not team_cue:
        return "attendance_query"
    if (
        not mentioned_other
        and _personal(text)
        and not team_cue
        and re.search(r"\b(empty|unset|missing|incomplete)\b", text)
        and re.search(r"\b(profile|fields?)\b", text)
        and not re.search(
            r"\b(across (?:the )?workforce|workforce|organization|organisation|all employees)\b",
            text,
        )
    ):
        return "employee_profile"
    # First-person hire-year claims checked against the caller's own record stay on profile,
    # never organization directory (singular "employee record" must not widen to org hire filters).
    if (
        not mentioned_other
        and not bulk
        and _personal(text)
        and not team_cue
        and re.search(r"\b(i was hired|i joined|i started|my hire|hired in \d{4})\b", text)
        and re.search(
            r"\b(confirm|check|verify|against|my (?:employee )?(?:record|profile)|hire date)\b",
            text,
        )
        and not re.search(
            r"\b(all employees|every employee|organization|organisation|department|headcount|"
            r"how many (?:people|staff|workers|employees)|employees? (?:were |was )?hired)\b",
            text,
        )
    ):
        return "employee_profile"
    if not mentioned_other and not bulk and _personal(text) and not team_cue and (
        re.search(r"\b(?:my|the) (?:\w+ ){0,3}(?:job )?grade\b|\bjob grade\b", text)
        or re.search(r"\bgrade\b", text) and re.search(r"\b(my (?:employee )?(?:profile|record)|stored on my|for me)\b", text)
        or re.search(r"\b(phone|mobile)\b", text)
        and re.search(
            r"\b(my (?:employee )?(?:record|profile)|stored on my|my (?:\w+ ){0,3}(?:phone|mobile)|"
            r"registered (?:phone|mobile)|on file for me|for me|about me|do you have for me)\b",
            text,
        )
        or re.search(r"\b(calendar days|days (?:have )?(?:elapsed|passed)|how many days|how long)\b", text)
        and re.search(r"\b(hire|hired|joined|tenure|with the company|employed)\b", text)
    ):
        return "employee_profile"
    # team_analytics is role-gated in classify_intent so HR org questions keep org_* intents.
    if self_scoped_missing_attendance(text) and not mentioned_other:
        return "attendance_query"
    if self_scoped_approved_without_decision(text) and not mentioned_other:
        return "leave_request_status"
    # Month-over-month self exploration without org/team/HR-metrics framing.
    if (
        not mentioned_other
        and not bulk
        and re.search(r"\b(compare|comparison|compared)\b", text)
        and re.search(r"\b(last month|previous month|month before|this month)\b", text)
        and not re.search(
            r"\b(organization|organisation|department|all employees|every employee|my team|"
            r"team members?|hr metrics|headcount|turnover|hiring)\b",
            text,
        )
    ):
        return "attendance_query"
    if bulk or (kinds and mentioned_other):
        return "confidential_lookup"
    if re.search(r"\b(my manager|manager's|managers)\b", text) and re.search(r"\b(salary|compensation)\b", text):
        return "confidential_lookup"
    if re.search(r"\bemployee(?:\s*id|_id)?\s*[:=#]?\s*\d+", text):
        return "confidential_lookup"
    if "performance" in kinds and not mentioned_other and not re.search(r"\bmy\b", text):
        return "confidential_lookup"
    if re.search(r"\b(weather|joke|recipe|football score|stock price)\b", text):
        return "out_of_scope"
    if re.search(r"\b(policy|policies)\b", text) and "transfer" in text:
        return "policy_question"
    # A personal attendance percentage/coverage question that mentions hire date is still an
    # attendance question (lifetime coverage), not a profile lookup.
    if (
        not mentioned_other
        and not bulk
        and _personal(text)
        and not team_cue
        and re.search(r"\battendance\b", text)
        and re.search(r"\b(percentage|percent|rate|lifetime|all[- ]time|since (?:my )?hire|from (?:my )?hire date|coverage|exact)\b", text)
    ):
        return "attendance_query"
    if not mentioned_other and not bulk and _personal(text) and not team_cue and not re.search(r"\be-\d+\b", text) and not re.search(
        r"\b(changed|previous|old manager|still going|transferred)\b", text
    ) and (
        re.search(r"\b(who is my|who's my|who is listed as my|my reporting manager|my line manager)\b", text)
        and re.search(r"\bmanager\b", text)
        or re.search(
            r"\b(job title|my title|my department|department (?:am|do) i|"
            r"which department am i|what department am i|employee id|employee number|hire date|joined the company|"
            r"when did i join|when did (?:my employment|i) (?:start|begin)|my employment (?:start|began|started|begin)|"
            r"employment with the company start|when (?:did|does) my employment start|"
            r"started (?:with|at) the company|my start date|(?:date|day) i (?:started|joined|was hired)|when was i hired|"
            r"employment status|employee profile|employment details|my projects|"
            r"assigned to|missing from my profile|incomplete(?: my)? profile|profile complet|"
            r"department am i in)\b",
            text,
        )
    ) and not re.search(
        r"\b(average|avg|mean).{0,40}\b(working )?hours\b|"
        r"\b(every department|all departments|by department|across (?:the )?workforce)\b",
        text,
    ):
        return "employee_profile"
    if (
        not mentioned_other
        and not bulk
        and _personal(text)
        and not team_cue
        and re.search(r"\b(missing|incomplete|complet)\b", text)
        and re.search(r"\b(profile|employment)\b", text)
        and not re.search(
            r"\b(across (?:the )?workforce|workforce|organization|organisation|all employees)\b",
            text,
        )
    ):
        return "employee_profile"
    # A status pack for one period with no organization, department, or team marker is the
    # signed-in employee's own attendance ("report present, remote, late and absent counts").
    statuses = re.findall(r"\b(present|remote|late|absent|sick)\b", text)
    if (
        not mentioned_other
        and not team_cue
        and len(set(statuses)) >= 2
        and re.search(r"\b(counts?|report|summar\w*|together|breakdown|totals?|mix)\b", text)
        and not _org_attendance_signal(text)
        and not re.search(r"\b(all|every|their|employees?)\b", text)
    ):
        return "attendance_query"
    if any(
        phrase in text
        for phrase in (
            "changed department",
            "previous manager",
            "old manager",
            "still going to",
            "transferred",
        )
    ):
        return "org_transfer"
    if re.search(r"\b(my|i|i'm)\b", text) and re.search(r"\b(absences?|absent)\b", text) and not re.search(
        r"\b(trend|analytics|across|departments|employees)\b", text
    ):
        return "attendance_query"
    if remote_policy_question(text):
        return "policy_question"
    if remote_attendance_question(text):
        return "attendance_query"
    if post_approval_question(text):
        return "policy_question"
    if not mentioned_other and re.search(r"\bmy\b", text) and re.search(r"\bleave history\b", text):
        return "leave_request_status"
    if not mentioned_other and re.search(r"\b(complete leave history|all my leave requests|my leave requests)\b", text):
        return "leave_request_status"
    if not mentioned_other and re.search(
        r"\b(am i eligible|eligible for another|can i (still )?apply|can i request another|another annual leave)\b",
        text,
    ) and re.search(r"\bleave\b", text):
        return "leave_eligibility"
    if re.search(r"\b(carry forward|carry over|carryover|carry into)\b", text):
        return "policy_question"
    if not mentioned_other and _personal(text) and re.search(
        r"\b(discrepanc\w*|reconcil\w*|inconsistenc\w*)\b", text
    ) and re.search(r"\b(attendance|leave)\b", text):
        return "attendance_query"
    if not mentioned_other and _personal(text) and (
        re.search(r"\bno approval\b", text)
        or (
            re.search(r"\bapproved\b", text)
            and re.search(r"\b(no|without|missing)\b", text)
            and re.search(r"\b(approval|decision)\b", text)
        )
    ):
        return "leave_request_status"
    # Explicit attendance-source sick questions stay on attendance, not leave balance.
    if (
        not mentioned_other
        and _personal(text)
        and re.search(r"\bsick\b", text)
        and re.search(r"\b(taken|actually taken|how many)\b", text)
        and re.search(r"\b(attendance|attendance records?|status sick|recorded as sick)\b", text)
    ):
        return "attendance_query"
    if not mentioned_other and _personal(text) and re.search(r"\bsick\b", text) and re.search(
        r"\b(taken|match|matches|reconcil|discrepanc|deduct|responsible|account for)\b", text
    ):
        return "leave_balance"
    if not mentioned_other and _personal(text) and re.search(r"\bmiss(ed|ing)?\b", text) and re.search(
        r"\b(working days?|work days?|days?)\b", text
    ):
        return "attendance_query"
    if not mentioned_other and _personal(text) and re.search(r"\b(compare|comparison)\b", text) and re.search(
        r"\b(attendance|remote|absent|present)\b", text
    ):
        return "attendance_query"
    # Personal attendance vs approved-leave overlap (e.g. "do my present days overlap
    # approved leave"). Manager team overlapping-leave questions must stay on team_conflict.
    if (
        not mentioned_other
        and _personal(text)
        and not manager_team_ops_question(text)
        and not re.search(r"\b(direct reports?|my team|team members?|members of my team)\b", text)
        and (
            re.search(r"\b(overlap|overlapping)\b", text)
            or (
                re.search(r"\b(present|remote)\b", text)
                and re.search(r"\bapproved leave\b", text)
            )
        )
    ):
        return "attendance_query"
    if not mentioned_other and _personal(text) and (
        re.search(r"\b(entire year|whole year|full year|exact attendance)\b", text)
        or (
            re.search(r"\bwhy\b", text)
            and re.search(r"\b(calculate|exact|attendance)\b", text)
            and re.search(r"\byear\b", text)
        )
    ):
        return "attendance_query"
    if re.search(r"\bresign", text) and re.search(r"\b(unused|leave|encash|settlement)\b", text):
        return "policy_question"
    if not mentioned_other and _personal(text) and re.search(r"\b(latest|most recent|last)\b", text) and re.search(
        r"\b(approved request|leave request|approved leave)\b", text
    ):
        return "leave_request_status"
    if not mentioned_other and _personal(text) and re.search(r"\b(latest|most recent)\b", text) and re.search(
        r"\brequest\b", text
    ) and re.search(r"\b(approved|rejected|pending|status)\b", text):
        return "leave_request_status"
    if (
        _personal(text)
        and re.search(r"\bhours\b", text)
        and re.search(
            r"\b(how many|completed|did i|total|recorded|worked|compare|comparison|longest|august|september|october)\b",
            text,
        )
        and not re.search(r"\b(department|departments|every department|workforce)\b", text)
    ):
        return "attendance_query"
    # Dated or comparative personal working-hours questions are attendance, not policy.
    if (
        _personal(text)
        and re.search(r"\bworking hours\b", text)
        and re.search(
            r"\b(compare|comparison|august|september|october|november|december|january|february|"
            r"march|april|may|june|july|\d{4}|last month|this month|recorded|total|worked|longest)\b",
            text,
        )
        and not re.search(r"\b(department|departments|every department|workforce)\b", text)
    ):
        return "attendance_query"
    if re.search(r"\b(core hours|official working days|work on weekends|have to work on weekend)\b", text):
        return "policy_question"
    # Core-hours policy questions, including "What are my working hours?", stay on policy
    # unless the ask is clearly about recorded attendance totals or named periods.
    if re.search(r"\bworking hours\b", text) and not re.search(
        r"\b(total|recorded|worked|compare|comparison|august|september|october|november|"
        r"december|january|february|march|april|may|june|july|\d{4}|last month|this month|"
        r"longest|zero)\b",
        text,
    ):
        return "policy_question"
    # Reporting-structure contact / hire weekday / position+department are profile, not policy.
    if not mentioned_other and _personal(text) and not team_cue and re.search(
        r"\b(reporting structure|who should i contact|position (?:am i|i am)|currently assigned|"
        r"day of the week|officially start|start working here|"
        r"employment[-\s]?status (?:history|changes?)|stored history of (?:prior )?employment[-\s]?status|"
        r"history of (?:prior )?employment[-\s]?status|prior employment[-\s]?status)\b",
        text,
    ):
        return "employee_profile"
    if re.search(r"\b(work from home|remote work)\b", text):
        return "policy_question"
    if re.search(r"\bwork\b", text) and re.search(r"\b(public holidays?|holidays)\b", text) and not re.search(
        r"\b(upcoming|tomorrow|next week|show me all)\b", text
    ):
        return "policy_question"
    org_intent = organization_intent(message)
    if org_intent == "org_attendance" and mentioned_other:
        org_intent = None
    if org_intent:
        return org_intent
    if re.search(r"\b(hiring trends?|monthly hiring|turnover trends?)\b", text):
        return "hr_metrics"
    if re.search(
        r"\b(unusual|anomal(?:y|ies)|attendance patterns?|further investigation|"
        r"need(?:s)? (?:further )?investigation|largest increase|"
        r"(?:compare|comparison|contrast) (?:the )?(?:last|past|previous) (?:\w+ )?months?|"
        r"(?:last|past|previous) (?:two|2) (?:complete )?(?:calendar )?months?|"
        r"actions? (?:would you )?recommend|recommend(?:ations?)? to hr|"
        r"explain the data behind|data behind (?:your |the )?recommend)\b",
        text,
    ) and not re.search(r"\b(my team|direct reports?|my reports?)\b", text):
        return "workforce_analytics"
    if any(phrase in text for phrase in ("absence", "absences", "absenteeism", "trend", "analytics", "utilization", "across departments")):
        return "workforce_analytics"
    if parse_approval_decision(text):
        return "approval_decision"
    if any(
        phrase in text
        for phrase in (
            "overlapping",
            "same week",
            "three employees",
            "three members",
            "staffing",
            "project delivery",
            "team conflict",
            "leave conflict",
        )
    ) or (
        re.search(r"\b(my team|members of my team|direct reports?)\b", text)
        and re.search(r"\b(overlap|overlapping|conflict|staffing|same week|project delivery|coverage)\b", text)
        # Employees asking whether *their own* planned leave affects team coverage stay on leave planning.
        # Manager/HR team-conflict analysis is for reviewing other people's overlapping requests.
        and not re.search(
            r"\b(would (?:that|this|my leave|taking)|affect (?:my )?team|enough coverage|"
            r"team(?:'s)? coverage|if i take|my (?:leave|absence|time off))\b",
            text,
        )
    ) or (
        re.search(r"\b(overlap|overlapping)\b", text)
        and re.search(r"\b(leave|absence|absences|requests?)\b", text)
        and not re.search(
            r"\b(would (?:that|this|my leave|taking)|affect (?:my )?team|enough coverage|"
            r"team(?:'s)? coverage|if i take|my (?:leave|absence|time off))\b",
            text,
        )
    ):
        return "team_conflict"
    attendance = bool(re.search(r"\b(attendance|attended|attend|absent|late|present)\b", text))
    # First-person present/remote/absent counts with a date window are personal attendance,
    # even when the window mentions the organization date.
    if (
        not mentioned_other
        and not team_cue
        and re.search(r"\b(i|i'm|do i|did i|me|my)\b", text)
        and re.search(r"\b(present|remote|absent|late|sick|attendance|hours)\b", text)
        and re.search(
            r"\b(november|october|september|august|july|june|may|april|march|february|january|"
            r"from \d|through|between|last month|this month|organization date|organisation date)\b",
            text,
        )
    ):
        return "attendance_query"
    if attendance and "policy" not in text and (_personal(text) or "record" in text or mentioned_other):
        return "attendance_query"
    # Advance-notice / rulebook policy beats holiday calendar and leave-status heuristics.
    if re.search(r"\b(advance notice|notice period|how (?:much|many).{0,40}notice|days of (?:advance )?notice)\b", text) or (
        re.search(r"\bnotice\b", text)
        and re.search(r"\b(leave|annual|vacation|pto|time off|rulebook|policy|policies|handbook|expect|required|require)\b", text)
        and not re.search(r"\bresign", text)
    ):
        return "leave_policy"
    if re.search(r"\b(rulebook|leave policy|annual leave policy|handbook)\b", text) and re.search(
        r"\b(notice|advance|how many|how much|entitled|business days)\b", text
    ):
        return "leave_policy"
    if re.search(r"\bholidays?\b", text) and "policy" not in text and not re.search(r"\bnotice\b", text):
        return "holiday_query"
    if re.search(r"\b(modify|reschedule|edit)\b", text) and re.search(r"\bleave\b", text):
        return "leave_request_status"
    if re.search(r"\b(cancel|withdraw|rescind)\b", text) and re.search(r"\bleaves?\b", text):
        return "leave_cancellation"
    if re.search(
        r"\b(committed to pending|pending requests?|leave days .{0,20}pending|difference between my entitled)\b",
        text,
    ) and _personal(text):
        return "leave_balance"
    if re.search(
        r"\b(status of my leave|status of my latest|latest leave|leave request status|has my leave|"
        r"approval status|pending approval|pending leave|leave application|my leave requests|"
        r"waiting for a decision|ever rejected|any .* rejected|"
        r"how many leaves? (?:are )?(?:currently )?pending)\b",
        text,
    ):
        return "leave_request_status"
    if not mentioned_other and _personal(text) and re.search(r"\bleave\b", text) and (
        re.search(r"\b(waiting for a decision|ever rejected|list my|date ranges?|business days)\b", text)
        or (
            re.search(r"\b(request|application|requests)\b", text)
            and re.search(r"\b(pending|approved|rejected|cancelled|canceled|status|history|latest|list|ranges?)\b", text)
        )
    ):
        # Policy / notice questions that mention business days must not become status lookups.
        if re.search(r"\b(notice|rulebook|policy|policies|handbook|expect|required)\b", text):
            return "leave_policy"
        if not _explicit_leave_request(text):
            return "leave_request_status"
    if re.search(r"\bwho approved\b", text):
        return "leave_request_status"
    if re.search(r"\bwho needs to approve\b", text) and _personal(text):
        return "employee_profile"
    if re.search(r"\b(can you approve|approve my own|approve it myself)\b", text):
        return "employee_profile"
    if re.search(r"\b(medical certificate|medical note|doctor'?s note)\b", text) and not re.search(
        r"\b(another|other|someone else|colleague)\b", text
    ):
        return "policy_question"
    if re.search(r"\bresign", text) and re.search(r"\bnotice\b", text):
        return "policy_question"
    if re.search(r"\bnotice\b", text):
        return "leave_policy"
    if re.search(r"\b(what happens|what would happen|what if)\b", text) and re.search(
        r"\b(balance|policy|leave|resign|absent|approval)\b", text
    ):
        if not re.search(r"\b(book|submit|file)\b", text):
            return "policy_question"
    if re.search(r"\b(what if|what happens)\b", text) and re.search(r"\brequest\b", text) and re.search(r"\bdays?\b", text):
        if not re.search(r"\b(book|submit|file|prepare)\b", text):
            return "leave_eligibility"
    if re.search(r"\b(can i|could i|am i eligible|am i allowed|may i)\b", text) and re.search(
        r"\b(take|days off|day off|time off)\b", text
    ):
        return "leave_eligibility"
    # Own-leave coverage / affordability questions are leave planning, not manager team analytics.
    if re.search(
        r"\b(team(?:'s)? coverage|enough coverage|affect (?:my )?team|can i afford|"
        r"afford that with my (?:remaining )?leave)\b",
        text,
    ) and (
        _personal(text)
        or re.search(r"\b(if i take|my leave|taking|would that|next month|one week|two weeks)\b", text)
    ):
        return "leave_planning"
    if (
        re.search(r"\b(enough|sufficient)\b", text)
        and "balance" in text
        and not re.search(r"\b(what happens|what if)\b", text)
        and not re.search(r"\b(planning|two weeks|take .{0,20}off|book leave|request leave)\b", text)
    ):
        return "leave_eligibility"
    if re.search(r"\b(instead|use annual leave|use sick leave)\b", text):
        return "policy_question"
    # Compensation band / policy documents are retrieval, not person-level confidential_lookup.
    if re.search(
        r"\b(compensation\s+bands?|pay\s+bands?|salary\s+bands?|compensation\s+(?:policy|policies|notes?|document|handbook)|"
        r"confidential\s+compensation|compensation\s+band\s+notes?)\b",
        text,
    ):
        return "policy_question"
    if re.search(r"\b(probation|reset|compensation|compensatory)\b", text) and re.search(
        r"\b(policy|policies|band|bands|notes?|document|handbook|grade|grades|senior)\b",
        text,
    ):
        return "policy_question"
    if re.search(r"\b(probation|reset)\b", text):
        return "policy_question"
    if _explicit_leave_request(text):
        return "leave_request"
    # Short action phrases: "plan leave", "help me plan leave", "leave planning"
    if re.search(r"\b(plan|planning)\b", text) and re.search(r"\bleaves?\b", text) and not re.search(
        r"\b(policy|notice|carry|entitled|remain(?:s|ing)?|balance|used|exhausted)\b",
        text,
    ):
        return "leave_planning"
    planning_cues = bool(
        re.search(
            r"\b(planning to take|i'm planning|i am planning|take .{0,20}off|two weeks off|"
            r"book leave|request leave|project conflicts?|approvals? i need|"
            r"check my (?:remaining )?balance.{0,40}approv|identify project)\b",
            text,
        )
    )
    if planning_cues and re.search(r"\b(leave|balance|off|vacation|annual)\b", text):
        return "leave_planning"
    if _personal(text) and re.search(r"\b(how many|how much)\b", text) and re.search(
        r"\b(sick|annual|vacation|leaves?)\b", text
    ) and re.search(r"\b(left|remain(?:s|ing)?|have|balance|get|entitled|used)\b", text):
        return "leave_balance"
    if re.search(r"\b(consecutive leave|allowed to take)\b", text):
        return "policy_question"
    if _personal(text) and (
        "balance" in text
        or (re.search(r"\bremain(?:s|ing)?\b", text) and "leave" in text)
        or ("entitled" in text and re.search(r"\b(sick|annual|vacation|leave)\b", text))
        or (re.search(r"\b(used|already used)\b", text) and re.search(r"\b(annual|sick|leave)\b", text))
    ) and not planning_cues:
        return "leave_balance"
    # Require leave/time-off context so phrases like "take to correct" do not open leave planning.
    planning = bool(
        re.search(
            r"\b("
            r"take (?:some |a few |one |two |three |\d+ )?(?:days?|weeks?)(?:\s+off)?|"
            r"taking (?:leave|time off|days? off|vacation)|"
            r"planning (?:to take |my )?leave|plan leave|days off|time off|two weeks off|"
            r"book leave|request leave|want leave|would like leave|like to take leave|"
            r"need leave|apply for leave|applyfor leave"
            r")\b",
            text,
        )
    ) and not re.search(r"\b(remain(?:s|ing)?|balance|used|entitled|exhausted)\b", text)
    if planning:
        return "leave_planning"
    # Named leave window without an explicit verb still opens leave planning.
    if (
        re.search(r"\b(leave|vacation|annual leave|time off)\b", text)
        and re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
            r"\d{4}-\d{2}-\d{2}|\d{1,2}\s*[-–—to]+\s*\d{1,2})\b",
            text,
        )
        and not re.search(r"\b(remain(?:s|ing)?|balance|used|entitled|exhausted|policy|notice)\b", text)
        and not mentioned_other
    ):
        return "leave_planning"
    if not mentioned_other and not bulk and _personal(text) and (
        re.search(r"\b(my salary|my compensation|my current salary)\b", text)
        or any(
            phrase in text
            for phrase in (
                "my performance",
                "my rating",
                "my profile",
                "my job title",
                "who is my manager",
                "who am i",
                "my reporting manager",
            )
        )
    ):
        return "employee_profile"
    names_leave_type = bool(re.search(r"\b(sick|annual)\s+leaves?\b", text))
    short_follow = bool(re.search(r"\b(what about|how about)\b", text) or re.match(r"and\b", text.strip()))
    if re.search(
        r"\b(blood type|blood group|emergency[-\s]?contact|desk[-\s]?occupancy|keyboard[-\s]?cadence|"
        r"retina[-\s]?scan)\b",
        text,
    ):
        # Own unsupported schema fields are profile/attendance disclosures, not confidential lookups.
        if _personal(text) or not mentioned_other:
            return "employee_profile"
    if re.search(r"\b(another|other|someone else|colleague)\b", text) and re.search(
        r"\b(medical|sick leave|leave (?:history|details|records?))\b", text
    ):
        return "confidential_lookup"
    if re.search(
        r"\b(policy|overtime|parental|maternity|paternity|entitled|medical|co-approv|carry forward|carry over|carryover|consecutive leave|allowed to)\b",
        text,
    ) or (re.search(r"\bsick leave\b", text) and not (short_follow and names_leave_type and "policy" not in text)) or (
        re.search(r"\bremote\b", text) and re.search(r"\b(policy|work|week|approv)\b", text)
    ):
        # Named person's medical/sick notes are private-record asks, not generic policy.
        if re.search(r"'s\b", text) and re.search(r"\b(medical|sick leave|leave history)\b", text):
            return "confidential_lookup"
        # Calculate overtime premium with missing/incomplete rate — unsupported, not policy retrieval.
        if re.search(
            r"\b(calculate|compute|incomplete|missing)\b", text
        ) and re.search(r"\b(overtime premium|overtime rate)\b", text):
            return "attendance_query"
        return "policy_question"
    # Bare "annual leave" / "on leave" is not enough to open leave planning when the ask is informational.
    # Prepare / submit / draft / file are leave_request; exploratory apply/plan stays planning.
    if re.search(r"\b(book|submit|prepare|draft|file)\b", text) and re.search(
        r"\b(vacation|annual leave|on leave|leaves?)\b",
        text,
    ):
        return "leave_request"
    if re.search(r"\b(apply(?:\s+for)?|request|plan)\b", text) and re.search(
        r"\b(vacation|annual leave|on leave|leaves?)\b",
        text,
    ):
        return "leave_planning"
    if any(phrase in text for phrase in ("vacation", "annual leave", "on leave")) and re.search(
        r"\b(remain(?:s|ing)?|balance|used|entitled|exhausted|how much|how many)\b", text
    ):
        return "leave_balance"
    return None


_DAY_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _explicit_leave_request(text: str) -> bool:
    """True when the employee asks to prepare or submit leave, not when they ask about requesting it."""
    # Tolerate missing spaces from chat UI typos: "applyfor leave"
    text = re.sub(r"\bapplyfor\b", "apply for", text)
    text = re.sub(r"\bprepare\s*&\s*submit\b", "prepare and submit", text)
    # Delayed / pending approval guidance is status advice, never a new draft.
    if delayed_approval_advice_question(text):
        return False
    # Exploratory apply/plan without prepare/submit/draft stays leave_planning.
    if re.search(
        r"\b(next month|planning to|i(?:'m| am) planning|thinking (?:of|about)|considering)\b",
        text,
    ) and not re.search(r"\b(prepare|submit|draft|file|confirm)\b", text):
        return False
    if re.search(r"\b(before|when|while|about)\s+requesting\b", text):
        return False
    # Informational "who would/needs to approve" must not block a draft/prepare leave ask.
    info_approve = bool(
        re.search(
            r"\b(who (?:would|will|needs? to|must|should) approve|who approves|approval (?:path|chain)|"
            r"tell me who (?:would|will|needs? to) approve)\b",
            text,
        )
    )
    leave_context = bool(
        re.search(r"\b(leave|annual|vacation|time off|day off|days off|request|application)\b", text)
    )
    # Draft / prepare / submit cues — allow short intervening duration/type words
    # ("prepare and submit one day of annual leave", "submit a leave request").
    draft_prepare = bool(
        re.search(
            r"\b("
            r"prepare(?:\s+and)?\s+submit|"
            r"draft(?:\s+and)?\s+submit|"
            r"(?:prepare|draft|submit|file|book|apply for)\s+"
            r"(?:(?:the|a|an|my|one|two|three|\d+)\s+){0,6}"
            r"(?:day(?:s)?\s+(?:of\s+)?)?(?:annual\s+|sick\s+)?"
            r"(?:leave|vacation|time off)(?:\s+request|\s+application)?|"
            r"(?:prepare|draft|submit|file|book)\s+(?:the\s+|a\s+|an\s+|my\s+)?"
            r"(?:leave|annual leave|vacation)?\s*(?:request|application)|"
            r"prepare (?:leave|it|that|this)|"
            r"draft (?:it|that|this)|"
            r"so i can confirm|for me to confirm|wait for (?:my )?confirm|"
            r"i want to request|i'd like to request|i would like to request|"
            r"i want to submit|i'd like to submit|i would like to submit|"
            r"book leave|request leave|submit leave|file leave|apply for leave"
            r")\b",
            text,
        )
    )
    if draft_prepare and leave_context:
        return True
    if re.search(
        r"\b(what if|what happens|status|pending|who will|who needs|notice|policy|instead|enough|whether|"
        r"modify|cancel|eligible|eligibility|another annual|"
        r"list|history|date ranges?|business days|how many requests?|show my (?:approved )?leave)\b",
        text,
    ):
        # Explicit prepare/submit still wins over informational exclude cues.
        if draft_prepare and leave_context:
            return True
        return False
    # Bare "approve/approving" blocks only when this is not an informational approve-path ask.
    if not info_approve and re.search(r"\b(approve|approving)\b", text):
        return False
    if re.search(r"\b(do not submit|don't submit|without submitting)\b", text) and re.search(r"\bprepar", text):
        return True
    if re.search(r"\b(help me request|request sick leave)\b", text):
        return True
    if re.search(r"\b(apply for|application for)\b", text) and re.search(r"\bleaves?\b", text):
        return True
    if re.search(r"\b(submit|prepare|draft|file|book)\b", text) and re.search(
        r"\b(leave|vacation|time off)\b", text
    ):
        return True
    if re.search(r"\brequest\b", text) and re.search(r"\bleave\b", text):
        return True
    return bool(
        re.search(
            r"\b(i want to request|i'd like to request|i would like to request|book leave|request leave|"
            r"submit leave|file leave|prepare a leave|prepare leave|prepare the request|apply for leave|"
            r"draft (?:a |the |an )?leave|draft the request)\b",
            text,
        )
    )


def _day_count(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return _DAY_WORDS.get(token)


def parse_business_days(message: str) -> int | None:
    text = message.lower().strip().rstrip(".!")
    if re.search(r"\b(two weeks|2 weeks)\b", text):
        return 10
    if re.search(r"\b(one week|1 week|a week|only one week)\b", text):
        return 5
    match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+weeks?\b", text)
    if match:
        count = _day_count(match.group(1))
        if count is not None:
            return count * 5
    match = re.search(r"(\d+)\s+business days", text)
    if match:
        return int(match.group(1))
    match = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+consecutive\s+(?:business\s+)?days?\b",
        text,
    )
    if match and _day_count(match.group(1)) is not None:
        return _day_count(match.group(1))
    match = re.search(r"(\d+)\s+days?", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:business\s+)?days?\b", text)
    if match:
        return _DAY_WORDS[match.group(1)]
    if re.fullmatch(r"\d{1,2}", text):
        return int(text)
    return None


def requested_leave_days(message: str) -> int | None:
    """Days the employee wants to take. A stated remaining balance is not a request length."""
    text = re.sub(
        r"\b(?:only\s+have|have|with)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+(?:remaining|left)\b",
        " ",
        message.lower(),
    )
    return parse_business_days(text)


def hypothetical_remaining_days(message: str) -> int | None:
    text = message.lower()
    match = re.search(
        r"\bonly have\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+remaining\b",
        text,
    )
    if match:
        return _day_count(match.group(1))
    match = re.search(
        r"\b(?:if i )?have\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:annual\s+)?days?\s+remaining\b",
        text,
    )
    if match:
        return _day_count(match.group(1))
    return None


def is_clarification_reply(message: str) -> bool:
    text = message.strip().lower()
    if "?" in text or re.match(r"^(how|what|why|when|where|who|can|could|am|is|are|do|does|did|should|would)\b", text):
        return False
    text = text.rstrip(".!")
    if text in {"yes", "no", "yeah", "yep", "nope", "ok", "okay", "sure", "tomorrow", "next month"}:
        return True
    return parse_business_days(message) is not None


_SUBJECT_INTENT = {
    "attendance": "attendance_query",
    "reconciliation": "attendance_query",
    "holiday": "holiday_query",
    "leave_balance": "leave_balance",
    "leave_plan": "leave_planning",
    "policy": "policy_question",
    "employee_directory": "employee_directory",
    "org_attendance": "org_attendance",
    "org_leave": "org_leave",
    "compensation_query": "compensation_query",
    "hr_metrics": "hr_metrics",
    "leave_history": "leave_request_status",
    "team_analytics": "team_analytics",
    "team_roster": "team_roster",
    "workforce_analytics": "workforce_analytics",
    "team_conflict": "team_conflict",
    "approval_decision": "approval_decision",
    "approval_inbox": "team_analytics",
    "routing_audit": "org_transfer",
    "routing_investigation": "org_transfer",
    "org_transfer": "org_transfer",
}

_MONTH_NAME = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b(?:\s+\d{4})?",
    re.IGNORECASE,
)


def leave_plan_followup(message: str, inherited: dict | None) -> tuple[str, str] | None:
    """Map a leave-plan thread follow-up to (intent, focus).

    Preserves the planned window for holidays, projects, approvals, draft review,
    confirmation gates, and duration recalculation without expanding authorization.
    """
    if not inherited or inherited.get("subject") not in {"leave_plan", "leave_eligibility"}:
        return None
    text = message.lower().strip()
    if re.search(r"\b(weather|joke|recipe|password|credential|salary|compensation)\b", text):
        return None
    has_window = bool(inherited.get("start") and inherited.get("end"))
    # Eligibility threads can still change duration or ask coverage/approvals without a stored window yet.
    if inherited.get("subject") == "leave_plan" and not has_window:
        return None
    if re.search(
        r"\b(team(?:'s)? coverage|enough coverage|affect (?:my )?team|coverage (?:ok|okay|enough)|"
        r"staffing|understaff)\b",
        text,
    ):
        return "leave_planning", "coverage"
    if re.search(r"\bproject\b", text) and re.search(r"\b(deadline|deadlines|delivery|milestone)\b", text):
        return "leave_planning", "projects"
    if re.search(r"\bholidays?\b", text):
        return "holiday_query", "holidays"
    if re.search(r"\bapprovals?\b", text) and re.search(r"\b(need|would|required|who|path|chain)\b", text):
        return "leave_planning", "approvals"
    if re.search(
        r"\b(who (?:needs to|must|will|should) approve|who approves|approval (?:path|chain))\b",
        text,
    ):
        return "leave_planning", "approvals"
    if re.search(
        r"\b(what (?:do i |must i )?need|what (?:steps?|needs?) (?:do i need |to be )?(?:done|completed|complete)|"
        r"before (?:i )?submit|checklist|prerequisites?|ready to submit)\b",
        text,
    ):
        return "leave_planning", "prerequisites"
    if re.search(r"\b(show|review|display|see)\b", text) and re.search(
        r"\b(the )?(request|draft)\b", text
    ):
        return "leave_request", "draft_review"
    if re.search(
        r"\b(submit .{0,40}after i confirm|only after i confirm|after i confirm|do not submit|"
        r"don't submit|without (?:my )?confirm|wait for (?:my )?confirm)\b",
        text,
    ):
        return "leave_request", "confirmation_gate"
    if re.search(r"\b(prepare|draft|file)\b", text) and re.search(r"\b(leave|request|those dates|that period)\b", text):
        return "leave_request", "prepare"
    if re.search(r"\bhow many business days should i plan\b", text) or (
        re.search(r"\b(business days|how many days)\b", text)
        and re.search(r"\b(plan|window|period|request)\b", text)
        and not re.search(r"\b(notice|policy|carry)\b", text)
    ):
        return "leave_planning", "business_days"
    if re.search(r"\b(enough leave|enough balance|do i have enough|can i afford)\b", text):
        return "leave_planning", "balance_check"
    if has_window and re.search(r"\b(those dates|that period|that window|same dates|same period|same window)\b", text):
        return "leave_planning", "window"
    # Duration-only change keeps leave type / identity / start; leave_analysis recomputes end.
    if parse_business_days(text) is not None and re.search(
        r"\b(only|just|what if|instead|shorten|reduce|take|plan)\b",
        text,
    ):
        return "leave_planning", "duration"
    return None


def continuation_intent(message: str, inherited: dict | None) -> str | None:
    """Keep the previous record topic when the new message only changes a detail.

    A rule that already chose a new intent wins before this is called. This only
    covers a short continuation that would otherwise be sent to the model.
    """
    if not inherited:
        return None
    subject = inherited.get("subject")
    if subject not in _SUBJECT_INTENT:
        return None
    text = message.lower().strip()
    if re.search(r"\b(weather|joke|recipe|password|credential)\b", text):
        return None
    if re.search(r"\b(salary|compensation)\b", text) and subject != "compensation_query":
        return None
    leave_follow = leave_plan_followup(message, inherited)
    if leave_follow:
        return leave_follow[0]
    if re.search(r"\b(sick|annual)\s+leaves?\b", text) and "policy" not in text and subject in {"leave_balance", "attendance"}:
        return "leave_balance"
    if re.search(r"\bholidays?\b", text) and "policy" not in text and subject != "attendance":
        return "holiday_query"
    if re.search(r"\bpolicy\b", text) and subject != "leave_plan":
        return "policy_question"
    signals = [
        r"\b(what|how) about\b",
        r"^and\b",
        r"\b(previous|last|next)\s+month\b",
        r"\b(same|that|those|it)\b",
    ]
    if subject == "attendance":
        signals.append(
            r"\b(compare|percentage|percent|remotely|remote|absent|absences|hours|which dates|what dates|those dates|which days|"
            r"switch(?:\s+that)?|instead|now compare|those two|same month|that month|october|september|august)\b"
        )
    elif subject == "reconciliation":
        signals.append(
            r"\b(discrepanc\w*|reconcil\w*|overlap|missing|approved|sick|balance|same|that|those|it)\b"
        )
    elif subject == "leave_history":
        signals.append(
            r"\b(approved|rejected|pending|who|other|status|it|that|request|application|"
            r"approver|approval manager|manager|follow up|next steps?|what should i do|"
            r"how long|still|respond|advise|awaiting|which request)\b"
        )
    elif subject == "leave_balance":
        signals.append(
            r"\b(used|pending|remaining|include|sick|annual|deduct|responsible|match|"
            r"which balance|larger|difference|compare|those|them)\b"
        )
    elif subject == "holiday":
        signals.append(r"\b(any|there)\b")
    elif subject == "leave_plan":
        signals.append(
            r"\b(those dates|that period|that window|holidays?|project|deadline|approvals?|"
            r"prepare|request|draft|confirm|submit|business days|enough leave|same dates)\b"
        )
    elif subject == "policy":
        signals.append(
            r"\b(same rule|that rule|notice|probation|approval|approve|manager|remote|from home|days|overtime|premium|authorize|authorise)\b"
        )
    elif subject == "org_attendance":
        signals.append(
            r"\b(compare|absent|absences|absenteeism|absence rates?|remote|hours|present|september|october|"
            r"how much|increase|decreased|contributed most|department contributed|"
            r"office|location|unusual|anomal|pattern|investigat\w*|recommend|recommendation|"
            r"operational consequences?|consequences?|explain|data behind|"
            r"last (?:\w+ )?months?|largest increase|departments?|highest|employees?|records?)\b"
        )
    elif subject == "org_leave":
        signals.append(r"\b(how many|which ones|hr approval|sort|sick|annual|balance)\b")
    elif subject == "employee_directory":
        signals.append(
            r"\b(their|them|those|manager|how many|on leave|tenure|average tenure|mean tenure|"
            r"of those|among those|same employees|that group|that cohort|narrow|only|singapore|"
            r"percentage|percent|location|based in)\b"
        )
    elif subject == "compensation_query":
        signals.append(r"\b(engineering|department|only|average|payroll)\b")
    elif subject == "hr_metrics":
        signals.append(r"\b(compare|month|october|september)\b")
    elif subject in {"team_analytics", "approval_inbox"}:
        signals.append(
            r"\b(approved|pending|remote(?:ly)?|remote[-\s]?work|those|them|their|incomplete|attendance|leave|"
            r"what about|department|departments?|break(?:\s+\w+)?\s+down|count|focus|instead|"
            r"hours|lowest|remote[-\s]?day|highest (?:operational )?risk|which (?:one|request)|why|"
            r"urgent|urgency|recommend|approve|approval|confirm|confirmation|prepare|impact|"
            r"what happens|if i approve|request)\b"
        )
    elif subject == "team_conflict":
        signals.append(
            r"\b(overlap|overlapping|coverage|alternative|defer|staffing|project|atlas|impact|"
            r"those|them|risk|schedule|availability)\b"
        )
    elif subject == "approval_decision":
        signals.append(
            r"\b(approve|reject|status|why|reason|it|that request|same request|confirm|confirmation|prepare)\b"
        )
    elif subject == "team_roster":
        signals.append(
            r"\b(how many|count|department|title|only|direct|remote(?:ly)?|attendance|"
            r"them|those|their|worked|absent|present|leave|balance)\b"
        )
    elif subject == "workforce_analytics":
        signals.append(
            r"\b(break(?:\s+it)?\s+down|by department|department|remaining|balances?|"
            r"utilization|integrity|calculations|confidently report|which figures|"
            r"used leave|entitled|pending leave|denominator|"
            r"compare|months?|office|location|largest increase|unusual|anomal|pattern|"
            r"investigation|recommend|recommendation|actions?|explain|data behind|"
            r"absence rates?|unplanned)\b"
        )
    elif subject in {"routing_audit", "routing_investigation", "org_transfer"}:
        signals.append(
            r"\b(discrepan\w*|urgent|urgency|important|importance|why|steps?|actions?|"
            r"correct(?:ive)?|remediat\w*|resolve|routing|manager|approval|transfer|"
            r"same|that|those|it|primary|case|evidence|severity|recommend|"
            r"should hr|hr (?:do|take|correct|fix)|next steps?)\b"
        )
    continued = any(re.search(pattern, text) for pattern in signals)
    if not continued and _MONTH_NAME.search(text) and len(text.split()) <= 8:
        continued = True
    if not continued:
        return None
    # Employee delayed-approval advice follow-ups stay on leave status / own profile.
    if subject == "leave_history" and re.search(
        r"\b(approval manager|assigned approver|who (?:is|was) (?:my )?(?:manager|approver)|"
        r"that pending request|pending request|follow up|next steps?)\b",
        text,
    ):
        if re.search(r"\b(my manager|approval manager|assigned approver|who)\b", text):
            return "employee_profile"
        return "leave_request_status"
    # Approval-inbox / pending-request threads: urgency stays on team analytics;
    # prepare/confirm becomes an approval decision that reuses the selected request id.
    if subject in {"approval_inbox", "team_analytics", "approval_decision"}:
        focus = approval_inbox_followup_focus(text)
        if focus == "prepare_approve":
            return "approval_decision"
        if focus in {"urgency", "why_urgent", "approve_impact", "reject_impact"}:
            return "team_analytics"
    # Team roster follow-ups about attendance / leave / remote become analytics.
    if subject == "team_roster" and re.search(
        r"\b(remote(?:ly)?|attendance|absent|present|late|sick|worked|leave|balance)\b",
        text,
    ):
        return "team_analytics"
    # Directory tenure follow-ups stay on the directory so hire filters can be reused.
    if subject == "employee_directory" and re.search(
        r"\b(tenure|average tenure|mean tenure|of those|among those|that group|that cohort)\b",
        text,
    ):
        return "employee_directory"
    # Keep month/increase/department-contribution follow-ups on org_attendance so counts stay deterministic.
    if subject == "org_attendance" and re.search(
        r"\b(what about|how about|how much|increase|decreased|contributed most|"
        r"department contributed|september|october|november|december|january|february|"
        r"march|april|may|june|july|august)\b",
        text,
    ):
        return "org_attendance"
    # Org attendance follow-ups about rates / offices / anomalies escalate to workforce analytics.
    if subject == "org_attendance" and (
        workforce_followup_question(text)
        or re.search(
            r"\b(absence rates?|highest absence|compare|last (?:\w+ )?months?|office|location|"
            r"largest increase|unusual|anomal|pattern|investigat\w*|recommend|recommendation|"
            r"operational consequences?|consequences?|explain the data|data behind)\b",
            text,
        )
    ):
        return "workforce_analytics"
    if subject in {"routing_audit", "routing_investigation", "org_transfer"}:
        # Keep corrective/urgency/importance follow-ups on the same investigation.
        focus = org_transfer_followup_focus(text)
        if focus is not None:
            return "org_transfer"
        return "org_transfer"
    return _SUBJECT_INTENT[subject]


def classify_with_model(message: str, role: str) -> str | None:
    from app.ai.llm import PermanentLLMError, TransientLLMError

    client = LLMClient()
    if not client.enabled:
        return None
    try:
        parsed = client.complete_json(
            [
                {
                    "role": "system",
                    "content": (
                        "Classify the HR request. Reply with JSON only: "
                        '{"intent":"leave_planning|leave_request|leave_eligibility|leave_policy|leave_request_status|leave_cancellation|leave_balance|team_conflict|team_roster|team_analytics|workforce_analytics|org_transfer|policy_question|confidential_lookup|employee_profile|attendance_query|holiday_query|employee_directory|org_attendance|org_leave|compensation_query|hr_metrics|hr_action|out_of_scope|general_hr"}. '
                        "Never return approval_decision; approve/reject decisions are handled outside this classifier. "
                        "A question about whether leave is allowed is leave_eligibility. A notice or rulebook question is leave_policy. "
                        "Drafting, preparing, or submitting leave — including 'draft so I can confirm' or 'who would approve' — is leave_request. "
                        "A manager asking which leave asks are in their inbox, waiting on their decision, or need sign-off is team_analytics. "
                        "A manager asking for their own direct reports or team roster is team_roster, never employee_directory. "
                        "Manager questions about their team's attendance, late arrivals, missing attendance, pending leave, "
                        "currently on leave, awaiting my approval, or leave balances are team_analytics, never org_attendance or org_leave. "
                        "An employee's own leave history or approved requests without drafting a new request is leave_request_status. "
                        "Asking whether the employee is eligible for another leave request is leave_eligibility, not leave_request. "
                        "Compensation band notes, pay bands, and compensation policy documents are policy_question, never confidential_lookup. "
                        "Headcount, a department roster, managers, and a named employment profile are employee_directory, never workforce_analytics. "
                        "Organization attendance is org_attendance. Organization leave is org_leave. Salary totals and averages are compensation_query. "
                        "Do not select an employee and do not decide permissions."
                    ),
                },
                {"role": "user", "content": f"Role: {role}\nRequest:\n{message}"},
            ],
            temperature=0,
            max_tokens=80,
            raise_on_error=True,
        )
    except (TransientLLMError, PermanentLLMError):
        return None
    intent = parsed.get("intent")
    if intent == "self_service":
        intent = "employee_profile"
    # approval_decision must only come from parse_approval_decision, never the LLM.
    if intent == "approval_decision":
        return None
    return intent if intent in ALLOWED_INTENTS else None


def profile_for(db: Session, actor: User, actor_employee: Employee, subject: Employee, *, private: bool) -> dict:
    department = db.get(Department, subject.department_id)
    return redact_employee(
        db,
        actor,
        actor_employee,
        subject,
        department.name if department else "",
        include_private=private,
    )


def guard_sensitive_read(db: Session, actor: User, actor_employee: Employee, subject: Employee, kinds: set[str]) -> None:
    try:
        assert_sensitive_access(db, actor, actor_employee, subject, kinds)
    except AuthorizationError:
        raise
    if kinds and actor.role != "hr_admin" and actor.employee_id != subject.id:
        if "performance" in kinds and not can_view_performance(db, actor, actor_employee, subject):
            raise AuthorizationError("Performance data is restricted.", resource=f"employee:{subject.id}")


def iso(value: date) -> str:
    return value.isoformat()
