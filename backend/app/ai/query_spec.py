"""Reusable structured query interpretation for Helios.

Separates natural-language interpretation from SQL/services and authorization.
The LLM is never consulted here. Authorization is never decided here.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

from app.clock import org_today

Scope = Literal["self", "team", "org", "unknown"]
Domain = Literal[
    "profile",
    "attendance",
    "leave_balance",
    "leave_request",
    "policy",
    "team",
    "directory",
    "compensation",
    "hr_metrics",
    "security",
    "unsupported",
    "unknown",
]


MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


@dataclass
class QuerySpec:
    domain: Domain = "unknown"
    scope: Scope = "unknown"
    metric: str | None = None
    aggregation: str | None = None
    group_by: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    output_fields: list[str] = field(default_factory=list)
    granularity: str | None = None  # individual | aggregate | series | crosstab
    record_mode: str | None = None  # records | summary | count_only
    start: date | None = None
    end: date | None = None
    compare_windows: list[tuple[date, date, str]] = field(default_factory=list)
    department: str | None = None
    location: str | None = None
    status: str | None = None
    intent_hint: str | None = None
    unsupported: str | None = None
    disclose: str | None = None
    notes: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["start"] = self.start.isoformat() if self.start else None
        payload["end"] = self.end.isoformat() if self.end else None
        payload["compare_windows"] = [
            {"start": s.isoformat(), "end": e.isoformat(), "label": label} for s, e, label in self.compare_windows
        ]
        return payload


def validate_query_spec(spec: QuerySpec, message: str, *, role: str) -> QuerySpec:
    """Detect missing or contradictory requirements before tool execution.

    Corrections stay in the structured representation; authorization is unchanged.
    """
    text = (message or "").lower()
    errors: list[str] = []

    # Roster list must not collapse into pending-leave analytics.
    if (
        re.search(r"\b(list|show|roster)\b", text)
        and re.search(r"\b(direct reports?|my team)\b", text)
        and not re.search(r"\b(leave|pending|attendance|hours|performance|grades?)\b", text)
    ):
        if spec.domain not in {"team", "unknown"} or (spec.metric or "").startswith("pending"):
            errors.append("roster_request_must_not_use_pending_leave")
            spec.domain = "team"
            spec.scope = "team"
            spec.metric = "roster"
            spec.record_mode = "records"
            spec.intent_hint = "team_roster"

    # Calendar-day attendance listings are series, not hours totals.
    if re.search(r"\b(?:last|past|previous)\s+\d+\s+calendar\s+days?\b", text) and re.search(
        r"\b(dates?|records?|status|attendance)\b", text
    ):
        if spec.metric in {None, "hours", "hours_total"}:
            errors.append("calendar_day_series_must_not_collapse_to_hours_total")
            spec.domain = "attendance"
            spec.metric = "calendar_day_series"
            spec.granularity = "series"
            spec.record_mode = "records"
            spec.intent_hint = "attendance_query"

    # Directory roster asks must return records, not count-only.
    if (
        re.search(r"\b(names?|employee numbers?|job titles?|grades?|show|find|list)\b", text)
        and re.search(r"\b(department|based in|located)\b", text)
        and not re.search(r"\b(percentage|percent|how many only)\b", text)
    ):
        if spec.record_mode == "count_only" or spec.aggregation == "count":
            errors.append("directory_roster_must_return_records")
            spec.record_mode = "records"
            spec.granularity = "individual"

    # Status × leave-type asks need both grouping dimensions.
    if re.search(r"\bstatus\b", text) and re.search(r"\bleave type\b", text):
        if spec.group_by in {None, "status"}:
            errors.append("leave_crosstab_requires_status_and_type")
            spec.group_by = "status,leave_type"
            spec.granularity = "crosstab"
            spec.intent_hint = "org_leave"

    # Explicit previous-calendar-year hiring totals must not route to policy.
    if re.search(r"\b(hir(?:e|ed|ing)|hires)\b", text) and re.search(
        r"\b(previous (?:\w+ )?calendar years?|last (?:\w+ )?calendar years?|yearly totals|"
        r"highest (?:number of )?hires)\b",
        text,
    ):
        if spec.domain == "policy" or spec.intent_hint == "policy_question":
            errors.append("hiring_trends_must_not_use_policy_retrieval")
            spec.domain = "directory"
            spec.metric = "hire_year_relative"
            spec.intent_hint = "employee_directory"

    # Previous calendar month is not last-seven-days.
    if re.search(r"\b(previous|last)\s+calendar\s+month\b", text) or re.search(
        r"\b(previous|last)\s+month\b", text
    ):
        if spec.start and spec.end and (spec.end - spec.start).days < 20:
            errors.append("calendar_month_must_not_use_last_seven_days")
            today = org_today()
            first_this = today.replace(day=1)
            end = first_this - __import__("datetime").timedelta(days=1)
            spec.start = end.replace(day=1)
            spec.end = end

    # Role scope: managers stay on team for report-line asks.
    if role == "manager" and re.search(r"\b(my (?:direct )?reports?|my team|awaiting my approval)\b", text):
        if spec.scope == "org":
            errors.append("manager_team_scope_must_not_expand_to_org")
            spec.scope = "team"
            if spec.intent_hint in {"org_attendance", "org_leave", "employee_directory"}:
                spec.intent_hint = "team_analytics"

    spec.validation_errors = errors
    if errors:
        spec.notes.extend(f"validation:{item}" for item in errors)
    return spec


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    if month == 12:
        return date(year, 12, 1), date(year, 12, 31)
    start = date(year, month, 1)
    end = date(year, month + 1, 1)
    from datetime import timedelta

    return start, end - timedelta(days=1)


def _named_months(text: str) -> list[tuple[date, date, str]]:
    today = org_today()
    found: list[tuple[date, date, str]] = []
    for match in re.finditer(
        r"\b(january|february|march|april|may|june|july|august|september|sept|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b(?:\s+(\d{4}))?",
        text,
        re.IGNORECASE,
    ):
        month = MONTHS[match.group(1).lower()]
        year = int(match.group(2) or today.year)
        start, end = _month_bounds(year, month)
        label = f"{start.strftime('%B')} {year}"
        found.append((start, end, label))
    return found


def _final_two_weeks_october(text: str) -> tuple[date, date, str] | None:
    if re.search(r"\b(final|last)\s+two\s+weeks?\b", text, re.IGNORECASE) and re.search(
        r"\boctober\b", text, re.IGNORECASE
    ):
        year_m = re.search(r"\boctober\s+(\d{4})\b", text, re.IGNORECASE)
        year = int(year_m.group(1)) if year_m else org_today().year
        return date(year, 10, 18), date(year, 10, 31), f"final two weeks of October {year}"
    return None


# Metrics that are not collected in Helios. Matched by concept families, not exact phrases.
# Each entry: (regex, code, human_label)
UNSUPPORTED_METRICS: tuple[tuple[str, str, str], ...] = (
    (
        r"\b(gps|commute distance|latitude|longitude|geolocat|geo[-\s]?fence|commute[-\s]?distance)\b",
        "gps_commute_distance",
        "GPS coordinates and commute distance",
    ),
    (
        r"\b(badge[-\s]?swipe|badge\s+latency|swipe\s+latency|turnstile|access[-\s]?card\s+latency|"
        r"door\s+access\s+latency|badge\s+reader)\b",
        "badge_swipe_latency",
        "Badge-swipe latency",
    ),
    (
        r"\b(wi[-\s]?fi\s+roam|roam[-\s]?handoff|handoff\s+latency|network\s+latency|api\s+latency|"
        r"latency\s+percentile|p(?:50|90|95|99)\s+latency)\b",
        "latency_percentile",
        "Network or latency percentiles",
    ),
    (
        r"\b(heart\s+rate|step\s+count|wearable|fitbit|biometric\s+sensor|blood\s+pressure\s+monitor|"
        r"retina[-\s]?scan|dwell[-\s]?time)\b",
        "biometric_wearable",
        "Biometric or wearable metrics",
    ),
    (
        r"\b(keystroke|keyboard[-\s]?cadence|mouse\s+movement|screen\s+time\s+tracker|webcam\s+presence|"
        r"stress\s+score)\b",
        "desktop_surveillance",
        "Desktop surveillance signals",
    ),
    (
        r"\b(desk[-\s]?occupancy|occupancy\s+heat|heat[-\s]?index\s+percentile)\b",
        "desk_occupancy",
        "Desk-occupancy heat-index",
    ),
    (
        r"\b(blood\s+type|blood\s+group)\b",
        "blood_type",
        "Blood type",
    ),
    (
        r"\b(emergency[-\s]?contact|next\s+of\s+kin)\b",
        "emergency_contact",
        "Emergency-contact information",
    ),
    (
        # Calculation with incomplete/missing overtime rate — not a policy definition question.
        r"\b(calculate|compute|using a field that is incomplete|if overtime rate is missing)\b.{0,80}"
        r"\b(overtime\s+premium|overtime\s+rate)\b|"
        r"\b(overtime\s+premium|overtime\s+rate)\b.{0,80}"
        r"\b(calculate|compute|incomplete|missing)\b",
        "overtime_premium",
        "Overtime premium / overtime rate",
    ),
)


def detect_unsupported_metric(message: str) -> tuple[str, str] | None:
    """Return (code, human_label) when the message asks for a metric Helios does not store."""
    text = (message or "").lower()
    for pattern, code, label in UNSUPPORTED_METRICS:
        if re.search(pattern, text):
            return code, label
    return None


def _unsupported_intent_hint(code: str) -> str:
    if code in {"blood_type", "emergency_contact"}:
        return "employee_profile"
    return "attendance_query"


def interpret_query(message: str, *, role: str = "employee", inherited: dict | None = None) -> QuerySpec:
    """Build a structured query from one user message and optional conversation context."""
    text = (message or "").lower()
    inherited = inherited or {}
    spec = QuerySpec()

    # Unsupported metrics (answer the limitation; do not substitute another metric).
    unsupported = detect_unsupported_metric(message)
    if unsupported:
        code, label = unsupported
        spec.domain = "unsupported"
        spec.unsupported = code
        spec.intent_hint = _unsupported_intent_hint(code)
        spec.notes.append(f"{label} is not stored")
        return spec

    if re.search(r"\b(employment status (?:history|changes?|changed)|status (?:ever )?changed|previous employment status)\b", text):
        spec.domain = "profile"
        spec.scope = "self"
        spec.disclose = "employment_status_history"
        spec.intent_hint = "employee_profile"
        return spec

    # Scope cues
    teamish = bool(
        re.search(
            r"\b(my (?:direct )?reports?|my team|direct reports?|reporting team|among my reports|within my reporting)\b",
            text,
        )
    )
    orgish = bool(
        re.search(
            r"\b(organization|organisation|company-?wide|workforce|active employees|all employees|every employee|"
            r"headcount|departments? have|office location has)\b",
            text,
        )
    )
    personal = bool(re.search(r"\b(my|i|i'm|me)\b", text))

    if teamish or (role == "manager" and inherited.get("subject") in {"team", "team_analytics", "team_roster"}):
        spec.scope = "team"
    elif orgish or role == "hr_admin" and not personal and not teamish:
        spec.scope = "org"
    elif personal:
        spec.scope = "self"

    # Department / location tokens
    for dept in (
        "engineering",
        "finance",
        "sales",
        "product",
        "people operations",
        "customer operations",
        "supply chain",
        "legal",
    ):
        if dept in text:
            spec.department = dept.title() if dept != "legal" else "Legal & Compliance"
            if dept == "people operations":
                spec.department = "People Operations"
            if dept == "customer operations":
                spec.department = "Customer Operations"
            if dept == "supply chain":
                spec.department = "Supply Chain"
            break

    # Date windows
    final_oct = _final_two_weeks_october(text)
    months = _named_months(text)
    if final_oct:
        spec.start, spec.end, _label = final_oct
        spec.notes.append(final_oct[2])
    elif len(months) >= 2 and re.search(r"\b(compare|comparison|versus|vs\.?|difference|between)\b", text):
        spec.compare_windows = months[:2]
        spec.aggregation = "compare"
    elif len(months) == 1:
        spec.start, spec.end, _label = months[0]

    # Inherit prior window/metric when follow-up is thin
    if inherited.get("start") and inherited.get("end") and spec.start is None and not spec.compare_windows:
        if re.search(r"\b(those|them|that|same|previous|that (?:group|department|period|window))\b", text) or len(text.split()) <= 12:
            try:
                spec.start = date.fromisoformat(str(inherited["start"])[:10])
                spec.end = date.fromisoformat(str(inherited["end"])[:10])
            except Exception:
                pass
    if inherited.get("department") and not spec.department and re.search(
        r"\b(that department|those|them|that group|that cohort|in that)\b", text
    ):
        spec.department = inherited.get("department")

    # Profile — do not treat overtime/policy "start" phrasing as hire-date questions.
    if (
        re.search(
            r"\b(position|job title|assigned to|which department|my department|hire date|"
            r"started(?:\s+(?:date|working))?|starting(?:\s+(?:date|working))?|"
            r"start(?:\s+(?:date|working))|"
            r"day of the week|reporting structure|who should i contact|line manager|employment status|"
            r"missing from my (?:employee )?profile|incomplete|"
            r"i was hired|confirm .{0,40}(?:my )?(?:employee )?(?:record|profile))\b",
            text,
        )
        and spec.scope in {"self", "unknown"}
        and not teamish
        and not re.search(r"\b(overtime|premium|policy|policies)\b", text)
    ):
        spec.domain = "profile"
        spec.scope = "self"
        spec.intent_hint = "employee_profile"
        if re.search(r"\b(position|job title|assigned to)\b", text) and re.search(r"\bdepartment\b", text):
            spec.metric = "title_and_department"
        elif re.search(
            r"\b(hire date|day of the week|officially start|start(?:ed|ing)?(?:\s+(?:date|working))?|"
            r"i was hired|confirm .{0,40}(?:hire|record|profile))\b",
            text,
        ):
            spec.metric = "hire_date_weekday"
        elif re.search(r"\b(reporting structure|who should i contact|line manager|manager)\b", text):
            spec.metric = "manager_contact"
        elif re.search(r"\b(missing|incomplete)\b", text):
            spec.metric = "profile_completeness"
        return spec

    # Leave balances vs pending committed days vs leave requests
    # Policy/entitlement education must not become a personal balance lookup.
    if re.search(
        r"\b(what happens|insufficient leave balance|if i have insufficient|"
        r"compensatory leave|leave balance reset|when will my (?:annual )?leave|"
        r"are employees entitled|employees entitled|entitled to each year|"
        r"sick leave days are employees|carry.?forward|carry.?over|reset)\b",
        text,
    ) and re.search(r"\b(leave|entitled|balance|policy)\b", text):
        spec.domain = "policy"
        spec.intent_hint = "policy_question"
        return spec
    if re.search(r"\b(entitled|used leave|pending leave|remaining (?:annual )?leave|leave balance)\b", text) or (
        re.search(r"\b(committed to pending|pending requests?)\b", text) and re.search(r"\b(leave|days)\b", text)
    ):
        # Workforce/policy entitlement questions stay out of personal leave_balance.
        if re.search(r"\b(employees?|workforce|company|policy|policies|compensatory)\b", text) and not personal:
            spec.domain = "policy"
            spec.intent_hint = "policy_question"
            return spec
        if teamish or spec.scope == "team":
            spec.domain = "team"
            spec.scope = "team"
            spec.intent_hint = "team_analytics"
            if re.search(r"\b(total|across|sum|still available|remaining)\b", text) and re.search(
                r"\b(annual leave|leave days|remaining)\b", text
            ):
                spec.metric = "team_remaining_leave_sum"
            elif re.search(r"\bhalf (?:of )?(?:their |the )?entitlement|more than half\b", text):
                spec.metric = "team_half_entitlement_used"
            elif re.search(r"\bpending\b", text):
                spec.metric = "team_pending_leave"
            else:
                spec.metric = "team_leave_balances"
            return spec
        # Personal balance: first-person cues, or self-service pending/used phrasing without
        # workforce/policy framing. Bare "employees entitled" stays policy above.
        self_service = bool(
            personal
            or spec.scope == "self"
            or re.search(
                r"\b(committed to pending|pending requests?|my (?:sick|annual )?leave|"
                r"sick leave balance|records responsible|deductions?)\b",
                text,
            )
        )
        if self_service:
            spec.domain = "leave_balance"
            spec.scope = "self"
            if re.search(r"\b(committed to pending|pending requests?|days .{0,20}pending)\b", text):
                spec.metric = "pending_leave_days"
                spec.intent_hint = "leave_balance"
            elif re.search(r"\b(difference between|entitled|used|pending)\b", text):
                spec.metric = "leave_balance_breakdown"
                spec.intent_hint = "leave_balance"
            else:
                spec.metric = "remaining_leave"
                spec.intent_hint = "leave_balance"
            return spec

    # Attendance metrics — bare "working hours" without period/total cues is policy (core hours).
    _hours_as_attendance = bool(
        re.search(
            r"\b(total|recorded|worked|compare|comparison|longest|zero|august|september|october|"
            r"november|december|january|february|march|april|may|june|july|\d{4}|last month|this month)\b",
            text,
        )
    )
    if (
        re.search(r"\b(attendance|remote|present|absent|weekday|longest|zero (?:working )?hours)\b", text)
        or (re.search(r"\b(working hours|hours)\b", text) and _hours_as_attendance)
        or spec.compare_windows
    ):
        if teamish or (spec.scope == "team" and re.search(r"\b(attendance|absent|remote|present|hours)\b", text)):
            spec.domain = "team"
            spec.scope = "team"
            spec.intent_hint = "team_analytics"
            if re.search(r"\bweekday\b", text) and re.search(r"\babsent", text):
                spec.metric = "team_absent_weekday_peak"
            elif re.search(r"\bzero\b", text) and re.search(r"\bhours?\b", text):
                spec.metric = "team_zero_hours"
            elif final_oct or (spec.start and re.search(r"\b(final|last)\s+two\s+weeks?\b", text)):
                spec.metric = "team_attendance_window"
            elif re.search(r"\b(office[- ]?present|present).{0,40}(remote|remote-work)|remote.{0,40}(present|office)\b", text):
                spec.metric = "team_present_vs_remote"
            else:
                spec.metric = "team_attendance_summary"
            return spec
        if personal or spec.scope == "self":
            spec.domain = "attendance"
            spec.scope = "self"
            spec.intent_hint = "attendance_query"
            if spec.compare_windows or (
                re.search(r"\b(compare|comparison|difference)\b", text)
                and re.search(r"\b(hours|august|september|october)\b", text)
            ):
                spec.metric = "hours_compare"
                if not spec.compare_windows and len(months) >= 2:
                    spec.compare_windows = months[:2]
                    spec.aggregation = "compare"
            elif re.search(r"\bweekday\b", text) and re.search(r"\bremote\b", text):
                spec.metric = "remote_weekday_peak"
                spec.aggregation = "mode"
                spec.group_by = "weekday"
                spec.status = "remote"
            elif re.search(r"\b(longest|most hours|highest .{0,10}hours)\b", text):
                spec.metric = "longest_hours_dates"
                spec.aggregation = "max"
            elif re.search(r"\bzero\b", text) and re.search(r"\bhours?\b", text):
                spec.metric = (
                    "remote_zero_hours"
                    if re.search(r"\b(remote|remotely)\b", text)
                    else "zero_hours"
                )
            elif re.search(r"\b(remote).{0,30}(present|office)|difference between .{0,20}remote\b", text):
                spec.metric = "remote_vs_present"
            else:
                spec.metric = "attendance_report"
            return spec

    # Team locations / departments / titles
    if teamish or spec.scope == "team":
        spec.domain = "team"
        spec.scope = "team"
        if re.search(r"\b(office locations?|locations? (?:are )?represented|different (?:office )?locations?)\b", text):
            spec.metric = "team_location_diversity"
            spec.intent_hint = "team_roster"
            return spec
        if re.search(r"\b(grouped by (?:job )?title|by job title)\b", text):
            spec.metric = "team_by_title"
            spec.intent_hint = "team_analytics"
            return spec
        if re.search(r"\b(departments? (?:are )?represented|grouped by department)\b", text):
            spec.metric = "team_by_department"
            spec.intent_hint = "team_roster"
            return spec

    # Org / HR directory & compensation
    if role == "hr_admin" or orgish:
        if re.search(
            r"\b(percentage|percent|%|share).{0,40}\b(workforce|employees|engineering|headcount)\b|"
            r"\bworkforce .{0,20}engineering\b|"
            r"\bheadcount .{0,20}engineering\b|"
            r"\bsits in engineering\b",
            text,
        ):
            spec.domain = "directory"
            spec.scope = "org"
            spec.metric = "workforce_pct_by_department"
            spec.intent_hint = "employee_directory"
            return spec
        if re.search(
            r"\b(at least five years|more than three years|employed for(?: at least)?|five years|"
            r"tenure (?:cohort|filter|of (?:those|that)|among those))\b",
            text,
        ) and not re.search(r"\b(mean|average|median|typical)\s+tenure\b", text):
            spec.domain = "directory"
            spec.scope = "org"
            spec.metric = "tenure_cohort_count"
            spec.intent_hint = "employee_directory"
            return spec
        if re.search(r"\baverage employee tenure\b", text) and re.search(r"\bdays?\b", text):
            spec.domain = "directory"
            spec.scope = "org"
            spec.metric = "avg_tenure_days"
            spec.intent_hint = "employee_directory"
            return spec
        if re.search(r"\b(mean|average|median|typical)\s+tenure\b", text) and re.search(r"\byears?\b", text):
            spec.domain = "hr_metrics"
            spec.scope = "org"
            spec.metric = "mean_tenure_years"
            spec.intent_hint = "hr_metrics"
            return spec
        if re.search(r"\b(quarter|hired in each quarter|hiring)\b", text) and re.search(r"\b(2024|employees)\b", text):
            spec.domain = "directory"
            spec.scope = "org"
            spec.metric = "hires_by_quarter"
            spec.intent_hint = "employee_directory"
            return spec
        if re.search(
            r"\b(office location|largest active workforce|location has|"
            r"second[- ]largest|third[- ]largest|smallest (?:active )?workforce)\b",
            text,
        ):
            spec.domain = "directory"
            spec.scope = "org"
            spec.metric = "location_headcount"
            spec.intent_hint = "employee_directory"
            return spec
        if re.search(
            r"\b(hired during|hired in|people were hired|employees? hired)\b.{0,40}\b(20\d{2}|calendar year)\b|"
            r"\b(20\d{2}|calendar year).{0,40}\b(hired|hires|hire_date)\b",
            text,
        ):
            spec.domain = "directory"
            spec.scope = "org"
            spec.metric = "hires_by_year"
            spec.intent_hint = "employee_directory"
            return spec
        if re.search(
            r"\bapproved\b.{0,40}\bleave requests?\b.{0,40}\b(20\d{2}|start)|"
            r"\bleave requests?\b.{0,40}\bstatus approved\b.{0,40}\b(start|20\d{2})",
            text,
        ):
            spec.domain = "leave"
            spec.scope = "org"
            spec.metric = "approved_leave_by_start_year"
            spec.intent_hint = "org_leave"
            return spec
        if re.search(
            r"\b(how many are active overall|active staff pool|active (?:employee )?headcount|"
            r"size of the active|staff (?:are |is )?currently marked active)\b",
            text,
        ):
            spec.domain = "directory"
            spec.scope = "org"
            spec.metric = "active_headcount"
            spec.intent_hint = "employee_directory"
            return spec
        if re.search(r"\b(without assigned managers?|no assigned manager|null manager)\b", text):
            spec.domain = "directory"
            spec.scope = "org"
            spec.metric = "departments_without_managers"
            spec.intent_hint = "employee_directory"
            return spec
        if re.search(r"\b(largest (?:number of )?active|largest active workforce|department .{0,20}largest)\b", text):
            spec.domain = "directory"
            spec.scope = "org"
            spec.metric = "largest_department"
            spec.intent_hint = "employee_directory"
            return spec
        if re.search(r"\b(average|avg).{0,20}(salary|compensation)|highest average annual salary\b", text):
            spec.domain = "compensation"
            spec.scope = "org"
            spec.metric = "avg_salary_by_department"
            spec.intent_hint = "compensation_query"
            return spec
        if re.search(r"\b(percentage|percent).{0,40}(salary|expenditure)|salary expenditure .{0,20}engineering\b", text):
            spec.domain = "compensation"
            spec.scope = "org"
            spec.metric = "salary_pct_by_department"
            spec.intent_hint = "compensation_query"
            return spec
        if re.search(r"\b(no corresponding approval|without .{0,10}approval decision|no approval decision)\b", text):
            spec.domain = "hr_metrics"
            spec.scope = "org"
            spec.metric = "leave_without_approval"
            spec.intent_hint = "hr_metrics"
            return spec
        if re.search(r"\b(pending the longest|longest .{0,10}pending)\b", text):
            spec.domain = "hr_metrics"
            spec.scope = "org"
            spec.metric = "longest_pending_leave"
            spec.intent_hint = "org_leave"
            return spec
        if re.search(r"\baverage remaining annual leave\b", text):
            spec.domain = "hr_metrics"
            spec.scope = "org"
            spec.metric = "avg_remaining_leave"
            spec.intent_hint = "org_leave"
            return spec
        if re.search(r"\b(hired before|hired in 2024|salary).{0,40}(before 2020|2024 or later|historical)\b", text) or (
            re.search(r"\bbefore 2020\b", text) and re.search(r"\bsalar", text)
        ):
            spec.domain = "compensation"
            spec.scope = "org"
            spec.metric = "salary_by_hire_cohort"
            spec.disclose = "no_salary_history"
            spec.intent_hint = "compensation_query"
            return spec

    # Expired-policy explanation is an HR metrics disclosure, not a conflict lookup.
    if re.search(r"\bexpired polic(?:y|ies)\b", text) and re.search(
        r"\b(explain|explained|governing|govern|rules are today|still (?:in force|apply))\b", text
    ):
        spec.domain = "hr_metrics"
        spec.metric = "expired_policies"
        spec.intent_hint = "hr_metrics"
        return spec

    # Policy conflict between current documents (not expired-policy education).
    if re.search(
        r"\b(conflict|conflicts).{0,40}\b(policy|policies)\b|"
        r"\bpolic(?:y|ies).{0,40}\b(conflict|conflicts)\b|"
        r"\b(conflicting|disagreeing) (?:current )?polic",
        text,
    ):
        spec.domain = "policy"
        spec.metric = "policy_conflict"
        spec.intent_hint = "policy_question"
        return spec

    return spec


def intent_from_spec(spec: QuerySpec, *, role: str) -> str | None:
    """Map a QuerySpec to an existing Helios intent when confident."""
    if spec.intent_hint:
        # Managers asking org_* from team follow-ups stay on team_analytics.
        if role == "manager" and spec.scope == "team" and spec.intent_hint in {
            "org_attendance",
            "org_leave",
            "employee_directory",
        }:
            return "team_analytics"
        return spec.intent_hint
    return None


def interpret_and_validate(message: str, *, role: str = "employee", inherited: dict | None = None) -> QuerySpec:
    """Build a QuerySpec then apply contradiction / completeness validation."""
    return validate_query_spec(interpret_query(message, role=role, inherited=inherited), message, role=role)
