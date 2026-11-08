"""Manager team-scoped attendance and leave reads.

Authorization must already have confirmed the actor is a manager or HR admin.
This module only queries employees whose manager_id matches the signed-in manager.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date

from sqlalchemy.orm import Session

from app.clock import org_today
from app.models import Attendance, Department, Employee, LeaveBalance, LeaveRequest, Project, ProjectAssignment
from app.services.leave_engine import team_members
from app.services.records import (
    _scheduled_working_days,
    attendance_window,
    parse_asked_window,
    parse_named_month_periods,
)

PAGE = 25
OPEN_LEAVE = ("pending_manager", "pending_hr", "approved")


_NUMBER_WORDS = {
    "zero": 0,
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


def _number(token: str) -> float | None:
    token = token.lower()
    if token in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[token])
    try:
        return float(token)
    except ValueError:
        return None


def _team_department(db: Session, message: str) -> Department | None:
    """A department named in the message, used only to narrow the manager's own reports."""
    text = message.lower()
    for row in db.query(Department).all():
        name = row.name.lower()
        if name in text:
            return row
        first = name.split()[0]
        if len(first) >= 5 and re.search(rf"\b{re.escape(first)}\b", text):
            return row
    return None


def _hours_for_window(db: Session, member_ids: list[int], start: date, end: date) -> dict[int, float]:
    rows = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id.in_(member_ids or [-1]),
            Attendance.work_date >= start,
            Attendance.work_date <= end,
        )
        .all()
    )
    totals: dict[int, float] = defaultdict(float)
    for row in rows:
        totals[row.employee_id] += float(row.hours or 0)
    return totals


def _team_hours_compare(
    db: Session,
    members: list[Employee],
    departments: dict[int, str],
    message: str,
    today: date,
) -> dict:
    text = message.lower()
    periods = parse_named_month_periods(message, today)
    # "previous two calendar months" → two full prior calendar months (not last-7-days).
    if len(periods) < 2 and re.search(r"\b(previous|last|past)\s+(two|2|\d+)\s+calendar\s+months?\b", text):
        try:
            start, end, _label = attendance_window(message, today, None, subject="attendance")
            # Expand the contiguous range into calendar months.
            periods = []
            cursor = date(start.year, start.month, 1)
            while cursor <= end:
                from calendar import monthrange

                month_end = date(cursor.year, cursor.month, monthrange(cursor.year, cursor.month)[1])
                periods.append((cursor, month_end, cursor.strftime("%B %Y")))
                if cursor.month == 12:
                    cursor = date(cursor.year + 1, 1, 1)
                else:
                    cursor = date(cursor.year, cursor.month + 1, 1)
        except Exception:
            periods = periods
    if len(periods) < 2:
        start, end, label = attendance_window(message, today, None, subject="attendance")
        periods = [(start, end, label)]
    periods = periods[:2]
    member_ids = [row.id for row in members]
    per_member: dict[int, dict] = {}
    comparison = []
    for start, end, label in periods:
        hours = _hours_for_window(db, member_ids, start, end)
        total = round(sum(hours.values()), 1)
        comparison.append(
            {
                "label": label,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "total_hours": total,
            }
        )
        for member in members:
            bucket = per_member.setdefault(
                member.id,
                {
                    **_person(member, departments),
                    "hours_by_period": {},
                    "total_hours": 0.0,
                },
            )
            value = round(float(hours.get(member.id, 0.0)), 1)
            bucket["hours_by_period"][label] = value
            bucket["total_hours"] = round(float(bucket["total_hours"]) + value, 1)
    people = sorted(per_member.values(), key=lambda item: (-float(item.get("total_hours") or 0), item.get("full_name") or ""))
    most = people[0] if people else None
    least = people[-1] if people else None
    return {
        "kind": "team_analytics",
        "focus": "hours_compare",
        "team_size": len(members),
        "comparison": comparison,
        "people": people,
        "people_count": len(people),
        "most": most,
        "least": least,
        "start": periods[0][0].isoformat() if periods else None,
        "end": periods[-1][1].isoformat() if periods else None,
        "label": " / ".join(item[2] for item in periods),
        "note": (
            "Per-employee recorded working hours for direct reports across the compared calendar months. "
            "Totals are sums of attendance.hours."
        ),
    }


def _team_hours_by_member(
    db: Session,
    members: list[Employee],
    departments: dict[int, str],
    message: str,
    today: date,
) -> dict:
    """Recorded hours for each direct report in the asked window."""
    start, end, label = attendance_window(message, today, None, subject="attendance")
    member_ids = [row.id for row in members]
    hours = _hours_for_window(db, member_ids, start, end)
    people = []
    for member in members:
        person = _person(member, departments)
        person["hours"] = round(hours.get(member.id, 0.0), 1)
        people.append(person)
    people.sort(key=lambda item: (float(item.get("hours") or 0), item.get("full_name") or ""))
    return {
        "kind": "team_analytics",
        "focus": "hours_by_member",
        "team_size": len(members),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": label,
        "people": people,
        "people_count": len(people),
        "note": (
            f"Recorded working hours for each direct report in {label}. "
            "Totals are sums of attendance.hours; employees outside your reporting line are not included."
        ),
    }


def _named_team_member(members: list[Employee], text: str) -> Employee | None:
    """Return a direct report whose first or full name appears in the message."""
    text = text.lower()
    for row in members:
        full = row.full_name.lower()
        first = (row.first_name or "").lower()
        last = (row.last_name or "").lower()
        if full and full in text:
            return row
        if first and re.search(rf"\b{re.escape(first)}\b", text) and (not last or last in text or len(first) >= 4):
            return row
    return None


def _member_hours_window(
    db: Session,
    member: Employee,
    departments: dict[int, str],
    message: str,
    today: date,
) -> dict:
    start, end, label = attendance_window(message, today, None, subject="attendance")
    hours = _hours_for_window(db, [member.id], start, end).get(member.id, 0.0)
    person = _person(member, departments)
    person["hours"] = round(hours, 1)
    return {
        "kind": "team_analytics",
        "focus": "member_hours",
        "team_size": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": label,
        "people": [person],
        "people_count": 1,
        "note": f"Recorded working hours for {member.full_name} in {label} (direct report only).",
    }


def _member_hours_compare(
    db: Session,
    member: Employee,
    departments: dict[int, str],
    message: str,
    today: date,
) -> dict:
    periods = parse_named_month_periods(message, today)
    if len(periods) < 2:
        start, end, label = attendance_window(message, today, None, subject="attendance")
        periods = [(start, end, label)]
    comparison = []
    for start, end, label in periods[:2]:
        hours = _hours_for_window(db, [member.id], start, end).get(member.id, 0.0)
        comparison.append(
            {
                "label": label,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "total_hours": round(hours, 1),
                "employee": member.full_name,
            }
        )
    person = _person(member, departments)
    return {
        "kind": "team_analytics",
        "focus": "member_hours_compare",
        "team_size": 1,
        "comparison": comparison,
        "people": [person],
        "people_count": 1,
        "note": f"Period working-hours comparison for direct report {member.full_name} only.",
    }


def _team_hours_average(
    db: Session,
    members: list[Employee],
    departments: dict[int, str],
    message: str,
    today: date,
) -> dict:
    start, end, label = attendance_window(message, today, None, subject="attendance")
    hours = _hours_for_window(db, [row.id for row in members], start, end)
    values = [round(hours.get(row.id, 0.0), 1) for row in members]
    average = round(sum(values) / len(values), 1) if values else 0.0
    return {
        "kind": "team_analytics",
        "focus": "hours_average",
        "team_size": len(members),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": label,
        "average_hours": average,
        "people": [],
        "people_count": 0,
        "note": (
            f"Average recorded working hours across {len(members)} direct reports in {label}. "
            "Denominator is direct-report headcount, including members with zero hours."
        ),
    }


def _team_grades(members: list[Employee], departments: dict[int, str]) -> dict:
    grades = sorted({(row.grade or "").strip() for row in members if (row.grade or "").strip()})
    return {
        "kind": "team_analytics",
        "focus": "grades",
        "team_size": len(members),
        "grades": grades,
        "people": [_person(row, departments) for row in members],
        "people_count": len(members),
        "note": "Distinct employment grades represented among direct reports.",
    }


def _team_hours_extreme(
    db: Session,
    members: list[Employee],
    departments: dict[int, str],
    message: str,
    today: date,
) -> dict:
    text = message.lower()
    start, end, label = attendance_window(message, today, None, subject="attendance")
    member_ids = [row.id for row in members]
    hours = _hours_for_window(db, member_ids, start, end)
    # Include reports with zero hours in the window.
    ranked = []
    for row in members:
        item = _person(row, departments)
        item["hours"] = round(hours.get(row.id, 0.0), 1)
        ranked.append(item)
    want_min = bool(re.search(r"\b(fewest|least|lowest|minimum)\b", text))
    ranked.sort(key=lambda item: (item["hours"] if want_min else -item["hours"], item.get("full_name") or ""))
    if not ranked:
        people = []
    else:
        target = ranked[0]["hours"]
        people = [item for item in ranked if item["hours"] == target]
    return {
        "kind": "team_analytics",
        "focus": "hours_extreme",
        "team_size": len(members),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": label,
        "extreme": "fewest" if want_min else "most",
        "people": people,
        "people_count": len(people),
        "note": (
            f"{'Fewest' if want_min else 'Most'} recorded working hours among direct reports "
            f"in {label} (ties listed)."
        ),
    }


def team_analytics_report(db: Session, manager: Employee, message: str, inherited: dict | None = None) -> dict:
    text = message.lower()
    members = team_members(db, manager.id)
    member_ids = [row.id for row in members]
    departments = {row.id: row.name for row in db.query(Department).all()}
    today = org_today()
    inherited = inherited or {}
    department = _team_department(db, message)
    if department is not None and re.search(r"\b(in|only|from|within)\b", text):
        scoped = [row for row in members if row.department_id == department.id]
    else:
        scoped, department = members, None

    if re.search(r"\b(awaiting my approval|waiting for my (approval|decision)|assigned to me)\b", text):
        return _awaiting_approval(db, manager, members, departments)
    # Project assignments and work locations are roster attributes of the manager's own reports.
    if re.search(r"\b(projects?|assigned to|assignments?|allocation|allocated)\b", text) and not re.search(
        r"\b(leave|attendance|absent|remote)\b", text
    ):
        return _project_assignments(db, members, departments, message)
    # Distinct office-location count / listing for the reporting team.
    if re.search(
        r"\b(how many (?:different )?(?:office )?locations?|locations? (?:are )?represented|"
        r"different office locations?|office locations? represented)\b",
        text,
    ):
        locs = sorted({row.location for row in members if (row.location or "").strip()})
        return {
            "kind": "team_analytics",
            "focus": "location_diversity",
            "team_size": len(members),
            "locations": locs,
            "location_count": len(locs),
            "people": [],
            "people_count": 0,
            "note": "Distinct office locations among your direct reports only.",
        }
    if re.search(r"\b(based in|based out of|located in|located at|work(?:s|ing)? (?:from|out of|in) (?:the )?\w+ office|"
                 r"in (?:the )?\w+ office|office location|locations?)\b", text) and not re.search(
        r"\b(leave|attendance|absent|remote|late|hours)\b", text
    ):
        located = _location_filter(members, departments, message)
        if located is not None:
            return located
    # Total remaining annual leave across direct reports (not the pending-request list).
    if re.search(
        r"\b(total|sum|across|still available|how many total).{0,40}\b(remaining|available|annual leave)\b|"
        r"\bannual leave days are still available across\b",
        text,
    ):
        payload = _leave_balances(db, members, departments, lowest=False)
        total = round(sum(float(item.get("remaining") or 0) for item in (payload.get("people") or [])), 1)
        # people in payload is page-limited; recompute from all balances
        year = org_today().year
        rows = (
            db.query(LeaveBalance)
            .filter(
                LeaveBalance.employee_id.in_([row.id for row in members] or [-1]),
                LeaveBalance.leave_type == "annual",
                LeaveBalance.year == year,
            )
            .all()
        )
        total = round(sum(float(r.entitled_days) - float(r.used_days) - float(r.pending_days) for r in rows), 1)
        return {
            "kind": "team_analytics",
            "focus": "remaining_leave_total",
            "team_size": len(members),
            "total_remaining": total,
            "people": payload.get("people") or [],
            "people_count": len(rows),
            "note": f"Sum of remaining annual leave (entitled - used - pending) across direct reports for {year}.",
        }
    if re.search(r"\b(more than half|half of (?:their |the )?entitlement|used more .{0,20}half)\b", text):
        year = org_today().year
        rows = (
            db.query(LeaveBalance)
            .filter(
                LeaveBalance.employee_id.in_([row.id for row in members] or [-1]),
                LeaveBalance.leave_type == "annual",
                LeaveBalance.year == year,
            )
            .all()
        )
        by_id = {row.id: row for row in members}
        people = []
        for row in rows:
            entitled = float(row.entitled_days or 0)
            used = float(row.used_days or 0)
            if entitled > 0 and used > 0.5 * entitled:
                person = by_id.get(row.employee_id)
                item = _person(person, departments) if person else {"full_name": "", "employee_number": "", "department": ""}
                item.update({"entitled": entitled, "used": used, "remaining": entitled - used - float(row.pending_days or 0)})
                people.append(item)
        people.sort(key=lambda item: (-item.get("used", 0), item.get("full_name") or ""))
        return {
            "kind": "team_analytics",
            "focus": "half_entitlement_used",
            "team_size": len(members),
            "people": people,
            "people_count": len(people),
            "note": "Direct reports whose used annual leave exceeds half of their entitled days.",
        }
    if re.search(r"\bremote\b", text) and re.search(r"\bpending\b", text):
        return _remote_and_pending(db, members, departments, message, today)
    if re.search(r"\b(break(?:\s+\w+)?\s+down|by department|department breakdown)\b", text) and (
        re.search(r"\b(remote|remotely|remote[-\s]?work|attendance|absent|present|late|sick)\b", text)
        or inherited.get("asked_status")
        or inherited.get("focus") in {"attendance_summary", "attendance_by_department"}
    ):
        return _attendance_by_department(db, members, departments, message, today, inherited)
    if re.search(r"\b(pending leave|pending requests?|have pending)\b", text):
        return _pending_leave(db, members, departments)
    # A missing-attendance question that mentions approved leave wants the gap cross-checked
    # against leave, not the leave list.
    if re.search(r"\b(incomplete attendance|missing attendance|missing \w+ attendance|attendance gaps?)\b", text):
        return _missing_attendance(db, members, departments, message, today)
    if re.search(r"\b(currently on leave|who is on leave|on approved leave on|on leave on (?:the )?(?:organi[sz]ation|reference) date)\b", text):
        return _currently_on_leave(db, members, departments, today)
    if re.search(r"\bapproved (?:leave|requests?)\b", text) or (
        re.search(r"\bwhat about approved\b", text) or re.search(r"\bapproved requests?\b", text)
    ):
        return _approved_leave(db, members, departments, today)
    if re.search(r"\b(on leave)\b", text):
        return _currently_on_leave(db, members, departments, today)
    if re.search(r"\bexhausted(?:\s+their)?\s+annual|exhausted.{0,20}leave|used all (?:of )?(?:their|the)\b", text):
        return _leave_balances(db, members, departments, lowest=False, exhausted_only=True)
    threshold = re.search(
        r"\b(?:at or below|at most|no more than|up to|down to)\s+(\w+)\s+days?|"
        r"\b(\w+)\s+days?\s+(?:of\s+(?:annual\s+)?leave\s+)?or\s+(?:fewer|less)|"
        r"\b(\w+)\s+or\s+(?:fewer|less)(?:\s+\w+){0,6}\s+days?|"
        r"\b(?:below|under|less than|fewer than)\s+(\w+)\s+days?|<=\s*(\d+)|<\s*(\d+)",
        text,
    )
    if threshold and re.search(r"\b(remaining|balance|left|annual leave|available)\b", text):
        token = next(group for group in threshold.groups() if group)
        limit = _number(token)
        strict = bool(re.search(r"\b(below|under|less than|fewer than)\s+\w+\s+days?|<\s*\d+", text)) and not re.search(
            r"\bat or below\b", text
        )
        if limit is not None:
            return _leave_balances(db, members, departments, lowest=True, max_remaining=limit, strict=strict)
    # Working-hours extremes and comparisons must win over leave "lowest/highest" wording.
    if re.search(r"\b(hours|working hours)\b", text) and re.search(
        r"\b(compare|comparison|versus|vs\.?|against)\b", text
    ):
        named = _named_team_member(members, text)
        if named is not None:
            return _member_hours_compare(db, named, departments, message, today)
        return _team_hours_compare(db, scoped, departments, message, today)
    if re.search(r"\b(hours|working hours)\b", text) and re.search(
        r"\b(fewest|least|lowest|most|highest|maximum|minimum)\b", text
    ):
        return _team_hours_extreme(db, scoped, departments, message, today)
    if re.search(r"\b(average|mean|avg)\b", text) and re.search(r"\b(hours|working hours)\b", text):
        return _team_hours_average(db, scoped, departments, message, today)
    if re.search(r"\b(performance ratings?|grades?)\b", text) and re.search(
        r"\b(direct reports?|my team|team members?|reports?)\b", text
    ):
        # Grades and/or performance ratings for the reporting line.
        people = []
        for member in scoped:
            item = _person(member, departments)
            item["grade"] = member.grade
            item["performance_rating"] = member.performance_rating
            people.append(item)
        people.sort(key=lambda item: (item.get("full_name") or ""))
        return {
            "kind": "team_analytics",
            "focus": "performance_grades",
            "team_size": len(scoped),
            "people": people,
            "people_count": len(people),
            "note": "Employment grades and performance ratings for direct reports only.",
        }
    if re.search(r"\bgrades?\b", text) and re.search(
        r"\b(represented|distinct|which|what|on my team|across my team|team)\b", text
    ):
        return _team_grades(scoped, departments)
    named = _named_team_member(members, text)
    if named is not None and re.search(r"\b(hours|working hours|attendance)\b", text):
        if re.search(r"\b(compare|versus|vs\.?|september|october|august|against)\b", text):
            return _member_hours_compare(db, named, departments, message, today)
        return _member_hours_window(db, named, departments, message, today)
    if re.search(
        r"\b(lowest remaining|highest remaining|leave balances?|leave utilization|remaining (?:annual )?leave|"
        r"annual leave remaining|remaining days|remaining balances?|"
        r"(?:annual )?leave (?:days )?(?:still )?(?:available|left|unused|remaining)|days (?:of leave )?(?:left|remaining|available)|"
        r"entitlement (?:left|remaining|used)|how much leave)\b",
        text,
    ) or (
        re.search(r"\b(lowest|highest)\b", text)
        and re.search(r"\b(leave|balance|remaining|entitlement)\b", text)
        and not re.search(r"\bhours?\b", text)
    ):
        return _leave_balances(db, members, departments, lowest=bool(re.search(r"\blowest\b", text)))
    if re.search(
        r"\b(joined|hired|hire dates?|new hires?|recent hires?|tenure|started (?:with|at|in|before|after|since)|start dates?|"
        r"came on board|onboarded|\d{4} or (?:later|earlier|after|before))\b",
        text,
    ):
        return _recent_hires(members, departments, message, today)
    if re.search(
        r"\b(blank|missing|empty|without a|no)\s+(?:a |an |their |the |any |stored |recorded |valid )*"
        r"(?:phone|job titles?|titles?|email|location|manager)\b|"
        r"\b(?:phone|job titles?|titles?|email)\s+(?:field\s+)?(?:is\s+)?(?:blank|missing|empty)\b|"
        r"\b(?:profile|employment) (?:fields?|information|details?|records?) (?:\w+ )?(?:incomplete|missing|blank|empty)\b|"
        r"\bincomplete (?:\w+ )?(?:profiles?|fields?)\b|\bprofile completeness\b",
        text,
    ):
        return _incomplete_profiles(members, departments, message)
    # "Confirmed late rows" versus "unverified missing weekdays": two different evidence classes,
    # reported side by side and never merged into one risk score.
    if re.search(r"\blate\b", text) and re.search(
        r"\b(missing (?:weekdays?|days?|rows?|attendance)|unverified|no attendance row|gaps?|unrecorded|not recorded|"
        r"missing scheduled|confirmed (?:vs|versus|against|from))\b",
        text,
    ):
        return _late_vs_missing(db, members, departments, message, today)
    late_threshold = None
    if re.search(r"\blate\b", text):
        late_threshold = re.search(
            r"\b(?:at least|minimum of|a minimum of|\u2265)\s*(\w+)\s+(?:late|times|days|occasions|rows)|"
            r"\b(\w+)\s+or\s+more\s+(?:late|times|days|occasions|rows)|\b(\w+)\+\s+(?:late|times|days)|"
            r"\bmore than\s+(\w+)\s+(?:late|times|days|occasions|rows)|\blate\s+(?:on\s+|for\s+)?(?:at least\s+)?(\w+)\s+or\s+more",
            text,
        )
    if late_threshold or re.search(r"\b(repeated late|repeatedly late|late attendance|late arrivals?|multiple late|late more than once)\b", text):
        minimum = 2
        if late_threshold:
            token = next(group for group in late_threshold.groups() if group)
            value = _number(token)
            if value is not None:
                minimum = int(value) + (1 if re.search(r"\bmore than\b", text) else 0)
        if late_threshold or not re.search(r"\b(how many|total|count)\b", text):
            return _repeated_late(db, scoped, departments, message, today, minimum=minimum, department=department)
    if re.search(
        r"\b(incomplete (?:employment|profile|information)|missing profile|"
        r"attendance patterns?|attendance concerns?|main attendance|require attention|"
        r"anomal(?:y|ies)|data quality)\b",
        text,
    ):
        return _attention_patterns(db, members, departments, message, today)
    if re.search(r"\b(incomplete attendance|missing attendance|missing \w+ attendance|attendance gaps?)\b", text):
        return _missing_attendance(db, members, departments, message, today)
    if re.search(r"\bzero\b", text) and re.search(r"\bhours?\b", text):
        start, end, label = attendance_window(message, today, None, subject="attendance")
        member_ids = [row.id for row in scoped]
        rows = (
            db.query(Attendance)
            .filter(
                Attendance.employee_id.in_(member_ids or [-1]),
                Attendance.work_date >= start,
                Attendance.work_date <= end,
                Attendance.hours == 0,
            )
            .all()
        )
        return {
            "kind": "team_analytics",
            "focus": "zero_hours",
            "team_size": len(scoped),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "label": label,
            "zero_hours_rows": len(rows),
            "people": [],
            "people_count": len({row.employee_id for row in rows}),
            "note": "Attendance rows for direct reports with zero recorded hours in the selected window.",
        }
    if re.search(r"\bweekday\b", text) and re.search(r"\babsent", text):
        start, end, label = attendance_window(message, today, None, subject="attendance")
        member_ids = [row.id for row in scoped]
        rows = (
            db.query(Attendance)
            .filter(
                Attendance.employee_id.in_(member_ids or [-1]),
                Attendance.work_date >= start,
                Attendance.work_date <= end,
                Attendance.status == "absent",
            )
            .all()
        )
        from collections import Counter

        counts = Counter(row.work_date.strftime("%A") for row in rows)
        top = counts.most_common(1)[0] if counts else ("", 0)
        return {
            "kind": "team_analytics",
            "focus": "absent_weekday_peak",
            "team_size": len(scoped),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "label": label,
            "weekday_peak": top[0],
            "weekday_peak_count": int(top[1]),
            "weekday_counts": dict(counts),
            "people": [],
            "people_count": 0,
            "note": "Weekday with the most absent attendance rows among direct reports.",
        }
    # Per-person working hours for every direct report (not a status-count roster).
    if re.search(r"\b(hours|working hours)\b", text) and re.search(
        r"\b(each|every|per (?:person|report|member|employee)|for each|breakdown by (?:person|report|member))\b",
        text,
    ):
        return _team_hours_by_member(db, scoped, departments, message, today)
    # Multi-turn: "their present days for the same month" keeps prior top remote person + window.
    if (
        inherited.get("anchor_employee_id")
        and re.search(r"\b(their|them|they|that (?:person|employee|report)|his|her)\b", text)
        and re.search(r"\b(present|remote|absent|late|sick|leave|hours)\b", text)
    ):
        anchor_id = int(inherited["anchor_employee_id"])
        anchored = [row for row in members if row.id == anchor_id]
        if anchored:
            return _team_attendance_summary(
                db,
                anchored,
                departments,
                message,
                today,
                department=department,
                inherited=inherited,
            )
    if re.search(r"\b(absent|absences?|present|remote(?:ly)?|attendance|late|hours|sick)\b", text):
        return _team_attendance_summary(
            db, scoped, departments, message, today, department=department, inherited=inherited
        )
    if re.search(r"\b(remaining|balance|entitlement|annual leave)\b", text):
        return _leave_balances(db, members, departments, lowest=bool(re.search(r"\blowest\b", text)))
    # Follow-ups that only change the date window keep the prior attendance/analytics focus.
    prior_focus = str(inherited.get("focus") or "")
    if inherited.get("subject") in {"team_analytics", "team_roster"} and (
        prior_focus.startswith("attendance")
        or prior_focus in {"hours_compare", "hours_extreme", "zero_hours", "absent_weekday_peak", "performance_grades"}
        or inherited.get("asked_status")
        or re.search(r"\b(switch|change|now|instead|same summary|that summary|only|same month)\b", text)
    ) and (
        re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
            r"this month|last month|same month|window|period)\b",
            text,
        )
        or inherited.get("start")
    ):
        return _team_attendance_summary(
            db, scoped, departments, message, today, department=department, inherited=inherited
        )
    # Do not default roster-like or vague asks to pending leave.
    return {
        "kind": "team_analytics",
        "focus": "clarify_team_metric",
        "team_size": len(members),
        "people": [],
        "people_count": 0,
        "note": (
            "Ask for a specific team metric such as attendance, hours, leave balances, "
            "pending approvals, grades, or the direct-report roster."
        ),
    }


def _recent_hires(members: list[Employee], departments: dict[int, str], message: str, today: date) -> dict:
    """Direct reports filtered by hire year. Years come from the message; the roster never widens."""
    text = message.lower()
    later = re.search(r"\b(?:in\s+)?(\d{4})\s+or\s+(?:later|after)\b|\b(?:since|from|on or after)\s+(\d{4})\b", text)
    before = re.search(r"\b(?:before|prior to|earlier than)\s+(\d{4})\b", text)
    after = re.search(r"\bafter\s+(\d{4})\b", text)
    exact = re.search(r"\b(?:in|during)\s+(\d{4})\b", text)
    if later:
        year = int(next(group for group in later.groups() if group))
        selected = [row for row in members if row.hire_date and row.hire_date.year >= year]
        rule = f"hire_date year {year} or later"
    elif before:
        year = int(before.group(1))
        selected = [row for row in members if row.hire_date and row.hire_date.year < year]
        rule = f"hire_date before {year}"
    elif after:
        year = int(after.group(1))
        selected = [row for row in members if row.hire_date and row.hire_date.year > year]
        rule = f"hire_date after {year}"
    elif exact:
        year = int(exact.group(1))
        selected = [row for row in members if row.hire_date and row.hire_date.year == year]
        rule = f"hire_date in {year}"
    else:
        cutoff = today.replace(year=today.year - 1)
        selected = [row for row in members if row.hire_date and row.hire_date >= cutoff]
        rule = f"hire_date on or after {cutoff.isoformat()} (last 12 months)"
    people = []
    for row in sorted(selected, key=lambda item: (item.hire_date, item.full_name)):
        item = _person(row, departments)
        item["hire_date"] = row.hire_date.isoformat()
        people.append(item)
    return {
        "kind": "team_analytics",
        "focus": "recent_hires",
        "team_size": len(members),
        "people": people[:PAGE],
        "people_count": len(people),
        "rule": rule,
        "note": f"Direct reports whose {rule}. Hire dates come from the employee record; no other employees are included.",
    }


def _incomplete_profiles(members: list[Employee], departments: dict[int, str], message: str) -> dict:
    """Direct reports with blank stored fields that the message asked about."""
    text = message.lower()
    asked = []
    if re.search(r"\bphone\b", text):
        asked.append("phone")
    if re.search(r"\b(job titles?|titles?)\b", text):
        asked.append("job_title")
    if re.search(r"\bemail\b", text):
        asked.append("email")
    if re.search(r"\blocation\b", text):
        asked.append("location")
    if not asked:
        asked = ["phone", "job_title", "email", "location"]
    people = []
    for row in members:
        blanks = [field for field in asked if not (getattr(row, field, None) or "").strip()]
        if blanks:
            item = _person(row, departments)
            item["incomplete_fields"] = blanks
            people.append(item)
    fields = ", ".join(asked)
    return {
        "kind": "team_analytics",
        "focus": "incomplete_profiles",
        "team_size": len(members),
        "people": people[:PAGE],
        "people_count": len(people),
        "fields": asked,
        "note": (
            f"Checked stored field(s) {fields} on each direct report's employee record. "
            "Blank means an empty string or null; no field was inferred."
        ),
    }


def _project_assignments(db: Session, members: list[Employee], departments: dict[int, str], message: str) -> dict:
    """Direct reports' stored project assignments, optionally narrowed to a project named in the message."""
    text = message.lower()
    member_ids = [row.id for row in members]
    projects = db.query(Project).all()
    named = None
    for project in projects:
        name = project.name.lower()
        code = (project.code or "").lower()
        tokens = [token for token in re.split(r"\W+", name) if len(token) >= 4 and token != "project"]
        if name in text or (code and re.search(rf"\b{re.escape(code)}\b", text)) or any(
            re.search(rf"\b{re.escape(token)}\b", text) for token in tokens
        ):
            named = project
            break
    rows = []
    if member_ids:
        query = (
            db.query(ProjectAssignment, Project)
            .join(Project, Project.id == ProjectAssignment.project_id)
            .filter(ProjectAssignment.employee_id.in_(member_ids))
        )
        if named is not None:
            query = query.filter(Project.id == named.id)
        rows = query.all()
    by_employee: dict[int, list[dict]] = defaultdict(list)
    for assignment, project in rows:
        by_employee[assignment.employee_id].append(
            {
                "project": project.name,
                "code": project.code,
                "status": project.status,
                "role_name": assignment.role_name,
                "allocation_pct": assignment.allocation_pct,
            }
        )
    people = []
    for row in members:
        if row.id in by_employee:
            item = _person(row, departments)
            item["projects"] = sorted(by_employee[row.id], key=lambda entry: entry["project"])
            people.append(item)
    if named is not None:
        note = (
            f"Checked the project_assignments table for '{named.name}' ({named.code}, status {named.status}) against "
            "your direct reports only. Assignment rows are the only source; no project membership was inferred."
        )
    else:
        note = (
            "Checked the project_assignments table for your direct reports only. Assignment rows are the only source; "
            "no project membership was inferred."
        )
    return {
        "kind": "team_analytics",
        "focus": "project_assignments",
        "team_size": len(members),
        "people": people[:PAGE],
        "people_count": len(people),
        "project": None if named is None else {"name": named.name, "code": named.code, "status": named.status},
        "note": note,
    }


def _location_filter(members: list[Employee], departments: dict[int, str], message: str) -> dict | None:
    """Direct reports in a location named in the message, or grouped by stored location."""
    text = message.lower()
    locations = sorted({(row.location or "").strip() for row in members if (row.location or "").strip()})
    named = next((loc for loc in locations if re.search(rf"\b{re.escape(loc.lower())}\b", text)), None)
    if named is None:
        # A location that none of the reports has: answer zero rather than widening the roster.
        candidate = re.search(r"\b(?:based|located) (?:in|at|out of) (?:the )?([a-z][a-z .'-]+?)(?:\s+office)?[\s?.!,]*$", text)
        if candidate:
            named = candidate.group(1).strip().title()
        elif not re.search(r"\blocations?\b", text):
            return None
    if named is not None:
        selected = [row for row in members if (row.location or "").strip().lower() == named.lower()]
        people = []
        for row in sorted(selected, key=lambda item: item.full_name):
            item = _person(row, departments)
            item["location"] = row.location
            people.append(item)
        return {
            "kind": "team_analytics",
            "focus": "location_filter",
            "team_size": len(members),
            "people": people[:PAGE],
            "people_count": len(people),
            "location": named,
            "team_locations": locations,
            "note": (
                f"Location comes from the stored location field on each direct report's employee record; "
                f"{len(members)} direct reports were checked and no other employees are included."
            ),
        }
    grouped: dict[str, list[Employee]] = defaultdict(list)
    for row in members:
        grouped[(row.location or "").strip() or "(blank)"].append(row)
    people = []
    for row in sorted(members, key=lambda item: ((item.location or ""), item.full_name)):
        item = _person(row, departments)
        item["location"] = row.location or "(blank)"
        people.append(item)
    return {
        "kind": "team_analytics",
        "focus": "location_filter",
        "team_size": len(members),
        "people": people[:PAGE],
        "people_count": len(people),
        "location": None,
        "team_locations": locations,
        "by_location": [
            {"location": loc, "count": len(rows), "people": [r.full_name for r in rows]}
            for loc, rows in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ],
        "note": "Location comes from the stored location field on each direct report's employee record.",
    }


def _person(row: Employee, departments: dict[int, str]) -> dict:
    return {
        "full_name": row.full_name,
        "employee_number": row.employee_number,
        "department": departments.get(row.department_id, ""),
        "job_title": row.job_title,
    }


def _awaiting_approval(db: Session, manager: Employee, members: list[Employee], departments: dict[int, str]) -> dict:
    """Leave still pending where this manager is the routed/assigned approver."""
    member_ids = [row.id for row in members]
    rows = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.employee_id.in_(member_ids or [-1]),
            LeaveRequest.status == "pending_manager",
            LeaveRequest.routed_manager_id == manager.id,
        )
        .order_by(LeaveRequest.submitted_at.desc())
        .all()
    )
    by_id = {row.id: row for row in members}
    people = []
    for row in rows[:PAGE]:
        person = by_id.get(row.employee_id)
        item = _person(person, departments) if person else {"full_name": "", "employee_number": "", "department": ""}
        item.update(
            {
                "request_id": row.id,
                "leave_type": row.leave_type,
                "status": row.status,
                "start": row.start_date.isoformat(),
                "end": row.end_date.isoformat(),
                "business_days": row.business_days,
                "assigned_approver_id": row.routed_manager_id,
            }
        )
        people.append(item)
    return {
        "kind": "team_analytics",
        "focus": "awaiting_approval",
        "team_size": len(members),
        "people": people,
        "people_count": len(rows),
        "note": (
            "These are pending_manager requests routed to the signed-in manager. "
            "Assigned approver is not the same as an approval decision. No decision rows are implied."
        ),
    }


def _pending_leave(db: Session, members: list[Employee], departments: dict[int, str]) -> dict:
    member_ids = [row.id for row in members]
    rows = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.employee_id.in_(member_ids or [-1]),
            LeaveRequest.status.in_(("pending_manager", "pending_hr")),
        )
        .order_by(LeaveRequest.submitted_at.desc())
        .all()
    )
    by_id = {row.id: row for row in members}
    people = []
    for row in rows[:PAGE]:
        person = by_id.get(row.employee_id)
        item = _person(person, departments) if person else {"full_name": "", "employee_number": "", "department": ""}
        item.update(
            {
                "request_id": row.id,
                "leave_type": row.leave_type,
                "status": row.status,
                "start": row.start_date.isoformat(),
                "end": row.end_date.isoformat(),
                "business_days": row.business_days,
                "assigned_approver_id": row.routed_manager_id,
            }
        )
        people.append(item)
    return {
        "kind": "team_analytics",
        "focus": "pending_leave",
        "team_size": len(members),
        "people": people,
        "people_count": len(rows),
        "note": "Pending means leave status pending_manager or pending_hr for direct reports only.",
    }


def _currently_on_leave(
    db: Session, members: list[Employee], departments: dict[int, str], today: date
) -> dict:
    member_ids = [row.id for row in members]
    rows = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.employee_id.in_(member_ids or [-1]),
            LeaveRequest.status == "approved",
            LeaveRequest.start_date <= today,
            LeaveRequest.end_date >= today,
        )
        .all()
    )
    by_id = {row.id: row for row in members}
    people = []
    for row in rows:
        person = by_id.get(row.employee_id)
        item = _person(person, departments) if person else {"full_name": "", "employee_number": "", "department": ""}
        item.update(
            {
                "request_id": row.id,
                "leave_type": row.leave_type,
                "status": row.status,
                "start": row.start_date.isoformat(),
                "end": row.end_date.isoformat(),
            }
        )
        people.append(item)
    return {
        "kind": "team_analytics",
        "focus": "on_leave",
        "team_size": len(members),
        "people": people,
        "people_count": len(people),
        "note": (
            "Currently on leave means an approved request whose dates include the organization date. "
            "Pending requests are excluded."
        ),
    }


def _leave_balances(
    db: Session,
    members: list[Employee],
    departments: dict[int, str],
    *,
    lowest: bool,
    exhausted_only: bool = False,
    max_remaining: float | None = None,
    strict: bool = False,
) -> dict:
    member_ids = [row.id for row in members]
    year = org_today().year
    rows = (
        db.query(LeaveBalance)
        .filter(
            LeaveBalance.employee_id.in_(member_ids or [-1]),
            LeaveBalance.leave_type == "annual",
            LeaveBalance.year == year,
        )
        .all()
    )
    by_id = {row.id: row for row in members}
    balances = []
    for row in rows:
        person = by_id.get(row.employee_id)
        remaining = float(row.entitled_days) - float(row.used_days) - float(row.pending_days)
        if exhausted_only and remaining > 0:
            continue
        if max_remaining is not None and (remaining >= max_remaining if strict else remaining > max_remaining):
            continue
        item = _person(person, departments) if person else {"full_name": "", "employee_number": "", "department": ""}
        item.update(
            {
                "leave_type": "annual",
                "entitled": float(row.entitled_days),
                "used": float(row.used_days),
                "pending": float(row.pending_days),
                "remaining": remaining,
            }
        )
        balances.append(item)
    balances.sort(key=lambda item: (item["remaining"], item.get("full_name") or ""))
    shown = balances[:PAGE]
    if exhausted_only:
        note = (
            "Direct reports who have exhausted annual leave (remaining = entitled - used - pending is 0 or below). "
            "Pending leave is reserved but not yet used."
        )
        focus = "exhausted_leave"
    elif max_remaining is not None:
        comparator = "below" if strict else "at or below"
        note = (
            f"Direct reports whose remaining annual leave for {year} is {comparator} {max_remaining:g} days. "
            "Remaining = entitled - used - pending; pending leave is reserved, not used."
        )
        focus = "low_balances"
    else:
        note = (
            f"Annual leave remaining for direct reports only ({year} annual balances), ranked lowest first. "
            "Remaining = entitled - used - pending."
        )
        focus = "leave_balances"
    return {
        "kind": "team_analytics",
        "focus": focus,
        "team_size": len(members),
        "people": shown,
        "people_count": len(balances),
        "threshold": max_remaining,
        "note": note,
    }


def _repeated_late(
    db: Session,
    members: list[Employee],
    departments: dict[int, str],
    message: str,
    today: date,
    *,
    minimum: int = 2,
    department: Department | None = None,
) -> dict:
    start, end, label = attendance_window(message, today, None, subject="attendance")
    member_ids = [row.id for row in members]
    rows = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id.in_(member_ids or [-1]),
            Attendance.work_date >= start,
            Attendance.work_date <= end,
            Attendance.status == "late",
        )
        .all()
    )
    counts: dict[int, int] = defaultdict(int)
    for row in rows:
        counts[row.employee_id] += 1
    by_id = {row.id: row for row in members}
    people = []
    for employee_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count < minimum:
            continue
        person = by_id.get(employee_id)
        item = _person(person, departments) if person else {"full_name": "", "employee_number": "", "department": ""}
        item["late_count"] = count
        people.append(item)
    scope = f" in {department.name}" if department is not None else ""
    return {
        "kind": "team_analytics",
        "focus": "repeated_late",
        "team_size": len(members),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": label,
        "people": people[:PAGE],
        "people_count": len(people),
        "minimum": minimum,
        "late_rows_total": sum(counts.values()),
        "department": department.name if department is not None else "",
        "note": (
            f"Repeated late means {minimum} or more attendance rows with status late for a direct report{scope} "
            f"in the window. Direct reports with fewer late rows are not listed."
        ),
    }


def _missing_attendance(
    db: Session, members: list[Employee], departments: dict[int, str], message: str, today: date
) -> dict:
    start, end, label = attendance_window(message, today, None, subject="attendance")
    member_ids = [row.id for row in members]
    approved = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.employee_id.in_(member_ids or [-1]),
            LeaveRequest.status == "approved",
            LeaveRequest.start_date <= end,
            LeaveRequest.end_date >= start,
        )
        .all()
    )
    leave_by_employee: dict[int, list[LeaveRequest]] = defaultdict(list)
    for row in approved:
        leave_by_employee[row.employee_id].append(row)
    people = []
    explained_total = 0
    unexplained_total = 0
    for person in members:
        schedule = _scheduled_working_days(db, person, start, end)
        stored = {
            row.work_date
            for row in db.query(Attendance)
            .filter(
                Attendance.employee_id == person.id,
                Attendance.work_date >= start,
                Attendance.work_date <= end,
            )
            .all()
        }
        scheduled_dates = schedule.get("scheduled_dates") or []
        missing_days = [day for day in scheduled_dates if day not in stored]
        if missing_days:
            explained = [
                day
                for day in missing_days
                if any(leave.start_date <= day <= leave.end_date for leave in leave_by_employee.get(person.id, []))
            ]
            explained_total += len(explained)
            unexplained_total += len(missing_days) - len(explained)
            item = _person(person, departments)
            item["missing_count"] = len(missing_days)
            item["missing_dates"] = [day.isoformat() for day in missing_days[:10]]
            item["leave_explained_count"] = len(explained)
            people.append(item)
    return {
        "kind": "team_analytics",
        "focus": "missing_attendance",
        "team_size": len(members),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": label,
        "people": people[:PAGE],
        "people_count": len(people),
        "leave_cross_check": {
            "explained_days": explained_total,
            "unexplained_days": unexplained_total,
            "approved_requests_in_window": len(approved),
        },
        "note": (
            "Incomplete/missing attendance means scheduled working days with no attendance row. "
            "Missing days are not classified as absences; each missing day was cross-checked against approved leave."
        ),
    }


def _late_vs_missing(
    db: Session, members: list[Employee], departments: dict[int, str], message: str, today: date
) -> dict:
    """Confirmed late rows and unverified missing scheduled days, kept as separate evidence classes."""
    late = _repeated_late(db, members, departments, message, today, minimum=1)
    missing = _missing_attendance(db, members, departments, message, today)
    late_by_number = {item["employee_number"]: item.get("late_count", 0) for item in late["people"]}
    missing_by_number = {
        item["employee_number"]: (item.get("missing_count", 0), item.get("leave_explained_count", 0))
        for item in missing["people"]
    }
    people = []
    for person in members:
        confirmed = late_by_number.get(person.employee_number, 0)
        unverified, explained = missing_by_number.get(person.employee_number, (0, 0))
        if confirmed or unverified:
            item = _person(person, departments)
            item["late_count"] = confirmed
            item["missing_count"] = unverified
            item["leave_explained_count"] = explained
            item["evidence"] = (
                "confirmed late rows" if confirmed and not unverified else
                "unverified missing days only" if unverified and not confirmed else
                "confirmed late rows plus unverified missing days"
            )
            people.append(item)
    people.sort(key=lambda item: (-item["late_count"], -item["missing_count"], item["full_name"]))
    return {
        "kind": "team_analytics",
        "focus": "late_vs_missing",
        "team_size": len(members),
        "start": late["start"],
        "end": late["end"],
        "label": late["label"],
        "people": people[:PAGE],
        "people_count": len(people),
        "confirmed_late_rows": late.get("late_rows_total", 0),
        "confirmed_late_people": len(late["people"]),
        "unverified_missing_days": sum(count for count, _ in missing_by_number.values()),
        "unverified_missing_people": len(missing["people"]),
        "leave_cross_check": missing.get("leave_cross_check"),
        "note": (
            "Confirmed means a stored attendance row with status late. Unverified means a scheduled working day with "
            "no attendance row at all; it is not an absence and not lateness, only a gap in the record. The two classes "
            "are counted separately and were not combined into a single risk figure."
        ),
    }


def _approved_leave(
    db: Session, members: list[Employee], departments: dict[int, str], today: date
) -> dict:
    member_ids = [row.id for row in members]
    rows = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.employee_id.in_(member_ids or [-1]),
            LeaveRequest.status == "approved",
            LeaveRequest.end_date >= today,
        )
        .order_by(LeaveRequest.start_date.asc())
        .all()
    )
    by_id = {row.id: row for row in members}
    people = []
    for row in rows[:PAGE]:
        person = by_id.get(row.employee_id)
        item = _person(person, departments) if person else {"full_name": "", "employee_number": "", "department": ""}
        item.update(
            {
                "request_id": row.id,
                "leave_type": row.leave_type,
                "status": row.status,
                "start": row.start_date.isoformat(),
                "end": row.end_date.isoformat(),
                "business_days": row.business_days,
            }
        )
        people.append(item)
    return {
        "kind": "team_analytics",
        "focus": "approved_leave",
        "team_size": len(members),
        "people": people,
        "people_count": len(rows),
        "note": "Approved leave for direct reports only, including upcoming approved requests.",
    }


def _remote_and_pending(
    db: Session,
    members: list[Employee],
    departments: dict[int, str],
    message: str,
    today: date,
) -> dict:
    start, end, label = attendance_window(message, today, None, subject="attendance")
    member_ids = [row.id for row in members]
    remote_ids = {
        row.employee_id
        for row in db.query(Attendance)
        .filter(
            Attendance.employee_id.in_(member_ids or [-1]),
            Attendance.work_date >= start,
            Attendance.work_date <= end,
            Attendance.status == "remote",
        )
        .all()
    }
    pending = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.employee_id.in_(list(remote_ids) or [-1]),
            LeaveRequest.status.in_(("pending_manager", "pending_hr")),
        )
        .all()
    )
    by_id = {row.id: row for row in members}
    people = []
    for row in pending[:PAGE]:
        person = by_id.get(row.employee_id)
        item = _person(person, departments) if person else {"full_name": "", "employee_number": "", "department": ""}
        item.update(
            {
                "request_id": row.id,
                "leave_type": row.leave_type,
                "status": row.status,
                "start": row.start_date.isoformat(),
                "end": row.end_date.isoformat(),
                "business_days": row.business_days,
            }
        )
        people.append(item)
    return {
        "kind": "team_analytics",
        "focus": "remote_and_pending",
        "team_size": len(members),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": label,
        "people": people,
        "people_count": len(people),
        "note": (
            "Direct reports who recorded remote attendance in the window and also have pending leave. "
            "Organization-wide employees are not included."
        ),
    }


def _attention_patterns(
    db: Session,
    members: list[Employee],
    departments: dict[int, str],
    message: str,
    today: date,
) -> dict:
    """Team-scoped attendance/employment completeness findings for direct reports only."""
    missing = _missing_attendance(db, members, departments, message, today)
    late = _repeated_late(db, members, departments, message, today)
    by_number = {}
    for item in missing.get("people") or []:
        key = item.get("employee_number")
        entry = dict(item)
        entry["flags"] = ["incomplete_attendance"]
        by_number[key] = entry
    for item in late.get("people") or []:
        key = item.get("employee_number")
        entry = by_number.get(key) or dict(item)
        flags = list(entry.get("flags") or [])
        if "repeated_late" not in flags:
            flags.append("repeated_late")
        entry["flags"] = flags
        entry["late_count"] = item.get("late_count")
        by_number[key] = entry
    for person in members:
        incomplete_fields = []
        if not person.job_title:
            incomplete_fields.append("job_title")
        if not person.location:
            incomplete_fields.append("location")
        if person.manager_id is None:
            incomplete_fields.append("manager_id")
        if incomplete_fields:
            item = by_number.get(person.employee_number) or _person(person, departments)
            flags = list(item.get("flags") or [])
            if "incomplete_employment" not in flags:
                flags.append("incomplete_employment")
            item["flags"] = flags
            item["incomplete_fields"] = incomplete_fields
            by_number[person.employee_number] = item
    people = list(by_number.values())
    people.sort(key=lambda item: (-len(item.get("flags") or []), item.get("full_name") or ""))
    return {
        "kind": "team_analytics",
        "focus": "attention_patterns",
        "team_size": len(members),
        "start": missing.get("start"),
        "end": missing.get("end"),
        "label": missing.get("label"),
        "people": people[:PAGE],
        "people_count": len(people),
        "note": (
            "Attention patterns for direct reports only: missing scheduled attendance, "
            "repeated late attendance, and empty employment fields present on the employee record. "
            "No organization-wide employees are included."
        ),
    }


def _attendance_by_department(
    db: Session,
    members: list[Employee],
    departments: dict[int, str],
    message: str,
    today: date,
    inherited: dict | None = None,
) -> dict:
    """Unique direct reports with a status (default remote), grouped by department."""
    text = message.lower()
    inherited = inherited or {}
    # Team analytics context stores subject=team_analytics; reuse its window as attendance.
    window_ctx = dict(inherited)
    if window_ctx.get("start") and window_ctx.get("end") and window_ctx.get("subject") in {
        "team_analytics",
        "team_roster",
        "attendance",
    }:
        window_ctx["subject"] = "attendance"
    start, end, label = attendance_window(message, today, window_ctx, subject="attendance")
    statuses = ("present", "remote", "absent", "late", "sick", "leave")
    asked = [status for status in statuses if re.search(rf"\b{status}\b", text)]
    if re.search(r"\bremotely\b|remote[-\s]?work", text) and "remote" not in asked:
        asked.append("remote")
    asked_status = asked[0] if len(asked) == 1 else ("remote" if "remote" in asked else None)
    if asked_status is None:
        prior = inherited.get("asked_status")
        asked_status = prior if prior in statuses else "remote"
    member_ids = [row.id for row in members]
    rows = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id.in_(member_ids or [-1]),
            Attendance.work_date >= start,
            Attendance.work_date <= end,
            Attendance.status == asked_status,
        )
        .all()
    )
    by_emp: dict[int, int] = defaultdict(int)
    for row in rows:
        by_emp[row.employee_id] += 1
    by_id = {row.id: row for row in members}
    groups: dict[str, list[dict]] = defaultdict(list)
    for employee_id, day_count in by_emp.items():
        person = by_id.get(employee_id)
        if person is None:
            continue
        item = _person(person, departments)
        item["days"] = day_count
        groups[item["department"] or "(blank)"].append(item)
    breakdown = []
    for dept_name, people in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        breakdown.append(
            {
                "department": dept_name,
                "unique_people": len(people),
                "days": sum(item["days"] for item in people),
                "people": people[:PAGE],
            }
        )
    unique_total = sum(item["unique_people"] for item in breakdown)
    return {
        "kind": "team_analytics",
        "focus": "attendance_by_department",
        "team_size": len(members),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": label,
        "asked_status": asked_status,
        "people_count": unique_total,
        "by_department": breakdown,
        "note": (
            f"Unique direct reports with at least one {asked_status} attendance row in the window, "
            "grouped by department. Employees outside your reporting line are not included."
        ),
    }


def _team_attendance_summary(
    db: Session,
    members: list[Employee],
    departments: dict[int, str],
    message: str,
    today: date,
    *,
    department: Department | None = None,
    inherited: dict | None = None,
) -> dict:
    text = message.lower()
    inherited = inherited or {}
    # Preserve prior month when follow-up says "same month" / "their …" without a new window.
    if (
        inherited.get("start")
        and inherited.get("end")
        and (
            re.search(r"\b(same month|that month|same period|same window)\b", text)
            or (
                re.search(r"\b(their|them|that (?:person|employee|report))\b", text)
                and not parse_asked_window(message, today)
            )
        )
    ):
        try:
            start = date.fromisoformat(str(inherited["start"])[:10])
            end = date.fromisoformat(str(inherited["end"])[:10])
            label = str(inherited.get("label") or "same month")
        except Exception:
            start, end, label = attendance_window(message, today, inherited, subject="attendance")
    else:
        start, end, label = attendance_window(message, today, inherited, subject="attendance")
    member_ids = [row.id for row in members]
    rows = (
        db.query(Attendance)
        .filter(
            Attendance.employee_id.in_(member_ids or [-1]),
            Attendance.work_date >= start,
            Attendance.work_date <= end,
        )
        .all()
    )
    statuses = ("present", "remote", "absent", "late", "sick", "leave")
    counts = defaultdict(lambda: {status: 0 for status in statuses})
    hours = defaultdict(float)
    for row in rows:
        if row.status in counts[row.employee_id]:
            counts[row.employee_id][row.status] += 1
        hours[row.employee_id] += float(row.hours or 0)
    totals = {status: sum(item[status] for item in counts.values()) for status in statuses}
    totals["rows"] = len(rows)
    totals["hours"] = round(sum(hours.values()), 2)
    asked = [status for status in statuses if re.search(rf"\b{status}\b", text)]
    if re.search(r"\bremotely\b", text) and "remote" not in asked:
        asked.append("remote")
    if re.search(r"\babsences?\b", text) and "absent" not in asked:
        asked.append("absent")
    asked_status = asked[0] if len(asked) == 1 else None
    by_id = {row.id: row for row in members}
    people = []
    for employee_id, tallies in counts.items():
        person = by_id.get(employee_id)
        item = _person(person, departments) if person else {"full_name": "", "employee_number": "", "department": ""}
        item["counts"] = tallies
        item["hours"] = round(hours[employee_id], 2)
        people.append(item)
    if asked_status:
        people.sort(key=lambda item: (-item["counts"].get(asked_status, 0), item.get("full_name") or ""))
    unique_with_status = 0
    if asked_status:
        unique_with_status = sum(1 for item in people if item["counts"].get(asked_status, 0) > 0)
    # "How many of my direct reports worked remotely" asks for unique people, not remote-day rows.
    people_question = bool(
        asked_status
        and re.search(r"\b(how many|count|number of)\b", text)
        and re.search(r"\b(direct reports?|reports?|team members?|people|employees?|them|those)\b", text)
        and re.search(r"\b(worked|were|was|had)\b", text)
    )
    scope = f"direct reports in {department.name}" if department is not None else "direct reports"
    note = (
        f"Attendance counts for {scope} only, from stored attendance rows in the selected window. "
        "Employees outside your reporting line are not included."
    )
    if people_question and asked_status:
        note = (
            f"{unique_with_status} unique {scope} had at least one {asked_status} attendance row "
            f"in the selected window ({totals.get(asked_status, 0)} {asked_status} attendance days total). "
            "Employees outside your reporting line are not included."
        )
    return {
        "kind": "team_analytics",
        "focus": "attendance_summary",
        "team_size": len(members),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": label,
        "people": people[:PAGE],
        "people_count": unique_with_status if people_question else len(people),
        "totals": totals,
        "asked_status": asked_status,
        "unique_with_status": unique_with_status if asked_status else None,
        "count_mode": "unique_people" if people_question else "attendance_rows",
        "department": department.name if department is not None else "",
        "note": note,
    }
