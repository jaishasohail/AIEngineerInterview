"""Deterministic leave, staffing, and approval rules.

The model may explain these results. It does not calculate them and it cannot approve them.
"""

from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.clock import (
    add_business_days,
    business_days_inclusive,
    first_monday,
    format_day,
    next_month_bounds,
    notice_business_days,
    org_today,
)
from app.models import Department, Employee, LeaveBalance, LeaveRequest, Project, ProjectAssignment
from app.services.holidays import holidays_for_employee

ANNUAL_ENTITLEMENT = 24
SICK_ENTITLEMENT = 10
MAX_CONSECUTIVE = 15
NOTICE_SHORT = 5
NOTICE_LONG = 10
HR_DAY_THRESHOLD = 10
COVERAGE_THRESHOLD = 0.70
LEAVE_YEAR = 2026

OPEN_STATUSES = ("pending_manager", "pending_hr", "approved")


def notice_required(business_days: float) -> int:
    return NOTICE_SHORT if business_days <= 5 else NOTICE_LONG


def remaining_days(balance: LeaveBalance | None) -> float:
    if balance is None:
        return 0
    return round(balance.entitled_days - balance.used_days - balance.pending_days, 2)


def get_balance(db: Session, employee_id: int, leave_type: str = "annual", year: int = LEAVE_YEAR) -> LeaveBalance | None:
    return (
        db.query(LeaveBalance)
        .filter(
            LeaveBalance.employee_id == employee_id,
            LeaveBalance.leave_type == leave_type,
            LeaveBalance.year == year,
        )
        .one_or_none()
    )


def balance_payload(balance: LeaveBalance | None, leave_type: str = "annual") -> dict:
    if balance is None:
        return {
            "leave_type": leave_type,
            "year": LEAVE_YEAR,
            "entitled": 0,
            "used": 0,
            "pending": 0,
            "remaining": 0,
        }
    return {
        "leave_type": balance.leave_type,
        "year": balance.year,
        "entitled": balance.entitled_days,
        "used": balance.used_days,
        "pending": balance.pending_days,
        "remaining": remaining_days(balance),
    }


def _person(employee: Employee, department_name: str | None = None) -> dict:
    payload = {
        "id": employee.id,
        "full_name": employee.full_name,
        "job_title": employee.job_title,
        "location": employee.location,
    }
    if department_name:
        payload["department"] = department_name
    return payload


def project_rows(db: Session, employee_id: int) -> list[dict]:
    rows = (
        db.query(Project, ProjectAssignment)
        .join(ProjectAssignment, ProjectAssignment.project_id == Project.id)
        .filter(ProjectAssignment.employee_id == employee_id, Project.status == "active")
        .all()
    )
    payload = []
    for project, assignment in rows:
        payload.append(
            {
                "code": project.code,
                "name": project.name,
                "deadline": project.deadline.isoformat(),
                "criticality": project.criticality,
                "role": assignment.role_name,
                "allocation_pct": assignment.allocation_pct,
                "description": project.description,
            }
        )
    return payload


def _overlaps(start: date, end: date, other_start: date, other_end: date) -> bool:
    return start <= other_end and other_start <= end


def conflicting_projects(projects: list[dict], start: date, end: date) -> list[dict]:
    conflicts = []
    horizon = end + timedelta(days=5)
    for project in projects:
        deadline = date.fromisoformat(project["deadline"])
        conflict = start <= deadline <= horizon
        critical = project["criticality"] in {"critical", "high"} and start <= deadline <= end
        if conflict:
            conflicts.append({**project, "overlaps_window": start <= deadline <= end, "critical_conflict": critical})
    return conflicts


def team_members(db: Session, manager_id: int) -> list[Employee]:
    return db.query(Employee).filter(Employee.manager_id == manager_id, Employee.employment_status == "active").order_by(Employee.last_name).all()


def team_roster_report(db: Session, manager: Employee, message: str = "") -> dict:
    """Active direct reports for the signed-in manager. Never returns the organization directory.

    The message only selects how the roster is presented (count, grouped by location); it never
    selects which employees are included.
    """
    import re

    from app.models import Department

    members = team_members(db, manager.id)
    departments = {row.id: row.name for row in db.query(Department).all()}
    people = [
        {
            "full_name": person.full_name,
            "employee_number": person.employee_number,
            "job_title": person.job_title,
            "department": departments.get(person.department_id, ""),
            "employment_status": person.employment_status,
            "hire_date": person.hire_date.isoformat() if person.hire_date else None,
            "location": person.location,
            "grade": person.grade,
            "performance_rating": person.performance_rating,
        }
        for person in members
    ]
    text = (message or "").lower()
    focus = "direct_reports"
    by_location: list[dict] = []
    if re.search(r"\blocations?\b", text):
        focus = "locations"
        grouped: dict[str, list[str]] = {}
        for item in people:
            grouped.setdefault((item.get("location") or "").strip() or "(blank)", []).append(
                f"{item['full_name']} ({item['employee_number']})"
            )
        by_location = [
            {"location": name, "count": len(names), "people": names}
            for name, names in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0]))
        ]
    elif re.search(r"\b(how many|count|headcount|number of|manager_id)\b", text):
        focus = "count"
    return {
        "kind": "team_roster",
        "focus": focus,
        "people": people,
        "people_count": len(people),
        "team_size": len(people),
        "by_location": by_location,
        "note": (
            "Direct reports are active employees whose manager_id matches the signed-in manager. "
            "This list is limited to your reporting line, not the full company directory."
        ),
    }


def overlapping_leave(db: Session, employee_ids: list[int], start: date, end: date, exclude_request_id: int | None = None) -> list[dict]:
    if not employee_ids:
        return []
    query = db.query(LeaveRequest, Employee).join(Employee, Employee.id == LeaveRequest.employee_id).filter(
        LeaveRequest.employee_id.in_(employee_ids),
        LeaveRequest.status.in_(OPEN_STATUSES),
        LeaveRequest.start_date <= end,
        LeaveRequest.end_date >= start,
    )
    if exclude_request_id:
        query = query.filter(LeaveRequest.id != exclude_request_id)
    rows = []
    for request, employee in query.all():
        rows.append(
            {
                "request_id": request.id,
                "employee_id": employee.id,
                "full_name": employee.full_name,
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "business_days": request.business_days,
                "status": request.status,
                "leave_type": request.leave_type,
            }
        )
    return rows


def post_approval_facts() -> dict:
    """What the leave workflow does after the manager's decision. Facts come from the approval code."""
    return {
        "kind": "post_approval",
        "hr_day_threshold": HR_DAY_THRESHOLD,
        "hr_required_when": (
            f"the request is longer than {HR_DAY_THRESHOLD} consecutive business days "
            "or the absence overlaps a critical project deadline"
        ),
    }


def approval_chain(employee: Employee, manager: Employee | None, *, business_days: float, critical_conflict: bool) -> dict:
    hr_required = business_days > HR_DAY_THRESHOLD or critical_conflict
    chain = []
    if manager is not None:
        chain.append(
            {
                "stage": 1,
                "role": "manager",
                "employee_id": manager.id,
                "full_name": manager.full_name,
                "reason": "Leave requests route to the approval manager on the employee record.",
            }
        )
    if hr_required:
        chain.append(
            {
                "stage": 2,
                "role": "hr_admin",
                "employee_id": None,
                "full_name": "People Operations",
                "reason": "Global leave approval policy: co-approval above 10 consecutive business days or when a critical project deadline falls inside the absence.",
            }
        )
    return {"hr_required": hr_required, "chain": chain}


def assess_window(
    db: Session,
    employee: Employee,
    start: date,
    end: date,
    *,
    leave_type: str = "annual",
) -> dict:
    today = org_today()
    calendar = holidays_for_employee(db, employee, min(today, start), max(end, start))
    days = business_days_inclusive(start, end, calendar.dates)
    balance = get_balance(db, employee.id, leave_type, start.year if start.year == LEAVE_YEAR else start.year)
    # Balances in this demo are maintained on the 2026 leave year, which also covers requests
    # that start in January 2027 and draw on the current entitlement. Fall back to the seeded year.
    if balance is None:
        balance = get_balance(db, employee.id, leave_type, LEAVE_YEAR)
    remaining = remaining_days(balance)
    notice = notice_business_days(today, start, calendar.dates)
    required_notice = notice_required(days)
    projects = project_rows(db, employee.id)
    project_hits = conflicting_projects(projects, start, end)
    critical = any(item["critical_conflict"] for item in project_hits)
    manager = db.get(Employee, employee.approval_manager_id) if employee.approval_manager_id else None
    teammates = team_members(db, employee.manager_id) if employee.manager_id else []
    teammate_ids = [person.id for person in teammates if person.id != employee.id]
    overlaps = overlapping_leave(db, teammate_ids, start, end)
    team_size = len(teammates) or 1
    out_count = len({item["employee_id"] for item in overlaps}) + 1
    available = max(team_size - out_count, 0)
    available_pct = round(available / team_size, 4)
    chain = approval_chain(employee, manager, business_days=days, critical_conflict=critical)
    hard_blocks = []
    risks = []
    kind_label = "annual leave" if leave_type == "annual" else f"{leave_type} leave"
    if days <= 0:
        hard_blocks.append("The selected range does not contain a business day.")
    if start < today:
        hard_blocks.append("Leave cannot start in the past relative to the organization clock.")
    if leave_type == "annual" and days > MAX_CONSECUTIVE:
        hard_blocks.append(f"The request is {days} business days. The maximum consecutive annual leave is {MAX_CONSECUTIVE}.")
    if remaining < days:
        hard_blocks.append(f"Remaining {kind_label} is {remaining} days. This request needs {days}.")
    if leave_type == "annual" and notice < required_notice:
        hard_blocks.append(
            f"Notice is {notice} business days. This duration requires {required_notice} business days of notice."
        )
    if manager is None:
        hard_blocks.append("No approval manager is recorded, so the request cannot be routed.")
    if critical:
        risks.append("A critical project deadline falls inside this window, so People Operations co-approval is required.")
    if available_pct < COVERAGE_THRESHOLD:
        risks.append(
            f"Team availability would be {available_pct:.0%}, below the {COVERAGE_THRESHOLD:.0%} coverage guide."
        )
    if overlaps:
        names = ", ".join(sorted({item["full_name"] for item in overlaps}))
        risks.append(f"Overlapping team leave: {names}.")
    if employee.manager_id != employee.approval_manager_id:
        risks.append("The approval manager on record is not the current line manager. Routing will follow the approval manager until HR corrects it.")
    if calendar.status == "missing":
        risks.append(f"No holiday calendar is configured for {employee.location}. Only weekends were excluded.")
    holidays_in_window = sorted(day for day in calendar.dates if start <= day <= end)
    if not risks and not hard_blocks:
        risk_level = "low"
    elif hard_blocks:
        risk_level = "blocked"
    elif critical or available_pct < COVERAGE_THRESHOLD:
        risk_level = "high"
    else:
        risk_level = "medium"
    return {
        "leave_type": leave_type,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": f"{format_day(start)} to {format_day(end)}",
        "business_days": days,
        "notice_business_days": notice,
        "notice_required": required_notice,
        "notice_ok": notice >= required_notice,
        "balance_ok": remaining >= days,
        "within_max_consecutive": days <= MAX_CONSECUTIVE,
        "remaining_before": remaining,
        "projects": project_hits,
        "critical_conflict": critical,
        "overlaps": overlaps,
        "team_size": team_size,
        "available_after": available,
        "available_pct": available_pct,
        "hr_required": chain["hr_required"],
        "approval_chain": chain["chain"],
        "hard_blocks": hard_blocks,
        "risks": risks,
        "risk_level": risk_level,
        "holiday_calendar": {
            "location": employee.location,
            "country": calendar.country,
            "status": calendar.status,
            "holidays_in_window": [day.isoformat() for day in holidays_in_window],
        },
    }


def recommend_window(
    db: Session,
    employee: Employee,
    business_days: int,
    *,
    avoid_start: date | None = None,
    earliest: date | None = None,
) -> dict | None:
    today = org_today()
    cursor = earliest or (today + timedelta(days=1))
    while cursor.weekday() != 0:
        cursor += timedelta(days=1)
    closed = holidays_for_employee(db, employee, cursor, cursor + timedelta(days=180)).dates
    best = None
    best_score = None
    for _ in range(18):
        end = add_business_days(cursor, business_days, closed)
        if avoid_start and cursor == avoid_start:
            cursor += timedelta(days=7)
            continue
        assessed = assess_window(db, employee, cursor, end)
        score = 0
        if assessed["hard_blocks"]:
            score += 1000
        if assessed["critical_conflict"]:
            score += 50
        score += int((1 - assessed["available_pct"]) * 20)
        score += len(assessed["overlaps"])
        if best_score is None or score < best_score:
            best = assessed
            best_score = score
        if score == 0:
            break
        cursor += timedelta(days=7)
    return best


def default_requested_window(business_days: int = 10) -> tuple[date, date]:
    start_month, _end_month = next_month_bounds(org_today())
    start = first_monday(start_month.year, start_month.month)
    end = add_business_days(start, business_days)
    return start, end


def plan_leave(
    db: Session,
    employee: Employee,
    business_days: int = 10,
    start: date | None = None,
    end: date | None = None,
    leave_type: str = "annual",
) -> dict:
    if start is None or end is None:
        start, end = default_requested_window(business_days)
        closed = holidays_for_employee(db, employee, start, start + timedelta(days=45)).dates
        end = add_business_days(start, business_days, closed)
    department = db.get(Department, employee.department_id)
    manager = db.get(Employee, employee.manager_id) if employee.manager_id else None
    approval_manager = db.get(Employee, employee.approval_manager_id) if employee.approval_manager_id else None
    requested = assess_window(db, employee, start, end, leave_type=leave_type)
    span_days = requested["business_days"] or business_days
    alternative = None
    if leave_type == "annual" and span_days > 0:
        alternative = recommend_window(
            db,
            employee,
            span_days,
            avoid_start=start,
            earliest=end + timedelta(days=1),
        )
    balance = balance_payload(get_balance(db, employee.id, leave_type, LEAVE_YEAR), leave_type)
    return {
        "leave_type": leave_type,
        "employee": {
            **_person(employee, department.name if department else None),
            "employee_number": employee.employee_number,
            "manager": None if manager is None else _person(manager),
            "approval_manager": None if approval_manager is None else _person(approval_manager),
            "routing_consistent": employee.manager_id == employee.approval_manager_id,
        },
        "balance": balance,
        "requested_window": requested,
        "recommended_window": alternative,
        "rules_applied": [
            f"Annual entitlement used by the workflow is {ANNUAL_ENTITLEMENT} days, matching Annual Leave Policy 3.0.",
            f"Notice is {NOTICE_SHORT} business days up to 5 days and {NOTICE_LONG} business days above that.",
            f"People Operations co-approval applies above {HR_DAY_THRESHOLD} consecutive business days or when a critical deadline sits inside the absence.",
            "The engineering addendum is not applied while it conflicts with the global approval policy.",
            "The signed-in employee is taken from the session. The model cannot choose a different employee.",
        ],
    }


def _week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


def analyze_team(db: Session, manager: Employee) -> dict:
    members = team_members(db, manager.id)
    member_ids = [member.id for member in members]
    today = org_today()
    horizon = today + timedelta(days=75)
    requests = (
        db.query(LeaveRequest, Employee)
        .join(Employee, Employee.id == LeaveRequest.employee_id)
        .filter(
            LeaveRequest.employee_id.in_(member_ids or [0]),
            LeaveRequest.status.in_(OPEN_STATUSES),
            LeaveRequest.end_date >= today,
            LeaveRequest.start_date <= horizon,
        )
        .all()
    )
    clusters: dict[date, list] = {}
    for request, employee in requests:
        cursor = max(request.start_date, today)
        while cursor <= request.end_date:
            if cursor.weekday() < 5:
                key = _week_start(cursor)
                clusters.setdefault(key, {})
                clusters[key][request.id] = (request, employee)
            cursor += timedelta(days=1)
    ranked = sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))
    if ranked:
        week, bucket = ranked[0]
    else:
        week, bucket = _week_start(today + timedelta(days=30)), {}
    week_end = week + timedelta(days=4)
    selected = []
    for request, employee in bucket.values():
        selected.append(
            {
                "request_id": request.id,
                "employee_id": employee.id,
                "full_name": employee.full_name,
                "job_title": employee.job_title,
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "business_days": request.business_days,
                "status": request.status,
            }
        )
    selected.sort(key=lambda item: (-item["business_days"], item["full_name"]))
    out_ids = {item["employee_id"] for item in selected}
    team_size = len(members)
    available = team_size - len(out_ids)
    available_pct = round(available / team_size, 4) if team_size else 1
    project_hits = []
    seen_projects = set()
    for member in members:
        if member.id not in out_ids and member.id != manager.id:
            continue
        for project in project_rows(db, member.id):
            deadline = date.fromisoformat(project["deadline"])
            if week <= deadline <= week_end + timedelta(days=2) and project["code"] not in seen_projects:
                seen_projects.add(project["code"])
                project_hits.append({**project, "critical_conflict": project["criticality"] == "critical"})
    critical = any(item["criticality"] == "critical" for item in project_hits)
    defer = []
    keep = list(selected)
    simulated_out = set(out_ids)
    target_available = int(COVERAGE_THRESHOLD * team_size + 0.999)
    while team_size - len(simulated_out) < target_available and keep:
        longest = keep.pop(0)
        defer.append(longest)
        simulated_out.discard(longest["employee_id"])
    can_approve = keep
    final_available = team_size - (len(out_ids) - len(defer))
    final_pct = round(final_available / team_size, 4) if team_size else 1
    alternatives = []
    if defer:
        longest = defer[0]
        owner = db.get(Employee, longest["employee_id"])
        if owner is not None:
            suggestion = recommend_window(db, owner, int(longest["business_days"]), avoid_start=date.fromisoformat(longest["start_date"]))
            if suggestion and not suggestion["hard_blocks"]:
                alternatives.append(
                    f"Move {longest['full_name']} to {suggestion['label']}, which scores as {suggestion['risk_level']} risk."
                )
    if critical:
        alternatives.append("Keep at least one senior contributor assigned to the critical deadline in the office that week.")
    alternatives.append("Do not split a decision by seniority or tenure. This recommendation uses request length and the coverage threshold only.")
    risk_level = "high" if critical or available_pct < COVERAGE_THRESHOLD else "medium" if selected else "low"
    return {
        "manager": _person(manager),
        "team_size": team_size,
        "window": {"start": week.isoformat(), "end": week_end.isoformat(), "label": f"week of {format_day(week)}"},
        "requests": selected,
        "available_now": available,
        "available_pct": available_pct,
        "projects": project_hits,
        "critical_deadline": critical,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "recommendation": {
            "principle": "Defer the longest overlapping requests until team availability reaches 70%, then review any remaining critical-deadline exposure.",
            "defer": defer,
            "can_consider": can_approve,
            "projected_available_pct": final_pct,
            "alternatives": alternatives,
        },
        "risk_level": risk_level,
        "rules_applied": [
            "Counts come from leave requests and project deadlines in the simulated HRIS.",
            "Availability is team size minus distinct people with overlapping approved or pending leave.",
            "No likelihood, performance, or personal circumstance is inferred.",
        ],
    }


def investigate_routing(db: Session, employee: Employee) -> dict:
    department = db.get(Department, employee.department_id)
    prior = db.get(Department, employee.prior_department_id) if employee.prior_department_id else None
    manager = db.get(Employee, employee.manager_id) if employee.manager_id else None
    approval_manager = db.get(Employee, employee.approval_manager_id) if employee.approval_manager_id else None
    consistent = employee.manager_id == employee.approval_manager_id and employee.manager_id is not None
    open_requests = (
        db.query(LeaveRequest)
        .filter(LeaveRequest.employee_id == employee.id, LeaveRequest.status.in_(("pending_manager", "pending_hr")))
        .all()
    )
    misrouted = []
    for request in open_requests:
        if request.routed_manager_id and request.routed_manager_id != employee.manager_id:
            routed = db.get(Employee, request.routed_manager_id)
            misrouted.append(
                {
                    "request_id": request.id,
                    "start_date": request.start_date.isoformat(),
                    "end_date": request.end_date.isoformat(),
                    "status": request.status,
                    "routed_to": None if routed is None else routed.full_name,
                }
            )
    issue = None
    if not consistent:
        current_name = manager.full_name if manager else "nobody"
        approval_name = approval_manager.full_name if approval_manager else "nobody"
        issue = (
            f"{employee.full_name} currently reports to {current_name}, but leave approval is still routed to {approval_name}."
        )
    return {
        "employee": _person(employee, department.name if department else None),
        "employee_number": employee.employee_number,
        "current_department": None if department is None else department.name,
        "prior_department": None if prior is None else prior.name,
        "transfer_effective_date": employee.transfer_effective_date.isoformat() if employee.transfer_effective_date else None,
        "current_manager": None if manager is None else _person(manager),
        "approval_manager": None if approval_manager is None else _person(approval_manager),
        "consistent": consistent,
        "issue": issue,
        "open_requests_following_old_route": misrouted,
        "correction": None
        if consistent or manager is None or approval_manager is None
        else {
            "field": "approval_manager_id",
            "from_employee_id": approval_manager.id,
            "from_name": approval_manager.full_name,
            "to_employee_id": manager.id,
            "to_name": manager.full_name,
        },
        "rules_applied": [
            "Official routing uses approval_manager_id, not the chat message.",
            "A mismatch with manager_id is a data defect.",
            "Only an HR administrator can change the official record, after a human approval.",
        ],
    }
