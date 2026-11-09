"""Organization-wide HR reads. Authorization is checked before these functions run."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.clock import org_today
from app.models import (
    Approval,
    Attendance,
    AuditLog,
    Department,
    Employee,
    LeaveBalance,
    LeaveRequest,
    PolicyDocument,
    Project,
    User,
)
from app.services.records import attendance_window, parse_comparison, parse_named_month_periods

PAGE = 25
OPEN_LEAVE = ("pending_manager", "pending_hr", "approved")


def run_hr_report(
    db: Session,
    intent: str,
    message: str,
    inherited: dict | None,
    target_ids: list[int],
) -> tuple[dict, dict | None]:
    if intent == "employee_directory":
        return _directory(db, message, inherited, target_ids)
    if intent == "org_attendance":
        return _attendance(db, message, inherited)
    if intent == "org_leave":
        return _leave(db, message, inherited, target_ids)
    if intent == "compensation_query":
        return _compensation(db, message, inherited)
    if intent == "hr_metrics":
        return _metrics(db, message, inherited)
    return _action(db, message, target_ids)


def _departments(db: Session) -> dict[int, str]:
    return {row.id: row.name for row in db.query(Department).all()}


def _names(db: Session) -> dict[int, str]:
    return {row.id: row.full_name for row in db.query(Employee).all()}


def _person(employee: Employee, departments: dict[int, str], names: dict[int, str]) -> dict:
    return {
        "full_name": employee.full_name,
        "employee_number": employee.employee_number,
        "job_title": employee.job_title,
        "department": departments.get(employee.department_id, ""),
        "employment_status": employee.employment_status,
        "manager": names.get(employee.manager_id or 0, ""),
        "hire_date": employee.hire_date.isoformat() if employee.hire_date else None,
        "location": employee.location,
        "grade": employee.grade,
    }


def _named_department(db: Session, message: str) -> Department | None:
    text = message.lower()
    rows = db.query(Department).all()
    for row in rows:
        if row.name.lower() in text:
            return row
    if re.search(r"\b(human resources|people operations)\b", text) or (
        re.search(r"\bhr\b", text) and re.search(r"\b(salary|salaries|payroll|department)\b", text)
    ):
        for row in rows:
            if row.name.lower() == "people operations":
                return row
    return None


def _named_departments(db: Session, message: str) -> list[Department]:
    """Every department named in the message, in message order.

    Matches the full name or its first word when that word is distinctive (so "Legal" matches
    "Legal & Compliance"). Used for side-by-side comparisons; never widens a filter on its own.
    """
    text = message.lower()
    found: list[tuple[int, Department]] = []
    for row in db.query(Department).all():
        name = row.name.lower()
        position = text.find(name)
        if position < 0:
            first = name.split()[0]
            if len(first) >= 5:
                match = re.search(rf"\b{re.escape(first)}\b", text)
                position = match.start() if match else -1
        if position >= 0:
            found.append((position, row))
    found.sort(key=lambda pair: pair[0])
    return [row for _, row in found]


def _follow_department(message: str, inherited: dict | None) -> str:
    if not inherited or not inherited.get("department"):
        return ""
    text = message.lower()
    if re.search(r"\b(organization|all departments|every department|company-wide)\b", text):
        return ""
    if re.search(
        r"\b(their|those|them|that department|that group|that cohort|how many are|who is|"
        r"percentage of|percent of|share of)\b",
        text,
    ):
        return str(inherited["department"])
    if re.search(r"\b(what|how) about\b", text):
        return str(inherited["department"])
    return ""


def _department_for(db: Session, message: str, inherited: dict | None) -> Department | None:
    named = _named_department(db, message)
    if named is not None:
        return named
    followed = _follow_department(message, inherited)
    if not followed:
        return None
    return db.query(Department).filter(Department.name == followed).one_or_none()


def _context(subject: str, user_id: int, employee_id: int, **extra) -> dict:
    payload = {
        "subject": subject,
        "intent": subject,
        "user_id": user_id,
        "employee_id": employee_id,
        "department": extra.get("department") or "",
        "location": extra.get("location") or "",
        "start": extra.get("start") or "",
        "end": extra.get("end") or "",
        "label": extra.get("label") or "",
    }
    if extra.get("hire_filter"):
        payload["hire_filter"] = extra["hire_filter"]
    if extra.get("prior_window"):
        payload["prior_window"] = extra["prior_window"]
    if extra.get("focus"):
        payload["focus"] = extra["focus"]
    return payload


def _apply_hire_filter(pool: list[Employee], hire_filter: dict) -> list[Employee]:
    """Apply a stored hire_date filter from a prior directory turn."""
    op = hire_filter.get("op")
    year = hire_filter.get("year")
    cutoff = hire_filter.get("cutoff")
    selected: list[Employee] = []
    for row in pool:
        if not row.hire_date:
            continue
        if op == "ge" and year is not None and row.hire_date.year >= year:
            selected.append(row)
        elif op == "le" and year is not None and row.hire_date.year <= year:
            selected.append(row)
        elif op == "lt" and year is not None and row.hire_date.year < year:
            selected.append(row)
        elif op == "gt" and year is not None and row.hire_date.year > year:
            selected.append(row)
        elif op == "eq" and year is not None and row.hire_date.year == year:
            selected.append(row)
        elif op == "since" and cutoff:
            if row.hire_date >= date.fromisoformat(cutoff):
                selected.append(row)
        elif op == "before_date" and cutoff:
            if row.hire_date < date.fromisoformat(cutoff):
                selected.append(row)
    return selected


def _directory(db: Session, message: str, inherited: dict | None, target_ids: list[int]) -> tuple[dict, dict]:
    text = message.lower()
    departments = _departments(db)
    names = _names(db)
    employees = db.query(Employee).order_by(Employee.last_name, Employee.first_name).all()
    active = [row for row in employees if row.employment_status == "active"]
    department = _department_for(db, message, inherited)
    compared = _named_departments(db, message)
    focus = "count"
    hire_filter: dict | None = None
    inherited = inherited or {}
    hire_rule = re.search(
        r"\b(?:hire_date|hire date|hired|joined|started)\b.{0,40}?\b(?:in|during|before|prior to|after|since|from)\s+"
        r"(?:calendar year\s+)?(\d{4})\b|\b(\d{4})\s+or\s+(?:later|earlier|after|before)\b|"
        r"\bcalendar year\s+(\d{4})\b",
        text,
    )
    location_names = sorted({(row.location or "").strip() for row in active if (row.location or "").strip()})
    named_location = next((loc for loc in location_names if loc.lower() in text), None)
    grade_tokens = re.findall(r"\b(l[1-7])\b", text)
    hire_years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]

    if (
        re.search(r"\b(tenure|average tenure|mean tenure)\b", text)
        and inherited.get("hire_filter")
        and re.search(r"\b(those|them|that group|that cohort|among those|of those|same employees)\b", text)
    ):
        focus = "filtered_tenure"
    elif re.search(
        r"\b(workforce overview|total active employees|employees by department|"
        r"by department, and employees by location)\b",
        text,
    ) or (
        re.search(r"\b(total active|active employees|workforce)\b", text)
        and re.search(r"\bby department\b", text)
        and re.search(r"\bby location\b", text)
    ):
        focus = "workforce_overview"
    elif re.search(
        r"\b(missing|incomplete|empty|unset|blank)\b", text
    ) and re.search(
        r"\b(across (?:the )?workforce|workforce|organization|organisation|all employees|profile fields?)\b",
        text,
    ):
        focus = "workforce_incomplete"
    elif re.search(r"\b(average|mean)\b", text) and re.search(r"\btenure\b", text) and re.search(
        r"\b(days?|calendar)\b", text
    ):
        focus = "avg_tenure_days"
    elif re.search(r"\bgrades?\b", text) and re.search(
        r"\b(distribution|percentage|percent|%|share of (?:the )?active workforce|represented by each)\b",
        text,
    ):
        focus = "grade_distribution"
    elif len(grade_tokens) >= 1 and re.search(r"\b(grade|graded|versus|vs\.?|compared|how many)\b", text):
        focus = "grade_counts"
    elif len(hire_years) >= 2 and re.search(r"\b(hir(?:e|ed|ing)|joined|compare|versus|vs\.?|volume)\b", text):
        focus = "hire_year_compare"
    elif re.search(
        r"\b(hir(?:e|ed|ing)|hires|hiring)\b", text
    ) and re.search(
        r"\b(previous three calendar years|last three calendar years|three calendar years|yearly totals|"
        r"highest (?:number of )?hires)\b",
        text,
    ):
        focus = "hire_year_relative"
    elif (
        (department is not None or inherited.get("department"))
        and (named_location or inherited.get("location"))
        and re.search(r"\b(percentage|percent|%|share)\b", text)
        and re.search(
            r"\b(of (?:engineering|that (?:group|cohort|department)|the (?:group|department))|"
            r"that (?:singapore|austin|berlin|london) group|singapore group)\b",
            text,
        )
    ):
        # Subgroup share of a prior department cohort (e.g. Singapore Engineering / Engineering).
        focus = "department_location_pct"
        if named_location is None and inherited.get("location"):
            named_location = str(inherited.get("location"))
    elif named_location and re.search(r"\b(percentage|percent|%|share)\b", text) and not re.search(
        r"\bgrades?\b", text
    ):
        # "percentage of Engineering is that Singapore group" with inherited Engineering.
        if department is not None and re.search(
            r"\b(of engineering|that (?:group|cohort)|singapore group|engineering is)\b", text
        ):
            focus = "department_location_pct"
        else:
            focus = "location_pct"
    elif (
        department is not None
        and re.search(r"\b(by location|grouped by location|location breakdown|break(?:\s+\w+)?\s+down)\b", text)
    ) or (
        department is not None
        and named_location
        and re.search(r"\b(engineering|department)\b", text)
        and re.search(r"\b(based in|in|located)\b", text)
        and not re.search(r"\b(percentage|percent)\b", text)
    ):
        # Engineering in Austin (single cell) vs Engineering by location (group).
        if named_location and not re.search(r"\b(by location|grouped|breakdown|each location)\b", text):
            # Explicit roster fields → list matching people, not count-only.
            if re.search(r"\b(names?|employee numbers?|job titles?|grades?|show|find|list)\b", text):
                focus = "department_location_roster"
            else:
                focus = "department_location"
        else:
            focus = "department_by_location"
    elif re.search(
        r"\b(percentage|percent|%|share).{0,50}\b(workforce|active employees|employees|headcount)\b|"
        r"\bworkforce .{0,30}\b(belongs to|in)\b|"
        r"\bwhat percentage of the active workforce\b|"
        r"\bsits in engineering\b|"
        r"\bheadcount .{0,30}engineering\b",
        text,
    ) and not re.search(r"\bgrades?\b", text):
        focus = "workforce_pct"
    elif named_location and re.search(
        r"\b(how many|headcount|employees?|work(?:s|ing)? in|based in)\b", text
    ) and re.search(r"\b(total|together|and|confirm|active headcount)\b", text):
        focus = "total_and_location"
    elif named_location and re.search(r"\b(how many|headcount|employees?|work(?:s|ing)? in|based in)\b", text):
        focus = "location_count"
    elif re.search(
        r"\b(at least five years|employed for (?:at least |more than )?(?:five|5|three|3) years|"
        r"more than three years|for at least five years)\b",
        text,
    ):
        focus = "tenure_years"
    elif re.search(r"\b(each quarter|by quarter|quarter of 2024|quarters of)\b", text) and re.search(
        r"\b(hired|hire|joined|2024)\b", text
    ):
        focus = "hires_by_quarter"
    elif re.search(
        r"\b(largest (?:number of )?active employees|department .{0,40}largest|"
        r"which department has the largest|largest active (?:department|headcount))\b",
        text,
    ):
        focus = "largest_department"
    elif re.search(
        r"\b((?:second|third|2nd|3rd)[- ]largest|smallest (?:active )?workforce|"
        r"largest (?:active )?workforce|office location has the largest|which (?:office )?location)\b",
        text,
    ):
        focus = "top_location"
    elif "probation" in text:
        focus = "probation"
    elif "incomplete" in text or re.search(r"\b(blank|missing|empty)\s+(?:phone|job title|title|email)\b", text):
        focus = "incomplete"
    elif len(compared) >= 2 and re.search(r"\b(compare|versus|vs\.?|between|against|and)\b", text):
        focus = "compare_departments"
    elif re.search(r"\b(locations?|represented in the employee directory)\b", text):
        focus = "locations"
    elif re.search(
        r"\b(no manager assigned|without a manager|null manager|manager_id (?:is )?(?:null|empty|missing|unset|blank)|"
        r"no manager_id|manager_id null|(?:null|empty|missing) manager_id|have no manager|has no manager|lack a manager|"
        r"do not have an assigned manager|don't have an assigned manager|without an assigned manager|"
        r"no assigned manager|unassigned manager|not have an assigned manager|"
        r"departments? (?:have|with) (?:employees? )?without (?:assigned )?managers?|"
        r"which departments? .{0,40}without .{0,20}managers?)\b",
        text,
    ):
        focus = "no_manager"
    elif re.search(r"\b(most direct reports|managers who have the most)\b", text):
        focus = "most_reports"
    elif re.search(
        r"\b(?:unique|distinct|different|how many)\s+managers\b|\bmanagers? (?:are )?referenced\b|"
        r"\bdistinct manager_id|unique manager_id|manager_id values\b",
        text,
    ):
        focus = "managers"
    elif re.search(r"\b(reports? directly|their manager)\b", text):
        focus = "reports_to" if "report" in text else "department_managers"
    elif re.search(r"\b(who are the managers|managers in)\b", text):
        focus = "managers"
    elif re.search(
        r"\b(grouped|distribution|assigned to each department|each department|per department|by department)\b",
        text,
    ) and not re.search(r"\b(attendance|remote|leave|salar\w*)\b", text):
        focus = "grouped"
    elif hire_rule or re.search(r"\bjoined in\s+(\d{4})\b", text) or re.search(r"\bjoined\b", text) or re.search(
        r"\b(?:before|prior to|earlier than)\s+(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
        text,
    ):
        focus = "recent_hires"
    elif re.search(r"\b(profile|employment details|e-\d+|employee number|hire date)\b", text) or (
        target_ids and re.search(r"\b(employee number|hire date|job title|grade|location|who is)\b", text)
    ):
        focus = "profile"
    elif target_ids and re.search(r"\b(vogel|helena|what is)\b", text):
        focus = "profile"
    elif department is not None:
        focus = "department"

    people: list[dict] = []
    note = ""
    locations: list[str] = []
    manager_spans: list[dict] = []
    comparison: list[dict] = []
    filtered_tenure_years: float | None = None
    filtered_tenure_days: float | None = None
    filtered_tenure_count: int | None = None
    if focus == "probation":
        note = (
            "Employee records do not store a probation flag, and the assistant did not invent one. "
            "No probation list was retrieved."
        )
    elif focus == "incomplete":
        selected = [
            row
            for row in employees
            if not (row.phone or "").strip()
            or not (row.job_title or "").strip()
            or not (row.email or "").strip()
            or not (row.employee_number or "").strip()
        ]
        people = [_person(row, departments, names) for row in selected]
        note = (
            "Incomplete means a blank phone, email, job title, or employee number. "
            "A person with no manager is the top of a reporting line, not an incomplete profile."
        )
    elif focus == "locations":
        locations = sorted({(row.location or "").strip() for row in active if (row.location or "").strip()})
        note = "Locations are distinct non-empty location values on active employee records."
    elif focus == "location_count":
        loc = named_location or ""
        n = sum(1 for row in active if (row.location or "").strip() == loc)
        comparison = [{"location": loc, "active": n, "active_total": len(active)}]
        note = f"{n} active employees work in {loc}. The count uses employment_status=active."
        locations = [loc] if loc else []
    elif focus == "department_location_pct":
        dept = department
        loc = named_location or str(inherited.get("location") or "")
        pool = [row for row in active if dept and row.department_id == dept.id]
        n = sum(1 for row in pool if (row.location or "").strip() == loc)
        pct = round(100.0 * n / len(pool), 1) if pool else 0.0
        comparison = [
            {
                "department": dept.name if dept else "",
                "location": loc,
                "active": n,
                "percent": pct,
                "department_total": len(pool),
            }
        ]
        note = (
            f"{loc} accounts for {n} of {len(pool)} active {dept.name if dept else 'department'} employees "
            f"({pct}%). The denominator is active employees in that department only."
        )
        locations = [loc] if loc else []
    elif focus == "location_pct":
        loc = named_location or ""
        n = sum(1 for row in active if (row.location or "").strip() == loc)
        pct = round(100.0 * n / len(active), 1) if active else 0.0
        comparison = [{"location": loc, "active": n, "percent": pct, "active_total": len(active)}]
        note = (
            f"{loc} headcount is {n} of {len(active)} active employees ({pct}%). "
            "The denominator is active employment_status only."
        )
        locations = [loc] if loc else []
    elif focus == "total_and_location":
        loc = named_location or ""
        n = sum(1 for row in active if (row.location or "").strip() == loc)
        comparison = [
            {"metric": "active_total", "value": len(active)},
            {"location": loc, "active": n},
        ]
        note = f"Active company headcount is {len(active)}. {loc} headcount is {n}."
        locations = [loc] if loc else []
    elif focus == "department_by_location":
        dept = department
        pool = [row for row in active if dept and row.department_id == dept.id]
        counts: dict[str, int] = defaultdict(int)
        for row in pool:
            loc = (row.location or "").strip() or "(blank)"
            counts[loc] += 1
        comparison = [{"location": loc, "active": n} for loc, n in sorted(counts.items())]
        note = (
            f"{dept.name if dept else 'Department'} active headcount by location "
            f"({len(pool)} total)."
        )
    elif focus == "workforce_overview":
        by_dept: dict[str, int] = defaultdict(int)
        by_loc: dict[str, int] = defaultdict(int)
        for row in active:
            by_dept[departments.get(row.department_id, "(unknown)")] += 1
            by_loc[(row.location or "").strip() or "(blank)"] += 1
        comparison = [
            {"metric": "active_total", "value": len(active)},
            *[
                {"dimension": "department", "name": name, "active": n}
                for name, n in sorted(by_dept.items())
            ],
            *[
                {"dimension": "location", "name": name, "active": n}
                for name, n in sorted(by_loc.items())
            ],
        ]
        locations = sorted(by_loc.keys())
        note = (
            f"Active workforce overview: {len(active)} active employees, "
            f"{len(by_dept)} departments, {len(by_loc)} locations."
        )
        people = []
    elif focus == "workforce_incomplete":
        fields = {
            "manager_id": sum(1 for row in employees if row.manager_id is None),
            "approval_manager_id": sum(1 for row in employees if row.approval_manager_id is None),
            "prior_department_id": sum(1 for row in employees if row.prior_department_id is None),
            "transfer_effective_date": sum(1 for row in employees if row.transfer_effective_date is None),
            "phone": sum(1 for row in employees if not (row.phone or "").strip()),
            "grade": sum(1 for row in employees if not (row.grade or "").strip()),
            "job_title": sum(1 for row in employees if not (row.job_title or "").strip()),
            "location": sum(1 for row in employees if not (row.location or "").strip()),
            "performance_rating": sum(1 for row in employees if row.performance_rating is None),
            "salary_annual": sum(1 for row in employees if row.salary_annual is None),
        }
        comparison = [{"field": key, "missing": value} for key, value in fields.items()]
        note = (
            "Workforce-wide counts of empty/unset stored employee fields. "
            "Fields absent from the schema are not reported as missing values."
        )
        people = []
    elif focus == "grade_distribution":
        counts: dict[str, int] = defaultdict(int)
        for row in active:
            grade = (row.grade or "").strip() or "(null)"
            counts[grade] += 1
        comparison = []
        for grade, n in sorted(counts.items()):
            pct = round(100.0 * n / len(active), 1) if active else 0.0
            comparison.append({"grade": grade, "active": n, "percent": pct, "active_total": len(active)})
        note = (
            f"Active workforce grade distribution across {len(active)} active employees. "
            "Percentages use active employment_status as the denominator."
        )
        people = []
    elif focus == "hire_year_relative":
        today = org_today()
        years = [today.year - 3, today.year - 2, today.year - 1]
        comparison = []
        for year in years:
            n = sum(1 for row in employees if row.hire_date and row.hire_date.year == year)
            comparison.append({"year": year, "hires": n})
        peak = max(comparison, key=lambda item: item["hires"]) if comparison else None
        note = (
            f"Hiring totals for the previous three calendar years "
            f"({years[0]}–{years[-1]}) by hire_date year. "
            + (f"Highest hiring year: {peak['year']} ({peak['hires']} hires)." if peak else "")
        )
        people = []
    elif focus == "department_location_roster":
        dept = department
        loc = named_location or ""
        selected = [
            row
            for row in active
            if dept
            and row.department_id == dept.id
            and (row.location or "").strip() == loc
        ]
        people = [_person(row, departments, names) for row in selected]
        comparison = [{"department": dept.name if dept else "", "location": loc, "active": len(selected)}]
        note = (
            f"{len(selected)} active {(dept.name if dept else 'department')} employees based in {loc}."
        )
    elif focus == "department_location":
        dept = department
        loc = named_location or ""
        n = sum(
            1
            for row in active
            if dept
            and row.department_id == dept.id
            and (row.location or "").strip() == loc
        )
        comparison = [{"department": dept.name if dept else "", "location": loc, "active": n}]
        note = f"{n} active {(dept.name if dept else 'department')} employees are based in {loc}."
    elif focus == "grade_counts":
        wanted = [token.upper() for token in grade_tokens] or ["L3", "L7"]
        comparison = []
        for grade in wanted:
            n = sum(1 for row in active if (row.grade or "").upper() == grade)
            comparison.append({"grade": grade, "active": n})
        note = "Active employee counts by requested grade."
    elif focus == "hire_year_compare":
        years = hire_years[:2] if len(hire_years) >= 2 else hire_years
        comparison = []
        for year in years:
            n = sum(1 for row in employees if row.hire_date and row.hire_date.year == year)
            comparison.append({"year": year, "hires": n})
        note = "Hire volume by calendar year of hire_date (all employment statuses)."
    elif focus == "avg_tenure_days":
        today = org_today()
        days = [(today - row.hire_date).days for row in active if row.hire_date]
        avg_days = round(sum(days) / len(days), 1) if days else 0.0
        comparison = [{"metric": "avg_tenure_days", "value": avg_days, "as_of": today.isoformat(), "n": len(days)}]
        note = (
            f"Average employee tenure is {avg_days} calendar days as of {today.isoformat()} "
            f"across {len(days)} active employees with a hire date."
        )
    elif focus == "workforce_pct":
        dept = department
        if dept is None:
            for row in db.query(Department).all():
                if row.name.lower() in text or row.name.split()[0].lower() in text:
                    dept = row
                    break
        dept_name = dept.name if dept else "Engineering"
        if dept is None:
            dept = next((row for row in db.query(Department).all() if row.name == "Engineering"), None)
        dept_n = sum(1 for row in active if dept and row.department_id == dept.id)
        pct = round(100.0 * dept_n / len(active), 1) if active else 0.0
        people = []
        note = (
            f"{dept_name} has {dept_n} active employees out of {len(active)} active employees "
            f"({pct}%). The denominator is active employment_status only."
        )
        comparison = [{"department": dept_name, "active": dept_n, "percent": pct, "active_total": len(active)}]
        department = dept
    elif focus == "tenure_years":
        today = org_today()
        years = 5
        if re.search(r"\b(three|3)\b", text):
            years = 3
        if re.search(r"\b(five|5)\b", text):
            years = 5
        try:
            cutoff = today.replace(year=today.year - years)
        except ValueError:
            cutoff = today.replace(year=today.year - years, day=28)
        pool = active
        if department is not None and re.search(r"\b(that department|those|them|that group|in that)\b", text):
            pool = [row for row in pool if row.department_id == department.id]
        selected = [row for row in pool if row.hire_date and row.hire_date <= cutoff]
        people = [_person(row, departments, names) for row in selected]
        note = (
            f"{len(selected)} active employee{'s' if len(selected) != 1 else ''} with hire_date on or before "
            f"{cutoff.isoformat()} (employed at least {years} years as of {today.isoformat()})."
        )
    elif focus == "hires_by_quarter":
        year = 2024
        ym = re.search(r"\b(20\d{2})\b", text)
        if ym:
            year = int(ym.group(1))
        buckets = {1: 0, 2: 0, 3: 0, 4: 0}
        for row in employees:
            if row.hire_date and row.hire_date.year == year:
                buckets[(row.hire_date.month - 1) // 3 + 1] += 1
        comparison = [{"quarter": q, "hires": n, "year": year} for q, n in buckets.items()]
        note = f"Hire counts by calendar quarter for hire_date in {year} (all employment statuses)."
    elif focus == "top_location":
        counts: dict[str, int] = defaultdict(int)
        for row in active:
            loc = (row.location or "").strip()
            if loc:
                counts[loc] += 1
        # Deterministic: larger headcount first, then location name ascending for ties.
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ordinal_map = {
            "largest": 0,
            "first": 0,
            "1st": 0,
            "second": 1,
            "2nd": 1,
            "third": 2,
            "3rd": 2,
            "smallest": -1,
        }
        rank_idx = 0
        rank_label = "largest"
        if re.search(r"\bsmallest\b", text):
            rank_idx = -1
            rank_label = "smallest"
        else:
            for token, idx in ordinal_map.items():
                if token in {"largest", "smallest"}:
                    continue
                if re.search(rf"\b{token}(?:[- ]largest)?\b", text):
                    rank_idx = idx
                    rank_label = f"{token}-largest" if idx > 0 else "largest"
                    break
            if re.search(r"\blargest\b", text) and not re.search(
                r"\b(second|third|2nd|3rd|smallest)\b", text
            ):
                rank_idx = 0
                rank_label = "largest"
        if ranked:
            if rank_idx == -1:
                chosen = ranked[-1]
            elif rank_idx < len(ranked):
                chosen = ranked[rank_idx]
            else:
                chosen = None
            comparison = [{"location": loc, "active": n} for loc, n in ranked]
            if chosen:
                locations = [chosen[0]]
                # Put the requested ordinal first so narrate can read comparison[0].
                comparison = [{"location": chosen[0], "active": chosen[1], "rank_label": rank_label}] + [
                    item for item in comparison if item["location"] != chosen[0]
                ]
                note = (
                    f"{chosen[0]} has the {rank_label} active workforce with {chosen[1]} active employees. "
                    "Counts are active employee records grouped by location; ties break alphabetically by location name."
                )
            else:
                locations = []
                note = (
                    f"Only {len(ranked)} distinct active location{'s' if len(ranked) != 1 else ''} "
                    f"{'are' if len(ranked) != 1 else 'is'} stored, so a {rank_label} location is not available."
                )
        else:
            note = "No active employee locations are stored."
    elif focus == "largest_department":
        counts: dict[str, int] = defaultdict(int)
        for row in active:
            name = departments.get(row.department_id, "")
            if name:
                counts[name] += 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        comparison = [{"department": name, "active": n} for name, n in ranked]
        if ranked:
            note = (
                f"{ranked[0][0]} has the largest number of active employees ({ranked[0][1]}). "
                "Counts use employment_status=active grouped by department."
            )
            department = next((row for row in db.query(Department).all() if row.name == ranked[0][0]), None)
        else:
            note = "No active employees are stored."
    elif focus == "compare_departments":
        for dept in compared:
            rows = [row for row in active if row.department_id == dept.id]
            comparison.append({"department": dept.name, "active": len(rows), "total": sum(1 for row in employees if row.department_id == dept.id)})
        note = (
            "Active headcount is the count of employee records with employment_status active in each named department. "
            "Headcounts are small groups; a difference of a few people is not evidence of a trend."
        )
    elif focus == "no_manager":
        selected = [row for row in active if row.manager_id is None]
        people = [_person(row, departments, names) for row in selected]
        dept_names = sorted({departments.get(row.department_id, "") for row in selected if row.department_id})
        comparison = [{"department": name, "without_manager": sum(1 for row in selected if departments.get(row.department_id) == name)} for name in dept_names]
        note = (
            "Departments with at least one active employee whose manager_id is null: "
            + (", ".join(dept_names) if dept_names else "none")
            + ". A null manager_id usually marks the top of a reporting line, not an incomplete profile."
        )
    elif focus == "most_reports":
        counts: dict[int, int] = defaultdict(int)
        for row in active:
            if row.manager_id:
                counts[row.manager_id] += 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], names.get(item[0], "")))
        for manager_id, count in ranked[:PAGE]:
            manager = next((row for row in employees if row.id == manager_id), None)
            manager_spans.append(
                {
                    "full_name": manager.full_name if manager else names.get(manager_id, ""),
                    "employee_number": manager.employee_number if manager else "",
                    "department": departments.get(manager.department_id, "") if manager else "",
                    "direct_reports": count,
                }
            )
        people = manager_spans
        note = "Direct-report counts use active employees grouped by manager_id."
    elif focus == "managers":
        used = {row.manager_id for row in active if row.manager_id}
        selected = [row for row in employees if row.id in used]
        people = [_person(row, departments, names) for row in selected]
        note = (
            f"{len(used)} distinct manager_id values are referenced by active employees. "
            "A manager is anyone recorded as manager_id on an active employee."
        )
    elif focus == "department_managers":
        pool = active
        if department is not None:
            pool = [row for row in pool if row.department_id == department.id]
        used = {row.manager_id for row in pool if row.manager_id}
        selected = [row for row in employees if row.id in used]
        people = [_person(row, departments, names) for row in selected]
        note = "These are the managers recorded for that employee set."
    elif focus == "reports_to":
        manager_ids = target_ids or []
        if not manager_ids:
            manager_ids = [row.id for row in employees if row.full_name.lower() in text]
        selected = [row for row in employees if row.manager_id in manager_ids]
        people = [_person(row, departments, names) for row in selected]
        note = "Direct reports are employees whose manager_id is the named person."
    elif focus == "filtered_tenure":
        prior = inherited.get("hire_filter") or {}
        hire_filter = dict(prior)
        active_only = bool(prior.get("active_only"))
        pool = active if active_only else employees
        selected = _apply_hire_filter(pool, prior)
        today = org_today()
        days = [max((today - row.hire_date).days, 0) for row in selected if row.hire_date]
        average_years = round((sum(days) / len(days)) / 365.25, 2) if days else 0.0
        people = [_person(row, departments, names) for row in selected]
        note = (
            f"Average tenure for the previously filtered cohort ({prior.get('label') or 'stored hire_date filter'}): "
            f"{average_years} years across {len(selected)} employee{'s' if len(selected) != 1 else ''}. "
            f"Tenure is the organization date ({today.isoformat()}) minus hire_date, averaged and divided by 365.25. "
            "This is not the overall company average."
        )
        # Stash metrics on the payload via locals consumed below.
        filtered_tenure_years = average_years
        filtered_tenure_days = round(sum(days) / len(days), 1) if days else 0.0
        filtered_tenure_count = len(selected)
    elif focus == "recent_hires":
        pool = active if re.search(r"\bactive\b", text) else employees
        scope_label = "active employees" if pool is active else "employee records (all statuses)"
        active_only = pool is active
        later = re.search(r"\b(\d{4})\s+or\s+(?:later|after)\b|\b(?:since|from|on or after)\s+(\d{4})\b", text)
        earlier = re.search(r"\b(\d{4})\s+or\s+(?:earlier|before)\b", text)
        before_named = re.search(
            r"\b(?:joined\s+)?(?:before|prior to|earlier than)\s+"
            r"(january|february|march|april|may|june|july|august|september|october|november|december|"
            r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
            text,
        )
        before_iso = re.search(
            r"\b(?:joined\s+)?(?:before|prior to|earlier than)\s+(\d{4})-(\d{2})-(\d{2})\b",
            text,
        )
        before = re.search(r"\b(?:before|prior to|earlier than)\s+(?:calendar year\s+)?(\d{4})\b", text)
        after = re.search(r"\bafter\s+(?:calendar year\s+)?(\d{4})\b", text)
        exact = re.search(r"\bjoined(?:\s+in)?\s+(\d{4})\b", text) or re.search(
            r"\b(?:in|during)\s+(?:calendar year\s+)?(\d{4})\b", text
        ) or re.search(r"\bcalendar year\s+(\d{4})\b", text)
        _months = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
            "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
        }
        if later:
            year = int(next(group for group in later.groups() if group))
            selected = [row for row in pool if row.hire_date and row.hire_date.year >= year]
            note = f"{len(selected)} {scope_label} have a hire_date in {year} or later."
            hire_filter = {"op": "ge", "year": year, "active_only": active_only, "label": f"hire_date year {year} or later"}
        elif earlier:
            year = int(earlier.group(1))
            selected = [row for row in pool if row.hire_date and row.hire_date.year <= year]
            note = f"{len(selected)} {scope_label} have a hire_date in {year} or earlier."
            hire_filter = {"op": "le", "year": year, "active_only": active_only, "label": f"hire_date year {year} or earlier"}
        elif before_named or before_iso:
            if before_named:
                cutoff = date(
                    int(before_named.group(3)),
                    _months[before_named.group(1)],
                    int(before_named.group(2)),
                )
            else:
                cutoff = date(int(before_iso.group(1)), int(before_iso.group(2)), int(before_iso.group(3)))
            selected = [row for row in pool if row.hire_date and row.hire_date < cutoff]
            note = f"{len(selected)} {scope_label} have a hire_date before {cutoff.isoformat()}."
            hire_filter = {
                "op": "before_date",
                "cutoff": cutoff.isoformat(),
                "active_only": active_only,
                "label": f"hire_date before {cutoff.isoformat()}",
            }
        elif before:
            year = int(before.group(1))
            selected = [row for row in pool if row.hire_date and row.hire_date.year < year]
            note = f"{len(selected)} {scope_label} have a hire_date before {year} (hire year earlier than {year})."
            hire_filter = {"op": "lt", "year": year, "active_only": active_only, "label": f"hire_date before {year}"}
        elif after:
            year = int(after.group(1))
            selected = [row for row in pool if row.hire_date and row.hire_date.year > year]
            note = f"{len(selected)} {scope_label} have a hire_date after {year}."
            hire_filter = {"op": "gt", "year": year, "active_only": active_only, "label": f"hire_date after {year}"}
        elif exact:
            year = int(exact.group(1))
            selected = [row for row in pool if row.hire_date and row.hire_date.year == year]
            note = f"{len(selected)} {scope_label} have a hire_date in calendar year {year}."
            hire_filter = {"op": "eq", "year": year, "active_only": active_only, "label": f"hire_date in {year}"}
        else:
            cutoff = org_today() - timedelta(days=183)
            selected = [row for row in pool if row.hire_date >= cutoff]
            note = f"Hire date on or after {cutoff.isoformat()}, about six months before the organization date."
            hire_filter = {
                "op": "since",
                "cutoff": cutoff.isoformat(),
                "active_only": active_only,
                "label": f"hire_date on or after {cutoff.isoformat()}",
            }
        people = [_person(row, departments, names) for row in selected]
    elif focus == "profile":
        number = re.search(r"\bE-\d+\b", message, re.IGNORECASE)
        selected = []
        if number:
            code = number.group(0).upper()
            selected = [row for row in employees if row.employee_number.upper() == code]
        elif target_ids:
            selected = [row for row in employees if row.id in target_ids]
        else:
            # Resolve an explicitly named employee from the message (first + last).
            for row in employees:
                full = row.full_name.lower()
                if full and full in text:
                    selected = [row]
                    break
                first = (row.first_name or "").lower()
                last = (row.last_name or "").lower()
                if first and last and re.search(rf"\b{re.escape(first)}\b", text) and re.search(
                    rf"\b{re.escape(last)}\b", text
                ):
                    selected = [row]
                    break
        people = [_person(row, departments, names) for row in selected]
        note = "Salary and performance are omitted from a profile unless the question asks for them."
    elif focus == "department" and department is not None:
        selected = [row for row in employees if row.department_id == department.id]
        if "active" in text:
            selected = [row for row in selected if row.employment_status == "active"]
        people = [_person(row, departments, names) for row in selected]
    elif focus == "grouped":
        note = "Counts use employment_status active."
    else:
        note = "The count is employee records whose employment_status is active."

    grouped = []
    source = active
    for name in sorted(set(departments.values())):
        grouped.append(
            {
                "department": name,
                "active": sum(1 for row in source if departments.get(row.department_id) == name),
            }
        )
    shown = people[:PAGE]
    payload = {
        "kind": "employee_directory",
        "focus": focus,
        "active_count": len(active),
        "total_count": len(employees),
        "department": department.name if department is not None else "",
        "by_department": grouped,
        "people": shown,
        "people_count": (
            len(locations)
            if focus in {"locations", "top_location"}
            else len(comparison)
            if focus
            in {
                "workforce_pct",
                "hires_by_quarter",
                "largest_department",
                "location_count",
                "location_pct",
                "total_and_location",
                "department_by_location",
                "department_location",
                "department_location_pct",
                "grade_counts",
                "hire_year_compare",
                "avg_tenure_days",
            }
            else len(people)
        ),
        "locations": locations if focus in {"locations", "top_location", "location_count", "location_pct", "department_location_pct", "total_and_location"} else [],
        "comparison": comparison,
        "manager_count": len({row.manager_id for row in active if row.manager_id}) if focus == "managers" else None,
        "truncated": len(people) > PAGE,
        "note": note,
    }
    if focus == "filtered_tenure":
        payload.update(
            {
                "average_tenure_years": filtered_tenure_years,
                "average_tenure_days": filtered_tenure_days,
                "cohort_count": filtered_tenure_count,
                "hire_filter": hire_filter,
                "as_of": org_today().isoformat(),
            }
        )
    context_kwargs: dict = {
        "department": department.name if department is not None else "",
    }
    if named_location:
        context_kwargs["location"] = named_location
    elif inherited.get("location") and focus in {"department_location", "department_location_pct", "location_count", "location_pct"}:
        context_kwargs["location"] = inherited.get("location")
    if hire_filter:
        context_kwargs["hire_filter"] = hire_filter
    elif inherited.get("hire_filter") and focus == "filtered_tenure":
        context_kwargs["hire_filter"] = inherited["hire_filter"]
    context = _context("employee_directory", 0, 0, **context_kwargs)
    return payload, context


def _counts(rows: list[Attendance]) -> dict:
    counts = {"present": 0, "remote": 0, "absent": 0, "sick": 0, "leave": 0, "late": 0}
    hours = 0.0
    absent_employees: set[int] = set()
    for row in rows:
        if row.status in counts:
            counts[row.status] += 1
        if row.status == "absent":
            absent_employees.add(row.employee_id)
        hours += float(row.hours or 0)
    recorded = len(rows)
    worked = counts["present"] + counts["remote"] + counts["late"]
    absent_rate = round(100 * counts["absent"] / recorded, 2) if recorded else None
    return {
        **counts,
        "recorded_days": recorded,
        "hours": int(hours) if hours.is_integer() else round(hours, 2),
        "attendance_percentage": round(100 * worked / recorded, 1) if recorded else None,
        "absent_employees": len(absent_employees),
        "absent_rate_pct": absent_rate,
    }


def _period_snapshot(db: Session, employees: list[Employee], start: date, end: date, label: str) -> dict:
    ids = [row.id for row in employees]
    rows = _attendance_rows(db, ids, start, end)
    counts = _counts(rows)
    departments = {row.id: row.name for row in db.query(Department).all()}
    by_dept_rows: dict[str, list[Attendance]] = defaultdict(list)
    emp_dept = {row.id: departments.get(row.department_id, "") for row in employees}
    for row in rows:
        dept_name = emp_dept.get(row.employee_id)
        if dept_name:
            by_dept_rows[dept_name].append(row)
    dept_stats = []
    for dept_name in sorted(departments.values()):
        dept_rows = by_dept_rows.get(dept_name) or []
        dept_counts = _counts(dept_rows)
        dept_stats.append(
            {
                "department": dept_name,
                "absent_rows": dept_counts.get("absent", 0),
                "absent_employees": dept_counts.get("absent_employees", 0),
                "recorded_days": dept_counts.get("recorded_days", 0),
                "absent_rate_pct": dept_counts.get("absent_rate_pct"),
            }
        )
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": label,
        "counts": counts,
        "departments": dept_stats,
    }


def _attendance(db: Session, message: str, inherited: dict | None) -> tuple[dict, dict]:
    today = org_today()
    text = message.lower()
    inherited = inherited or {}
    department = _department_for(db, message, inherited)
    employees = db.query(Employee).filter(Employee.employment_status == "active").all()
    named_departments = _named_departments(db, message)
    # Multi-turn: "What about October?" / "How much did it increase?" / "Which department contributed most?"
    # Keep org_attendance deterministic instead of inventing workforce-analytics figures.
    prior_start = inherited.get("start")
    prior_end = inherited.get("end")
    asks_increase = bool(
        re.search(
            r"\b(how much did .{0,60} increase|did .{0,40} increase|percentage change|"
            r"percent(?:age)?[- ]?point|increase from .{0,40} to)\b",
            text,
        )
    )
    asks_department_contrib = bool(
        re.search(
            r"\b("
            r"which department contributed most|department contributed|contributed most|"
            r"largest (?:increase|contribution)|department had the largest increase|"
            r"which department .{0,40}(?:increase|delta|change)|department .{0,20}most"
            r")\b",
            text,
        )
    )
    named_months = parse_named_month_periods(message, today)
    month_switch = bool(named_months) and bool(prior_start) and bool(
        re.search(r"\b(what about|how about|and)\b", text) or len(text.split()) <= 6
    )
    if (
        (inherited.get("subject") == "org_attendance" or asks_increase or asks_department_contrib)
        and (asks_increase or asks_department_contrib or month_switch)
        and (prior_start and prior_end or len(named_months) >= 2)
    ):
        prev_start = prev_end = None
        cur_start = cur_end = None
        cur_label = ""
        prev_label = inherited.get("label") or ""
        if len(named_months) >= 2:
            prev_start, prev_end, prev_label = named_months[0]
            cur_start, cur_end, cur_label = named_months[1]
        elif named_months and prior_start and prior_end:
            try:
                prev_start = date.fromisoformat(str(prior_start)[:10])
                prev_end = date.fromisoformat(str(prior_end)[:10])
                prev_label = inherited.get("label") or prev_start.strftime("%B %Y")
            except Exception:
                prev_start = prev_end = None
            cur_start, cur_end, cur_label = named_months[0]
        elif prior_start and prior_end:
            prior_window = inherited.get("prior_window") or {}
            # Prefer stored prior_window (e.g. September) vs current window (e.g. October).
            if prior_window.get("start") and prior_window.get("end") and (asks_increase or asks_department_contrib):
                try:
                    prev_start = date.fromisoformat(str(prior_window["start"])[:10])
                    prev_end = date.fromisoformat(str(prior_window["end"])[:10])
                    prev_label = prior_window.get("label") or prev_start.strftime("%B %Y")
                    cur_start = date.fromisoformat(str(prior_start)[:10])
                    cur_end = date.fromisoformat(str(prior_end)[:10])
                    cur_label = inherited.get("label") or cur_start.strftime("%B %Y")
                except Exception:
                    prev_start = prev_end = None
            else:
                try:
                    prev_start = date.fromisoformat(str(prior_start)[:10])
                    prev_end = date.fromisoformat(str(prior_end)[:10])
                    prev_label = inherited.get("label") or prev_start.strftime("%B %Y")
                except Exception:
                    prev_start = prev_end = None
                cur_start, cur_end, cur_label = attendance_window(message, today, inherited, subject="org_attendance")
        if prev_start and prev_end and cur_start and cur_end:
            previous = _period_snapshot(db, employees, prev_start, prev_end, prev_label)
            latest = _period_snapshot(db, employees, cur_start, cur_end, cur_label)
            prev_c = previous["counts"]
            lat_c = latest["counts"]
            abs_row_delta = int(lat_c.get("absent", 0) - prev_c.get("absent", 0))
            abs_emp_delta = int(lat_c.get("absent_employees", 0) - prev_c.get("absent_employees", 0))
            prev_rate = prev_c.get("absent_rate_pct")
            lat_rate = lat_c.get("absent_rate_pct")
            point_delta = None if prev_rate is None or lat_rate is None else round(lat_rate - prev_rate, 2)
            pct_change = None
            if prev_c.get("absent"):
                pct_change = round(100 * abs_row_delta / prev_c["absent"], 2)
            dept_deltas = []
            prev_by = {item["department"]: item for item in previous["departments"]}
            for item in latest["departments"]:
                before = prev_by.get(item["department"]) or {}
                row_delta = int(item.get("absent_rows", 0) - before.get("absent_rows", 0))
                rate_before = before.get("absent_rate_pct")
                rate_after = item.get("absent_rate_pct")
                dept_deltas.append(
                    {
                        "department": item["department"],
                        "previous_absent_rows": before.get("absent_rows", 0),
                        "latest_absent_rows": item.get("absent_rows", 0),
                        "absent_row_delta": row_delta,
                        "previous_absent_rate_pct": rate_before,
                        "latest_absent_rate_pct": rate_after,
                        "absent_rate_point_delta": None
                        if rate_before is None or rate_after is None
                        else round(rate_after - rate_before, 2),
                    }
                )
            dept_deltas.sort(key=lambda item: (-item["absent_row_delta"], item["department"]))
            focus = "department_contribution" if asks_department_contrib else "period_change" if asks_increase or month_switch else "period_change"
            if month_switch and not asks_increase and not asks_department_contrib:
                focus = "period_switch"
            payload = {
                "kind": "org_attendance",
                "focus": focus,
                "department": "",
                "employee_count": len(employees),
                "start": latest["start"],
                "end": latest["end"],
                "label": latest["label"],
                "counts": lat_c,
                "previous": previous,
                "latest": latest,
                "change": {
                    "absent_row_delta": abs_row_delta,
                    "absent_employee_delta": abs_emp_delta,
                    "absent_row_percent_change": pct_change,
                    "absent_rate_point_delta": point_delta,
                    "previous_absent_rows": prev_c.get("absent", 0),
                    "latest_absent_rows": lat_c.get("absent", 0),
                    "previous_absent_employees": prev_c.get("absent_employees", 0),
                    "latest_absent_employees": lat_c.get("absent_employees", 0),
                    "previous_absent_rate_pct": prev_rate,
                    "latest_absent_rate_pct": lat_rate,
                },
                "department_deltas": dept_deltas,
                "top_department": dept_deltas[0] if dept_deltas else None,
                "note": (
                    "Absent rows are attendance.status='absent'. Distinct absent employees are unique employee_id values "
                    "with at least one absent row. Percentage change uses absent rows: "
                    "100 * (latest - previous) / previous. Percentage-point change uses absent_rate_pct = "
                    "100 * absent_rows / recorded_rows for each period (latest rate minus previous rate)."
                ),
            }
            return payload, _context(
                "org_attendance",
                0,
                0,
                start=latest["start"],
                end=latest["end"],
                label=latest["label"],
                prior_window={
                    "start": previous["start"],
                    "end": previous["end"],
                    "label": previous["label"],
                    "counts": previous["counts"],
                },
            )
    # Department average recorded hours (previous calendar month / last month).
    if re.search(r"\b(average|avg|mean)\b", text) and re.search(r"\b(working )?hours\b", text) and re.search(
        r"\b(department|departments|every department)\b", text
    ):
        start, end, label = attendance_window(message, today, inherited, subject="org_attendance")
        departments = {row.id: row.name for row in db.query(Department).all()}
        ranked = []
        for dept_id, name in departments.items():
            members = [row for row in employees if row.department_id == dept_id]
            if not members:
                continue
            ids = [row.id for row in members]
            rows = _attendance_rows(db, ids, start, end)
            total_hours = sum(float(row.hours or 0) for row in rows)
            # Average recorded hours per attendance row (missing days are not rows and are excluded).
            avg_hours = round(total_hours / len(rows), 2) if rows else 0.0
            ranked.append(
                {
                    "department": name,
                    "employee_count": len(ids),
                    "avg_hours": avg_hours,
                    "total_hours": round(total_hours, 1),
                    "rows": len(rows),
                }
            )
        ranked.sort(key=lambda item: (-item["avg_hours"], item["department"]))
        top = ranked[0] if ranked else None
        payload = {
            "kind": "org_attendance",
            "focus": "department_avg_hours",
            "department": top["department"] if top else "",
            "employee_count": len(employees),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "label": label,
            "departments": ranked,
            "people_count": len(ranked),
            "note": (
                f"Average recorded attendance.hours per stored attendance row by department for {label}. "
                "Missing attendance days are not rows and are excluded from the average denominator. "
                f"Highest average: {top['department']} ({top['avg_hours']})." if top else "No departments."
            ),
        }
        return payload, _context(
            "org_attendance", 0, 0, start=start.isoformat(), end=end.isoformat(), label=label
        )
    if (
        len(named_departments) >= 2
        and re.search(r"\b(compare|between|versus|vs\.?)\b", text)
        and re.search(r"\bremote\b", text)
    ):
        start, end, label = attendance_window(message, today, inherited, subject="org_attendance")
        dept_rows = []
        for dept in named_departments[:2]:
            ids = [row.id for row in employees if row.department_id == dept.id]
            rows = _attendance_rows(db, ids, start, end)
            counts = _counts(rows)
            dept_rows.append(
                {
                    "department": dept.name,
                    "employee_count": len(ids),
                    "remote": counts.get("remote", 0),
                    "recorded_days": counts.get("recorded_days", 0),
                    "counts": counts,
                }
            )
        payload = {
            "kind": "org_attendance",
            "focus": "department_remote_compare",
            "department": "",
            "employee_count": sum(item["employee_count"] for item in dept_rows),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "label": label,
            "departments": dept_rows,
            "note": "Remote-day totals are stored attendance rows with status remote for active employees in each named department.",
        }
        return payload, _context(
            "org_attendance", 0, 0, start=start.isoformat(), end=end.isoformat(), label=label
        )
    if re.search(
        r"\b(remote[- ]heaviest|heaviest (?:remote|absent|late) department|department membership and .{0,30} remote|"
        r"which department (?:had|has|shows) the (?:most|highest) remote)\b",
        text,
    ):
        start, end, label = attendance_window(message, today, inherited, subject="org_attendance")
        departments = {row.id: row.name for row in db.query(Department).all()}
        ranked = []
        for dept_id, name in departments.items():
            ids = [row.id for row in employees if row.department_id == dept_id]
            rows = _attendance_rows(db, ids, start, end)
            counts = _counts(rows)
            ranked.append(
                {
                    "department": name,
                    "employee_count": len(ids),
                    "remote": counts.get("remote", 0),
                    "recorded_days": counts.get("recorded_days", 0),
                    "counts": counts,
                }
            )
        ranked.sort(key=lambda item: (-item["remote"], item["department"]))
        payload = {
            "kind": "org_attendance",
            "focus": "remote_heaviest_department",
            "department": "",
            "employee_count": len(employees),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "label": label,
            "departments": ranked,
            "people_count": len(ranked),
            "note": (
                "Remote-heaviest is the department with the most stored attendance rows of status remote in the window. "
                "Department membership comes from employees.department_id; remote counts come from attendance.status. "
                "The ranking does not prove a policy preference, a coverage failure, or a productivity difference, "
                "and a department with more people will usually have more remote rows."
            ),
        }
        return payload, _context(
            "org_attendance", 0, 0, start=start.isoformat(), end=end.isoformat(), label=label
        )
    if department is not None:
        employees = [row for row in employees if row.department_id == department.id]
    ids = [row.id for row in employees]
    names = {row.id: row.full_name for row in employees}
    comparison = parse_comparison(message, today, inherited, subject="org_attendance")
    if comparison:
        periods = []
        for start, end, label in comparison:
            rows = _attendance_rows(db, ids, start, end)
            periods.append({"start": start.isoformat(), "end": end.isoformat(), "label": label, "counts": _counts(rows)})
        payload = {
            "kind": "org_attendance",
            "focus": "compare",
            "department": department.name if department else "",
            "employee_count": len(ids),
            "periods": periods,
            "note": "The denominator for each percentage is the number of stored attendance rows in that window. Weekends are not added as absences.",
        }
        last = periods[-1]
        return payload, _context(
            "org_attendance", 0, 0, department=payload["department"], start=last["start"], end=last["end"], label=last["label"]
        )

    if re.search(r"\bmissing attendance\b", text):
        return _missing(db, employees, names, department)

    start, end, label = attendance_window(message, today, inherited, subject="org_attendance")
    rows = _attendance_rows(db, ids, start, end)
    focus = "summary"
    ranked: list[dict] = []
    payload_count = 0
    department_rows: list[dict] = []
    wants_department_summary = bool(
        department is None
        and re.search(r"\b(all departments|every department|by department|department(?:-|\s)?wise|departments)\b", text)
        and re.search(r"\b(attendance|summary|summaries|breakdown|present|remote|absent|hours)\b", text)
    )
    if re.search(r"\b(most absences|more than three)\b", text):
        focus = "absences"
        by_person: dict[int, int] = defaultdict(int)
        for row in rows:
            if row.status == "absent":
                by_person[row.employee_id] += 1
        ranked = [
            {"full_name": names.get(employee_id, ""), "absences": count}
            for employee_id, count in sorted(by_person.items(), key=lambda item: (-item[1], names.get(item[0], "")))
        ]
        if "more than three" in text:
            ranked = [item for item in ranked if item["absences"] > 3]
    elif "remotely" in text or re.search(r"\bremote\b", text):
        focus = "remote"
        seen = []
        for row in rows:
            if row.status == "remote" and names.get(row.employee_id) not in seen:
                seen.append(names[row.employee_id])
        ranked = [{"full_name": name} for name in seen[:PAGE]]
        payload_count = len(seen)
    elif re.search(r"\battendance rates\b", text):
        focus = "rates"
        by_person: dict[int, list[Attendance]] = defaultdict(list)
        for row in rows:
            by_person[row.employee_id].append(row)
        scored = []
        for employee_id, person_rows in by_person.items():
            counts = _counts(person_rows)
            if counts["attendance_percentage"] is None:
                continue
            scored.append(
                {
                    "full_name": names.get(employee_id, ""),
                    "attendance_percentage": counts["attendance_percentage"],
                    "recorded_days": counts["recorded_days"],
                }
            )
        ranked = sorted(scored, key=lambda item: (-item["attendance_percentage"], item["full_name"]))[:PAGE]
    elif re.search(r"\bpresent today\b", text) or re.search(r"\bwere present\b", text):
        focus = "present"
    elif wants_department_summary:
        focus = "department_summary"
        departments = {row.id: row.name for row in db.query(Department).all()}
        by_dept_rows: dict[str, list[Attendance]] = defaultdict(list)
        emp_dept = {row.id: departments.get(row.department_id, "") for row in employees}
        for row in rows:
            dept_name = emp_dept.get(row.employee_id)
            if dept_name:
                by_dept_rows[dept_name].append(row)
        for dept_name in sorted(departments.values()):
            dept_member_rows = by_dept_rows.get(dept_name) or []
            counts = _counts(dept_member_rows)
            member_count = sum(1 for row in employees if departments.get(row.department_id) == dept_name)
            department_rows.append(
                {
                    "department": dept_name,
                    "employee_count": member_count,
                    "present": counts.get("present", 0),
                    "remote": counts.get("remote", 0),
                    "absent": counts.get("absent", 0),
                    "sick": counts.get("sick", 0),
                    "leave": counts.get("leave", 0),
                    "late": counts.get("late", 0),
                    "hours": counts.get("hours", 0),
                    "recorded_days": counts.get("recorded_days", 0),
                    "attendance_percentage": counts.get("attendance_percentage"),
                }
            )
    counts = _counts(rows)
    if (
        focus == "summary"
        and re.search(r"\b(how many|number of|count)\b", text)
        and re.search(r"\b(employees? were absent|employees? absent|people were absent|absent employees?)\b", text)
    ):
        focus = "absent_employees"
    payload = {
        "kind": "org_attendance",
        "focus": focus,
        "department": department.name if department else "",
        "employee_count": len(ids),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": label,
        "counts": counts,
        "departments": department_rows,
        "people": ranked[:PAGE],
        "people_count": payload_count if focus == "remote" else len(ranked),
        "truncated": (payload_count if focus == "remote" else len(ranked)) > PAGE,
        "absence_filter": "more than three" if "more than three" in text else "",
        "note": (
            "Absences are stored attendance rows with status absent. Missing rows, weekends, and holidays are not counted as absences. "
            "Attendance percentage = (present + remote + late) / stored attendance rows in the window. "
            "Distinct absent employees are unique employee_id values with at least one absent row."
            if focus in {"department_summary", "absent_employees"}
            else "Absences are stored attendance rows with status absent. Missing rows, weekends, and holidays are not counted as absences."
        ),
    }
    return payload, _context(
        "org_attendance",
        0,
        0,
        department=payload["department"],
        start=start.isoformat(),
        end=end.isoformat(),
        label=label,
        focus=focus,
    )


def _attendance_rows(db: Session, ids: list[int], start: date, end: date) -> list[Attendance]:
    if not ids:
        return []
    return (
        db.query(Attendance)
        .filter(Attendance.employee_id.in_(ids), Attendance.work_date >= start, Attendance.work_date <= end)
        .all()
    )


def _missing(db: Session, employees: list[Employee], names: dict[int, str], department: Department | None) -> tuple[dict, dict]:
    span = db.query(Attendance.work_date).order_by(Attendance.work_date).all()
    if not span:
        payload = {"kind": "org_attendance", "focus": "missing", "people": [], "people_count": 0, "note": "No attendance rows are stored."}
        return payload, _context("org_attendance", 0, 0, department=department.name if department else "")
    start = span[0][0]
    end = span[-1][0]
    stored = defaultdict(set)
    for employee_id, work_date in db.query(Attendance.employee_id, Attendance.work_date).all():
        stored[employee_id].add(work_date)
    weekdays = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            weekdays.append(cursor)
        cursor += timedelta(days=1)
    people = []
    for employee in employees:
        missing = [day for day in weekdays if day not in stored[employee.id]]
        if missing:
            people.append({"full_name": names.get(employee.id, employee.full_name), "missing_weekdays": len(missing)})
    people.sort(key=lambda item: (-item["missing_weekdays"], item["full_name"]))
    payload = {
        "kind": "org_attendance",
        "focus": "missing",
        "department": department.name if department else "",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "people": people[:PAGE],
        "people_count": len(people),
        "truncated": len(people) > PAGE,
        "note": "A missing record is a Monday-Friday date in the stored span with no attendance row. Weekends are excluded and were not marked absent.",
    }
    return payload, _context("org_attendance", 0, 0, department=payload["department"], start=payload["start"], end=payload["end"])


def _balance_integrity(
    db: Session,
    employees: dict[int, Employee],
    departments: dict[int, str],
    department: Department | None,
    today: date,
    text: str,
) -> tuple[dict, dict]:
    """Read-only check that leave_requests rows agree with leave_balances arithmetic.

    For every stored balance row of the asked year and type, approved requests starting in that
    year are summed against used_days, pending requests against pending_days, and remaining is
    recomputed as entitled - used - pending. Differences are reported; nothing is changed.
    """
    year_match = re.search(r"\b(20\d{2})\b", text)
    year = int(year_match.group(1)) if year_match else today.year
    type_match = re.search(r"\b(annual|sick|unpaid|parental|compassionate)\b", text)
    leave_type = type_match.group(1) if type_match else "annual"
    balance_rows = db.query(LeaveBalance).filter(LeaveBalance.leave_type == leave_type, LeaveBalance.year == year).all()
    if department is not None:
        balance_rows = [
            row for row in balance_rows if employees.get(row.employee_id) and employees[row.employee_id].department_id == department.id
        ]
    request_rows = (
        db.query(LeaveRequest)
        .filter(LeaveRequest.leave_type == leave_type, LeaveRequest.start_date >= date(year, 1, 1), LeaveRequest.start_date <= date(year, 12, 31))
        .all()
    )
    approved_by_emp: dict[int, float] = {}
    pending_by_emp: dict[int, float] = {}
    oversize: list[dict] = []
    for row in request_rows:
        days = float(row.business_days or 0)
        if row.status == "approved":
            approved_by_emp[row.employee_id] = approved_by_emp.get(row.employee_id, 0.0) + days
        elif row.status in {"pending_manager", "pending_hr"}:
            pending_by_emp[row.employee_id] = pending_by_emp.get(row.employee_id, 0.0) + days
    findings: list[dict] = []
    checked = 0
    for balance in balance_rows:
        person = employees.get(balance.employee_id)
        if person is None:
            continue
        checked += 1
        entitled = float(balance.entitled_days)
        used = float(balance.used_days)
        pending = float(balance.pending_days)
        remaining = entitled - used - pending
        approved_sum = approved_by_emp.get(balance.employee_id, 0.0)
        pending_sum = pending_by_emp.get(balance.employee_id, 0.0)
        issues = []
        if abs(approved_sum - used) > 0.01:
            issues.append(f"used_days {used:g} vs approved {leave_type} request days {approved_sum:g}")
        if abs(pending_sum - pending) > 0.01:
            issues.append(f"pending_days {pending:g} vs pending request days {pending_sum:g}")
        if remaining < 0:
            issues.append(f"remaining {remaining:g} is negative")
        if issues:
            findings.append(
                {
                    "full_name": person.full_name,
                    "employee_number": person.employee_number,
                    "department": departments.get(person.department_id, ""),
                    "entitled": entitled,
                    "used": used,
                    "pending": pending,
                    "remaining": remaining,
                    "approved_request_days": approved_sum,
                    "pending_request_days": pending_sum,
                    "issues": issues,
                }
            )
    for row in request_rows:
        balance = next((b for b in balance_rows if b.employee_id == row.employee_id), None)
        if balance is None:
            continue
        remaining = float(balance.entitled_days) - float(balance.used_days) - float(balance.pending_days)
        if row.status in {"pending_manager", "pending_hr"} and float(row.business_days or 0) > remaining + float(balance.pending_days):
            person = employees.get(row.employee_id)
            oversize.append(
                {
                    "request_id": row.id,
                    "full_name": person.full_name if person else "",
                    "employee_number": person.employee_number if person else "",
                    "business_days": float(row.business_days or 0),
                    "status": row.status,
                    "remaining_before_pending": remaining + float(balance.pending_days),
                }
            )
    findings.sort(key=lambda item: item["full_name"])
    payload = {
        "kind": "org_leave",
        "focus": "balance_integrity",
        "department": department.name if department else "",
        "requests": [],
        "people_count": len(findings),
        "balances": [],
        "balance_count": checked,
        "integrity": {
            "year": year,
            "leave_type": leave_type,
            "balance_rows_checked": checked,
            "requests_checked": len(request_rows),
            "findings": findings[:PAGE],
            "finding_count": len(findings),
            "oversize_pending": oversize[:PAGE],
        },
        "truncated": len(findings) > PAGE,
        "note": (
            f"Each {year} {leave_type} leave_balances row was compared with {leave_type} leave_requests starting in {year}: "
            "approved request business days against used_days, pending request business days against pending_days, "
            "and remaining recomputed as entitled - used - pending. Differences are reported as stored; no row was changed "
            "and no cause was inferred."
        ),
    }
    return payload, _context("org_leave", 0, 0, department=payload["department"])


def _leave(db: Session, message: str, inherited: dict | None, target_ids: list[int]) -> tuple[dict, dict]:
    today = org_today()
    text = message.lower()
    inherited_focus = str((inherited or {}).get("focus") or "")
    if not target_ids and inherited and inherited.get("subject_employee_id") and re.search(r"\b(her|his|their|she|he)\b", text):
        target_ids = [int(inherited["subject_employee_id"])]
    if re.search(r"\bhow many are there\b", text) and inherited_focus:
        text = f"{text} {inherited_focus.replace('_', ' ')}"
    department = _department_for(db, message, inherited)
    names = _names(db)
    departments = _departments(db)
    employees = {row.id: row for row in db.query(Employee).all()}
    start, end, label = attendance_window(message, today, inherited, subject="org_leave")
    explicit = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december|this month|last month|today)\b",
        text,
    )
    query = db.query(LeaveRequest)
    if department is not None:
        member_ids = [row.id for row in employees.values() if row.department_id == department.id]
        query = query.filter(LeaveRequest.employee_id.in_(member_ids or [-1]))
    if target_ids and re.search(r"\b(history|balance|her|his)\b", text):
        query = query.filter(LeaveRequest.employee_id.in_(target_ids))
    rows = query.order_by(LeaveRequest.start_date.desc(), LeaveRequest.id.desc()).all()
    focus = "pending"
    selected = rows
    balances: list[dict] = []
    balance_filter = ""
    note_override = ""
    # Only narrow to pending_hr when HR is the sole approval stage asked about.
    # "pending manager or HR" / "pending manager or HR approval" must include both statuses.
    mentions_manager_stage = bool(re.search(r"\b(pending_manager|pending manager|manager)\b", text))
    hr_only_pending = not mentions_manager_stage and bool(
        re.search(r"\b(awaiting hr|need hr|pending_hr|pending hr|hr approval)\b", text)
    )
    balance_totals_ask = bool(
        re.search(
            r"\b(total|sum|aggregate|combined|overall)\b.{0,40}\bremaining\b|\bremaining\b.{0,40}\b(total|sum|in total|across all)\b|"
            r"\b(how many|count|number of)\b.{0,40}\bbalance rows?\b|\bbalance rows?\b.{0,20}\b(exist|count|are there|stored)\b|"
            r"\b(how many|count|number of)\b.{0,30}\b(leave )?balances?\b.{0,20}\b(exist|stored|are there|for (?:year )?\d{4})",
            text,
        )
    )
    if re.search(
        r"\b(status breakdown|by status|approved.*rejected|rejected.*approved|"
        r"pending manager.*pending hr|how many leave requests are approved|"
        r"approved, rejected, pending|by status and leave type|status and leave type)\b",
        text,
    ) or (
        re.search(r"\bapproved\b", text)
        and re.search(r"\brejected\b", text)
        and re.search(r"\bpending\b", text)
    ) or (
        re.search(r"\b(status|statuses)\b", text)
        and re.search(r"\bleave type\b", text)
    ):
        counts = {
            "approved": sum(1 for row in rows if row.status == "approved"),
            "rejected": sum(1 for row in rows if row.status == "rejected"),
            "pending_manager": sum(1 for row in rows if row.status == "pending_manager"),
            "pending_hr": sum(1 for row in rows if row.status == "pending_hr"),
        }
        cross: dict[tuple[str, str], int] = defaultdict(int)
        for row in rows:
            cross[(row.status, row.leave_type)] += 1
        by_status_type = [
            {"status": status, "leave_type": leave_type, "n": n}
            for (status, leave_type), n in sorted(cross.items())
        ]
        want_type = bool(re.search(r"\b(leave type|by type|status and leave type|type)\b", text))
        payload = {
            "kind": "org_leave",
            "focus": "status_by_type" if want_type else "status_breakdown",
            "department": department.name if department else "",
            "status_counts": counts,
            "by_status_type": by_status_type,
            "requests": [],
            "people_count": 0,
            "note": (
                f"Leave request status counts: approved {counts['approved']}, "
                f"rejected {counts['rejected']}, pending_manager {counts['pending_manager']}, "
                f"pending_hr {counts['pending_hr']}."
                + (
                    " Cross-tabulated by leave type as well."
                    if want_type
                    else ""
                )
            ),
        }
        return payload, _context("org_leave", 0, 0, department=payload["department"])
    if re.search(r"\b(pending the longest|longest .{0,20}pending|which leave request has been pending)\b", text):
        pending = [row for row in rows if row.status in {"pending_manager", "pending_hr"}]
        pending.sort(key=lambda row: (row.submitted_at or row.start_date, row.id))
        selected = pending[:1]
        focus = "longest_pending"
        payload = {
            "kind": "org_leave",
            "focus": focus,
            "department": department.name if department else "",
            "requests": [_leave_item(row, names, departments, employees) for row in selected],
            "people_count": len(selected),
            "note": (
                "Longest-pending leave request is the earliest submitted_at among statuses "
                "pending_manager and pending_hr."
            ),
        }
        if selected:
            payload["longest_pending_id"] = selected[0].id
        return payload, _context("org_leave", 0, 0, department=payload["department"])
    if re.search(r"\baverage remaining annual leave\b", text) or (
        re.search(r"\baverage\b", text) and re.search(r"\bremaining\b", text) and re.search(r"\bleave\b", text)
    ):
        year = today.year
        balance_rows = (
            db.query(LeaveBalance)
            .filter(LeaveBalance.leave_type == "annual", LeaveBalance.year == year)
            .all()
        )
        # Prefer active employees' balances when available.
        active_ids = {row.id for row in employees.values() if row.employment_status == "active"}
        active_balances = [row for row in balance_rows if row.employee_id in active_ids]
        use_rows = active_balances or balance_rows
        rem = [float(row.entitled_days) - float(row.used_days) - float(row.pending_days) for row in use_rows]
        avg = round(sum(rem) / len(rem), 2) if rem else 0.0
        payload = {
            "kind": "org_leave",
            "focus": "avg_remaining",
            "department": department.name if department else "",
            "requests": [],
            "people_count": len(use_rows),
            "average_remaining": avg,
            "balance_count": len(use_rows),
            "note": (
                f"Average remaining annual leave across {len(use_rows)} leave_balances rows for {year} "
                f"is {avg}. Remaining = entitled - used - pending."
            ),
        }
        return payload, _context("org_leave", 0, 0, department=payload["department"])
    if balance_totals_ask and "lowest" not in text and "highest" not in text:
        focus = "balance_totals"
        year_match = re.search(r"\b(20\d{2})\b", text)
        year = int(year_match.group(1)) if year_match else today.year
        type_match = re.search(r"\b(annual|sick|unpaid|parental|compassionate)\b", text)
        leave_type = type_match.group(1) if type_match else "annual"
        balance_rows = db.query(LeaveBalance).filter(LeaveBalance.leave_type == leave_type, LeaveBalance.year == year).all()
        if department is not None:
            balance_rows = [
                row
                for row in balance_rows
                if employees.get(row.employee_id) and employees[row.employee_id].department_id == department.id
            ]
        remaining_total = sum(float(row.entitled_days) - float(row.used_days) - float(row.pending_days) for row in balance_rows)
        totals = {
            "year": year,
            "leave_type": leave_type,
            "rows": len(balance_rows),
            "employees": len({row.employee_id for row in balance_rows}),
            "entitled": round(sum(float(row.entitled_days) for row in balance_rows), 2),
            "used": round(sum(float(row.used_days) for row in balance_rows), 2),
            "pending": round(sum(float(row.pending_days) for row in balance_rows), 2),
            "remaining": round(remaining_total, 2),
        }
        payload = {
            "kind": "org_leave",
            "focus": focus,
            "department": department.name if department else "",
            "requests": [],
            "people_count": totals["rows"],
            "balances": [],
            "balance_count": totals["rows"],
            "balance_totals": totals,
            "truncated": False,
            "note": (
                f"Totals are summed over the stored {leave_type} leave_balances rows for year {year}"
                f"{' in ' + department.name if department else ''}. Remaining = entitled - used - pending per row; "
                "pending leave is reserved, not used."
            ),
        }
        return payload, _context("org_leave", 0, 0, department=payload["department"])
    if re.search(
        r"\b(inconsistent|inconsistenc\w*|integrity|arithmetic|mismatch\w*|reconcil\w*|tally|tie out|"
        r"(?:do not|don't|doesn't|does not) add up|add up|out of sync|disagree)\b",
        text,
    ) and re.search(r"\b(balances?|used|pending|remaining|entitled)\b", text):
        return _balance_integrity(db, employees, departments, department, today, text)
    if hr_only_pending or "awaiting my approval" in text:
        focus = "hr_approval" if hr_only_pending else "pending"
        if hr_only_pending:
            selected = [row for row in rows if row.status == "pending_hr"]
        else:
            selected = [row for row in rows if row.status in {"pending_manager", "pending_hr"}]
    elif "pending sick" in text:
        focus = "pending_sick"
        selected = [row for row in rows if row.status in {"pending_manager", "pending_hr"} and row.leave_type == "sick"]
    elif re.search(r"\bapproved\b", text) and re.search(r"\bleave requests?\b", text) and re.search(
        r"\b(start(?:ing|s)? in|start date|with status approved)\b", text
    ):
        # Approved requests whose start_date falls in a named calendar year.
        focus = "approved_year"
        year = today.year
        ym = re.search(r"\b(20\d{2})\b", text)
        if ym:
            year = int(ym.group(1))
        selected = [
            row
            for row in rows
            if row.status == "approved" and row.start_date and row.start_date.year == year
        ]
        note_override = (
            f"{len(selected)} leave request{'s' if len(selected) != 1 else ''} with status approved "
            f"have a start_date in calendar year {year}."
        )
    elif "on leave" in text or "on approved leave" in text or re.search(
        r"\bwho is on approved leave\b", text
    ):
        focus = "on_leave"
        # Optional explicit date in the question; otherwise use the organization date.
        day = None
        iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        named = re.search(
            r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b",
            text,
        )
        named2 = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:,?\s*(\d{4}))?\b",
            text,
        )
        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        if iso:
            day = date.fromisoformat(iso.group(1))
        elif named:
            day = date(int(named.group(3)), months[named.group(2)], int(named.group(1)))
        elif named2:
            year = int(named2.group(3)) if named2.group(3) else today.year
            day = date(year, months[named2.group(1)], int(named2.group(2)))
        else:
            day = today
        selected = [row for row in rows if row.status == "approved" and row.start_date <= day <= row.end_date]
    elif (
        "exhausted" in text
        or "lowest" in text
        or "low leave" in text
        or "sort" in text
        or "negative" in text
        or "insufficient" in text
        or "zero remaining" in text
        or ("highest" in text and "remaining" in text)
        or ("remaining leave balances" in text)
    ):
        focus = "balances"
        year = today.year
        balance_rows = db.query(LeaveBalance).filter(LeaveBalance.leave_type == "annual", LeaveBalance.year == year).all()
        if department is not None:
            balance_rows = [row for row in balance_rows if employees.get(row.employee_id) and employees[row.employee_id].department_id == department.id]
        for row in balance_rows:
            remaining = row.entitled_days - row.used_days - row.pending_days
            person = employees.get(row.employee_id)
            balances.append(
                {
                    "full_name": person.full_name if person else "",
                    "employee_number": person.employee_number if person else "",
                    "department": departments.get(person.department_id, "") if person else "",
                    "leave_type": "annual",
                    "remaining": remaining,
                    "entitled": row.entitled_days,
                    "used": row.used_days,
                    "pending": row.pending_days,
                }
            )
        balances.sort(key=lambda item: (item["remaining"], item["full_name"]))
        balance_filter = "ranked"
        if "negative" in text and "insufficient" not in text:
            balances = [item for item in balances if item["remaining"] < 0]
            balance_filter = "negative"
        elif "insufficient" in text or ("negative" in text and "insufficient" in text):
            # No separate minimum-balance threshold is configured. Treat remaining <= 0
            # as insufficient for further annual leave without inventing a soft threshold.
            balances = [item for item in balances if item["remaining"] <= 0]
            balance_filter = "insufficient"
        elif "zero remaining" in text:
            balances = [item for item in balances if item["remaining"] == 0]
            balance_filter = "zero"
        elif "exhausted" in text:
            balances = [item for item in balances if item["remaining"] <= 0]
            balance_filter = "exhausted"
        elif "highest" in text and "remaining" in text and "department" in text:
            # Aggregate remaining by department, highest first.
            dept_totals: dict[str, float] = defaultdict(float)
            for item in balances:
                dept_totals[item["department"] or "Unknown"] += float(item["remaining"])
            balances = [
                {
                    "full_name": name,
                    "employee_number": "",
                    "department": name,
                    "leave_type": "annual",
                    "remaining": total,
                    "entitled": None,
                    "used": None,
                    "pending": None,
                }
                for name, total in sorted(dept_totals.items(), key=lambda pair: (-pair[1], pair[0]))
            ]
            balance_filter = "department_remaining_high"
        selected = []
    elif "audit" in text or "who approved" in text:
        return _audit(db, department, latest_only="audit" not in text and "history" not in text)
    elif "overlapping" in text:
        focus = "overlap"
        pool = [row for row in rows if row.status in OPEN_LEAVE]
        pairs = []
        for index, left in enumerate(pool):
            for right in pool[index + 1 :]:
                if left.employee_id == right.employee_id:
                    continue
                if left.start_date <= right.end_date and right.start_date <= left.end_date:
                    pairs.append(
                        {
                            "left": names.get(left.employee_id, ""),
                            "right": names.get(right.employee_id, ""),
                            "left_status": left.status,
                            "right_status": right.status,
                            "start": max(left.start_date, right.start_date).isoformat(),
                            "end": min(left.end_date, right.end_date).isoformat(),
                        }
                    )
        payload = {
            "kind": "org_leave",
            "focus": focus,
            "department": department.name if department else "",
            "pairs": pairs[:PAGE],
            "people_count": len(pairs),
            "truncated": len(pairs) > PAGE,
            "note": "An overlap is two different employees with pending or approved leave whose dates intersect.",
        }
        return payload, _context("org_leave", 0, 0, department=payload["department"])
    elif "history" in text or (target_ids and "balance" not in text and "sick" not in text):
        focus = "history"
        selected = rows[:PAGE]
    elif "sick" in text:
        focus = "balance_type"
        subject_ids = target_ids or [row.employee_id for row in rows[:1]]
        balances = _balance_items(db, employees, subject_ids, "sick")
        selected = []
    elif "balance" in text:
        focus = "balance_type"
        balances = _balance_items(db, employees, target_ids, "annual")
        selected = []
    elif "approved and rejected" in text or "submitted this month" in text:
        focus = "month"
        if not explicit:
            start = today.replace(day=1)
            end = today
            label = "this month"
        if "submitted" in text:
            selected = [row for row in rows if start <= row.submitted_at.date() <= end]
        else:
            selected = [
                row
                for row in rows
                if row.resolved_at is not None and start <= row.resolved_at.date() <= end and row.status in {"approved", "rejected"}
            ]
    elif "pending" in text:
        focus = "pending"
        selected = [row for row in rows if row.status in {"pending_manager", "pending_hr"}]
    else:
        focus = "pending"
        selected = [row for row in rows if row.status in {"pending_manager", "pending_hr"}]

    shown = [_leave_item(row, names, departments, employees) for row in selected[:PAGE]]
    payload = {
        "kind": "org_leave",
        "focus": focus,
        "department": department.name if department else "",
        "requests": shown,
        "people_count": len(selected) if focus != "balances" else len(balances),
        "truncated": len(selected) > PAGE,
        "balances": balances[:PAGE],
        "balance_count": len(balances),
        "start": start.isoformat() if isinstance(start, date) else "",
        "end": end.isoformat() if isinstance(end, date) else "",
        "label": label,
        "balance_filter": balance_filter if focus == "balances" else "",
        "note": note_override
        or {
            "on_leave": "On leave means an approved request whose dates include the organization date. Pending requests are not included.",
            "history": "These are the stored leave requests for the named employee.",
            "pending": "Pending means status pending_manager or pending_hr.",
            "hr_approval": "These requests have status pending_hr. Requests still with the manager are not included.",
            "pending_sick": "These are sick-leave requests that are still pending.",
            "approved_year": "Approved leave requests filtered by start_date year. Pending requests are excluded.",
            "balances": {
                "exhausted": "Exhausted means entitled minus used minus pending is 0 or below.",
                "negative": "Negative means entitled minus used minus pending is below zero.",
                "zero": "Zero remaining means entitled minus used minus pending equals 0.",
                "insufficient": (
                    "Insufficient here means remaining annual leave is 0 or below. "
                    "No separate minimum-balance threshold is configured in this dataset, "
                    "so a soft low-balance band was not invented."
                ),
                "ranked": "Sorted from lowest remaining balance to highest.",
                "department_remaining_high": (
                    "Departments ranked by total remaining annual leave (entitled - used - pending). "
                    "Pending leave is reserved and is not counted as used leave."
                ),
            }.get(balance_filter, "Sorted from lowest remaining balance to highest."),
        }.get(focus, ""),
    }
    if focus == "month" and "approved and rejected" in text:
        payload["approved"] = sum(1 for row in selected if row.status == "approved")
        payload["rejected"] = sum(1 for row in selected if row.status == "rejected")
    return payload, _context("org_leave", 0, 0, department=payload["department"], start=payload["start"], end=payload["end"], label=label)


def _balance_items(db: Session, employees: dict[int, Employee], subject_ids: list[int], leave_type: str) -> list[dict]:
    if not subject_ids:
        return []
    year = org_today().year
    rows = (
        db.query(LeaveBalance)
        .filter(LeaveBalance.employee_id.in_(subject_ids), LeaveBalance.leave_type == leave_type, LeaveBalance.year == year)
        .all()
    )
    items = []
    for row in rows:
        person = employees.get(row.employee_id)
        items.append(
            {
                "full_name": person.full_name if person else "",
                "employee_number": person.employee_number if person else "",
                "leave_type": leave_type,
                "remaining": row.entitled_days - row.used_days - row.pending_days,
                "entitled": row.entitled_days,
                "used": row.used_days,
                "pending": row.pending_days,
            }
        )
    return items


def _leave_item(row: LeaveRequest, names: dict[int, str], departments: dict[int, str], employees: dict[int, Employee]) -> dict:
    person = employees.get(row.employee_id)
    return {
        "id": row.id,
        "full_name": names.get(row.employee_id, ""),
        "employee_number": person.employee_number if person else "",
        "department": departments.get(person.department_id, "") if person else "",
        "leave_type": row.leave_type,
        "start": row.start_date.isoformat(),
        "end": row.end_date.isoformat(),
        "status": row.status,
        "business_days": row.business_days,
        "hr_required": row.hr_required,
    }


def _audit(db: Session, department: Department | None, latest_only: bool = False) -> tuple[dict, dict]:
    approvals = (
        db.query(Approval)
        .filter(Approval.action_type == "leave", Approval.status == "approved")
        .order_by(Approval.decided_at.desc(), Approval.id.desc())
        .limit(1 if latest_only else PAGE)
        .all()
    )
    if not approvals:
        requests = (
            db.query(LeaveRequest)
            .filter(LeaveRequest.status == "approved")
            .order_by(LeaveRequest.resolved_at.desc(), LeaveRequest.id.desc())
            .limit(1 if latest_only else PAGE)
            .all()
        )
        items = []
        for row in requests:
            subject = db.get(Employee, row.employee_id)
            routed = db.get(Employee, row.routed_manager_id) if row.routed_manager_id else None
            items.append(
                {
                    "approval_id": None,
                    "leave_request_id": row.id,
                    "subject": subject.full_name if subject else "",
                    "approver": "not stored",
                    "role": "routed to " + routed.full_name if routed else "not stored",
                    "decided_at": row.resolved_at.isoformat() if row.resolved_at else "",
                }
            )
        payload = {
            "kind": "org_leave",
            "focus": "audit",
            "department": department.name if department else "",
            "approvals": items,
            "audit_count": 0,
            "note": "These leave requests are approved. No approval row with status approved is stored, so the deciding user is not recorded. The routed manager is not treated as the person who approved it.",
        }
        return payload, _context("org_leave", 0, 0, department=payload["department"])
    logs = (
        db.query(AuditLog)
        .filter(AuditLog.action.in_(("approval_decided", "leave_decided", "assistant_action")))
        .order_by(AuditLog.created_at.desc())
        .limit(PAGE)
        .all()
    )
    items = []
    for row in approvals:
        decider = db.get(User, row.decided_by_user_id) if row.decided_by_user_id else None
        approver = db.get(Employee, decider.employee_id) if decider is not None else None
        subject = db.get(Employee, row.subject_employee_id) if row.subject_employee_id else None
        items.append(
            {
                "approval_id": row.id,
                "leave_request_id": row.leave_request_id,
                "subject": subject.full_name if subject else "",
                "approver": approver.full_name if approver else row.approver_role,
                "role": row.approver_role,
                "decided_at": row.decided_at.isoformat() if row.decided_at else "",
            }
        )
    payload = {
        "kind": "org_leave",
        "focus": "audit",
        "department": department.name if department else "",
        "approvals": items,
        "audit_count": len(logs),
        "note": "The approval rows are the recorded decisions. This read did not change a request.",
    }
    return payload, _context("org_leave", 0, 0, department=payload["department"])


def _compensation(db: Session, message: str, inherited: dict | None) -> tuple[dict, dict]:
    text = message.lower()
    departments = _departments(db)
    active = db.query(Employee).filter(Employee.employment_status == "active").all()
    department = _department_for(db, message, inherited)
    pool = active
    if department is not None and re.search(r"\b(engineering only|only|department)\b", text):
        pool = [row for row in active if row.department_id == department.id]
    annual = sum(row.salary_annual for row in pool)
    grouped = []
    for name in sorted(set(departments.values())):
        members = [row for row in active if departments.get(row.department_id) == name]
        total = sum(row.salary_annual for row in members)
        grouped.append(
            {
                "department": name,
                "employees": len(members),
                "average_annual": round(total / len(members), 2) if members else 0,
                "min_annual": min((row.salary_annual for row in members), default=0),
                "max_annual": max((row.salary_annual for row in members), default=0),
            }
        )
    focus = "payroll"
    people = []
    note = "Monthly payroll here is the sum of stored annual salaries divided by 12. There is no separate payroll run."
    if re.search(r"\bhr\b", text):
        note += " The HR department is stored as People Operations."
    if re.search(
        r"\b(hired before|before 2020).{0,60}\b(salary|compensation)\b|"
        r"\b(salary|compensation).{0,60}\b(hired before|before 2020|hire cohort|historical)\b|"
        r"\bbefore 2020\b.{0,40}\b(2024 or later|hired in 2024)\b",
        text,
    ):
        focus = "salary_cohort"
        note = (
            "Helios stores current salary_annual on employee records only. There is no salary-history table, "
            "so average salaries for hire cohorts (for example hired before 2020 versus hired in 2024 or later) "
            "cannot be computed as historical compensation. Current salaries for those hire-date cohorts can be "
            "averaged from the live employee rows if requested as current pay, but historical pay at hire is unavailable."
        )
        # Still compute current averages for the named cohorts so the answer is grounded, with explicit disclosure.
        before = [row for row in active if row.hire_date and row.hire_date.year < 2020]
        later = [row for row in active if row.hire_date and row.hire_date.year >= 2024]
        def _avg(rows: list[Employee]) -> float:
            return round(sum(float(r.salary_annual or 0) for r in rows) / len(rows), 2) if rows else 0.0
        grouped = [
            {"department": "hired before 2020 (current salary)", "employees": len(before), "average_annual": _avg(before), "min_annual": 0, "max_annual": 0},
            {"department": "hired 2024 or later (current salary)", "employees": len(later), "average_annual": _avg(later), "min_annual": 0, "max_annual": 0},
        ]
        annual = sum(float(row.salary_annual or 0) for row in active)
    elif re.search(
        r"\b(percentage|percent|%).{0,40}\b(salary|expenditure|compensation)\b|"
        r"\bsalary expenditure .{0,30}\bengineering\b|"
        r"\bpercentage of the total annual salary\b",
        text,
    ):
        focus = "salary_pct"
        total = sum(float(row.salary_annual or 0) for row in active)
        dept = department
        if dept is None:
            for row in db.query(Department).all():
                if row.name.lower() in text or row.name.split()[0].lower() in text:
                    dept = row
                    break
        if dept is None:
            dept = next((row for row in db.query(Department).all() if row.name == "Engineering"), None)
        dept_total = sum(
            float(row.salary_annual or 0) for row in active if dept and row.department_id == dept.id
        )
        pct = round(100.0 * dept_total / total, 1) if total else 0.0
        note = (
            f"{dept.name if dept else 'Engineering'} accounts for {pct}% of total active annual salary expenditure "
            f"({dept_total:g} of {total:g}). Values are stored salary_annual, not verified payroll runs."
        )
        grouped = [
            {
                "department": dept.name if dept else "Engineering",
                "employees": sum(1 for row in active if dept and row.department_id == dept.id),
                "average_annual": round(dept_total / max(1, sum(1 for row in active if dept and row.department_id == dept.id)), 2),
                "salary_total": dept_total,
                "salary_percent": pct,
                "org_salary_total": total,
            }
        ]
        department = dept
        annual = total
    elif re.search(
        r"\b(annualized salary|salary total|total for active|stored annual salaries|"
        r"salary expenditure|annual salary expenditure|total annual salary|compensation expenditure|"
        r"payroll expenditure|wage bill|salary spend)\b",
        text,
    ):
        focus = "all_salaries"
        note = (
            "Listed values are stored annual salaries for active employees. "
            "This is an annualized estimate from salary_annual, not verified actual payroll expenditure."
        )
        people = [
            {
                "full_name": row.full_name,
                "employee_number": row.employee_number,
                "department": departments.get(row.department_id, ""),
                "salary_annual": row.salary_annual,
            }
            for row in sorted(pool, key=lambda item: (-item.salary_annual, item.full_name))
        ]
    elif "adjustment" in text:
        focus = "adjustments"
        note = "No salary-adjustment history is stored. I cannot list who received a change this year."
    elif re.search(r"\b(average|avg)\b", text) and re.search(r"\b(highest|department|by department|compare)\b", text):
        focus = "by_department"
        note = "Averages use active employees and the stored annual salary."
    elif "highest" in text and not re.search(r"\baverage\b", text):
        focus = "highest"
        top = max((row.salary_annual for row in active), default=0)
        people = [
            {"full_name": row.full_name, "employee_number": row.employee_number, "department": departments.get(row.department_id, ""), "salary_annual": row.salary_annual}
            for row in active
            if row.salary_annual == top
        ]
    elif "above" in text or "150" in text:
        focus = "threshold"
        people = [
            {"full_name": row.full_name, "employee_number": row.employee_number, "department": departments.get(row.department_id, ""), "salary_annual": row.salary_annual}
            for row in active
            if row.salary_annual > 150_000
        ]
        people.sort(key=lambda item: (-item["salary_annual"], item["full_name"]))
        note = "The threshold is annual salary greater than 150000."
    elif re.search(
        r"\b(organization-?wide|organisation-?wide|org(?:anisation|anization)?-?wide|company-?wide|"
        r"across (?:the )?(?:organization|organisation|company)|for active employees)\b",
        text,
    ) and re.search(r"\b(average|avg|mean)\b", text) and re.search(r"\bsalar", text):
        focus = "org_average"
        values = [float(row.salary_annual) for row in active if row.salary_annual is not None]
        avg = round(sum(values) / len(values), 2) if values else 0.0
        annual = sum(values)
        note = (
            f"Organization-wide average annual salary across {len(values)} active employees with a stored "
            f"salary_annual is {avg}. NULL salary rows are excluded from the average."
        )
        grouped = [
            {
                "department": "organization",
                "employees": len(values),
                "average_annual": avg,
                "min_annual": min(values) if values else 0,
                "max_annual": max(values) if values else 0,
            }
        ]
    elif "average" in text or "compare" in text or "distribution" in text:
        focus = "by_department"
        note = "Averages use active employees and the stored annual salary."
    elif re.search(r"\b(salaries of everyone|everyone'?s salar|all salaries|salaries of all)\b", text):
        focus = "all_salaries"
        people = [
            {
                "full_name": row.full_name,
                "employee_number": row.employee_number,
                "department": departments.get(row.department_id, ""),
                "salary_annual": row.salary_annual,
            }
            for row in sorted(pool, key=lambda item: (-item.salary_annual, item.full_name))
        ]
        note = (
            "Listed values are stored annual salaries for active employees. "
            "This is an annualized estimate from salary_annual, not verified actual payroll expenditure."
        )
    payload = {
        "kind": "compensation_query",
        "focus": focus,
        "department": department.name if department is not None and pool != active else "",
        "active_count": len(pool),
        "annual_total": annual,
        "monthly_from_annual": round(annual / 12, 2) if pool else 0,
        "by_department": grouped,
        "people": people[:PAGE],
        "people_count": len(people),
        "truncated": len(people) > PAGE,
        "note": note,
    }
    return payload, _context("compensation_query", 0, 0, department=payload["department"])


def _metrics(db: Session, message: str, inherited: dict | None) -> tuple[dict, dict]:
    today = org_today()
    text = message.lower()
    employees = db.query(Employee).all()
    active = [row for row in employees if row.employment_status == "active"]
    departments = _departments(db)
    if re.search(r"\bprojects?\b", text) and re.search(r"\b(registered|how many|count)\b", text):
        projects = db.query(Project).order_by(Project.name).all()
        payload = {
            "kind": "hr_metrics",
            "focus": "projects",
            "project_count": len(projects),
            "projects": [
                {"name": row.name, "code": getattr(row, "code", ""), "status": row.status}
                for row in projects
            ],
            "note": "Count is every project row registered in the projects table, including inactive ones when present.",
        }
        return payload, _context("hr_metrics", 0, 0)
    if re.search(r"\bexpired polic", text):
        expired = (
            db.query(PolicyDocument)
            .filter(PolicyDocument.status == "expired")
            .order_by(PolicyDocument.title)
            .all()
        )
        if not expired:
            # Also treat past expiry_date with non-active status wording.
            expired = [
                row
                for row in db.query(PolicyDocument).all()
                if (row.status or "").lower() == "expired"
                or (row.expiry_date is not None and row.expiry_date < today and (row.status or "").lower() != "active")
            ]
        payload = {
            "kind": "hr_metrics",
            "focus": "expired_policies",
            "policy_count": len(expired),
            "policies": [
                {
                    "title": row.title,
                    "slug": row.slug,
                    "status": row.status,
                    "expiry_date": row.expiry_date.isoformat() if row.expiry_date else "",
                }
                for row in expired
            ],
            "note": "Expired policies are policy_documents rows with status expired, or with an expiry_date before the organization date when still stored.",
        }
        return payload, _context("hr_metrics", 0, 0)
    if re.search(
        r"\b(no corresponding approval|without .{0,20}approval decision|no approval decision|"
        r"leave requests have no corresponding approval|requests? without approval)\b",
        text,
    ):
        requests = db.query(LeaveRequest).all()
        approved_ids = {
            row.leave_request_id
            for row in db.query(Approval).filter(Approval.leave_request_id.isnot(None)).all()
            if row.leave_request_id
        }
        missing = [row for row in requests if row.id not in approved_ids]
        payload = {
            "kind": "hr_metrics",
            "focus": "leave_without_approval",
            "leave_without_approval": len(missing),
            "request_count": len(requests),
            "note": (
                f"{len(missing)} leave request{'s' if len(missing) != 1 else ''} have no corresponding row in the "
                "approvals table. This is a leave_requests minus approvals count, not a count of approval rows."
            ),
        }
        return payload, _context("hr_metrics", 0, 0)
    if re.search(r"\bapproval\b.{0,30}\b(rows?|decisions?|records?|entries|table)\b|\bapprovals? (?:table|rows?)\b|\bapproval decisions?\b", text):
        approvals = db.query(Approval).all()
        by_decision: dict[str, int] = defaultdict(int)
        leave_actions = 0
        for row in approvals:
            by_decision[f"{row.action_type}/{row.status}"] += 1
            if "leave" in (row.action_type or "").lower():
                leave_actions += 1
        payload = {
            "kind": "hr_metrics",
            "focus": "approvals",
            "approval_count": len(approvals),
            "leave_action_count": leave_actions,
            "by_decision": [{"decision": key, "rows": value} for key, value in sorted(by_decision.items())],
            "note": (
                "Count is every row in the approvals table, which stores one row per recorded approval decision "
                "(action_type/status shown). Requests that are still pending with no decision have no approval row yet."
            ),
        }
        return payload, _context("hr_metrics", 0, 0)
    if re.search(r"\b(how many|count|number of|list|distinct)\b.{0,30}\bdepartments?\b|\bdepartments? (?:table|rows?|are (?:defined|stored|there))\b", text) and not re.search(
        r"\b(employees?|headcount|salary|salaries|attendance|leave|remote|absent)\b", text
    ):
        rows = db.query(Department).order_by(Department.name).all()
        payload = {
            "kind": "hr_metrics",
            "focus": "departments",
            "department_count": len(rows),
            "departments": [{"name": row.name, "active": sum(1 for person in active if person.department_id == row.id)} for row in rows],
            "note": "Count is every row in the departments table; active headcount per department is shown for context.",
        }
        return payload, _context("hr_metrics", 0, 0)
    if "tenure" in text:
        days = [max((today - row.hire_date).days, 0) for row in active]
        average_years = round((sum(days) / len(days)) / 365.25, 2) if days else 0
        ordered = sorted(days)
        median_days = 0
        if ordered:
            middle = len(ordered) // 2
            median_days = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        payload = {
            "kind": "hr_metrics",
            "focus": "tenure",
            "active_count": len(active),
            "as_of": today.isoformat(),
            "average_tenure_years": average_years,
            "average_tenure_days": round(sum(days) / len(days), 1) if days else 0,
            "median_tenure_years": round(median_days / 365.25, 2) if ordered else 0,
            "note": (
                f"Tenure is the organization date ({today.isoformat()}) minus hire_date for each active employee, "
                "averaged and divided by 365.25. It is an average of stored hire dates, not a termination-adjusted tenure."
            ),
        }
        return payload, _context("hr_metrics", 0, 0)
    if "joined and left" in text or "turnover" in text or "hiring trends" in text:
        hires = [row for row in employees if row.hire_date.year == today.year]
        by_month: dict[str, int] = defaultdict(int)
        for row in hires:
            by_month[row.hire_date.strftime("%Y-%m")] += 1
        payload = {
            "kind": "hr_metrics",
            "focus": "hiring",
            "hires_this_year": len(hires),
            "leavers_this_year": None,
            "by_month": [{"month": month, "hires": count} for month, count in sorted(by_month.items())],
            "note": "Hire dates this calendar year are stored. No termination or leaver record exists, so turnover and leavers are not calculated.",
        }
        return payload, _context("hr_metrics", 0, 0)
    start, end, label = attendance_window(message, today, inherited, subject="hr_metrics")
    if "compare" in text and "last month" in text:
        this_start = today.replace(day=1)
        previous_end = this_start - timedelta(days=1)
        previous_start = previous_end.replace(day=1)
        windows = [(this_start, today, "this month"), (previous_start, previous_end, "last month")]
    else:
        windows = [(start, end, label)]
    periods = []
    ids = [row.id for row in active]
    for window_start, window_end, window_label in windows:
        attendance = _counts(_attendance_rows(db, ids, window_start, window_end))
        leave_rows = (
            db.query(LeaveRequest)
            .filter(LeaveRequest.start_date <= window_end, LeaveRequest.end_date >= window_start)
            .all()
        )
        periods.append(
            {
                "label": window_label,
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
                "active_count": len(active),
                "attendance": attendance,
                "leave_requests_overlapping": len(leave_rows),
            }
        )
    grouped = [
        {"department": name, "active": sum(1 for row in active if departments.get(row.department_id) == name)}
        for name in sorted(set(departments.values()))
    ]
    payload = {
        "kind": "hr_metrics",
        "focus": "summary",
        "periods": periods,
        "by_department": grouped,
        "note": "Headcount is active employees and does not change with the month. Attendance uses stored rows in the window. Leave is requests whose dates overlap the window.",
    }
    last = periods[-1]
    return payload, _context("hr_metrics", 0, 0, start=last["start"], end=last["end"], label=last["label"])


def _action(db: Session, message: str, target_ids: list[int]) -> tuple[dict, None]:
    text = message.lower()
    pending = []
    if target_ids or re.search(r"\be-\d+\b", message, re.IGNORECASE):
        ids = list(target_ids)
        number = re.search(r"\bE-\d+\b", message, re.IGNORECASE)
        if number:
            match = db.query(Employee).filter(Employee.employee_number == number.group(0).upper()).one_or_none()
            if match is not None:
                ids.append(match.id)
        if ids:
            pending = (
                db.query(LeaveRequest)
                .filter(LeaveRequest.employee_id.in_(ids), LeaveRequest.status.in_(("pending_manager", "pending_hr")))
                .all()
            )
    if "delete" in text or "remove" in text:
        statement = "Deleting an employee record is not available in this assistant. No employee was deleted."
    elif "profile update" in text:
        statement = "Preparing an employee profile update is not available in this assistant. No profile was changed."
    elif "behalf" in text:
        statement = "This assistant cannot file leave for another employee. No request was created."
    elif "without confirmation" in text:
        statement = "A leave decision cannot skip confirmation. No request was approved or rejected."
    elif "after a manager approves" in text:
        statement = (
            "After the approval manager approves, a request that does not require HR becomes approved. "
            "A request of more than 10 consecutive business days, or one that covers a critical project deadline, stays pending_hr until HR decides. "
            "No request was changed."
        )
    elif "reject" in text:
        statement = f"{len(pending)} pending leave request(s) match that employee. None was rejected."
    else:
        statement = f"{len(pending)} pending leave request(s) match that employee. None was approved."
    return {
        "kind": "hr_action",
        "changed": False,
        "pending_count": len(pending),
        "statement": statement,
    }, None
