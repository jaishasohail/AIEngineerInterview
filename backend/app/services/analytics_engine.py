"""Workforce analytics computed from attendance rows. The model only explains this payload."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

from sqlalchemy.orm import Session

from app.access import direct_report_ids, employee_for_user
from app.clock import org_today, shift_months
from app.models import Attendance, Department, Employee, LeaveBalance, User
from app.services.records import complete_month_series

UNPLANNED = {"absent", "sick"}


def _scope(db: Session, actor: User) -> tuple[str, list[int] | None, str | None]:
    if actor.role == "hr_admin":
        return "organization", None, None
    employee = employee_for_user(db, actor)
    if actor.role == "manager":
        return "team", direct_report_ids(db, employee.id), f"{employee.full_name}'s direct reports"
    return "self", [employee.id], employee.full_name


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _employees_in_scope(db: Session, actor: User) -> list[Employee]:
    """HR Admin: org. Manager: authorized direct reports only. Employee: self only.

    Attendance rows are filtered to these ids before aggregation — never load org-wide
    rows and filter afterward for managers.
    """
    actor_employee = employee_for_user(db, actor)
    employee_query = db.query(Employee).filter(Employee.employment_status == "active")
    if actor.role == "hr_admin":
        return employee_query.all()
    if actor.role == "manager":
        allowed = list(direct_report_ids(db, actor_employee.id))
        if not allowed:
            return []
        return employee_query.filter(Employee.id.in_(allowed)).all()
    return employee_query.filter(Employee.id == actor_employee.id).all()


def absence_analytics(db: Session, actor: User, months: int = 3) -> dict:
    months = max(1, min(months, 6))
    today = org_today()
    start = shift_months(today, months - 1)
    scope, _department_ids, scope_label = _scope(db, actor)
    employees = _employees_in_scope(db, actor)
    employee_ids = [employee.id for employee in employees]
    departments = {department.id: department.name for department in db.query(Department).all()}
    dept_of = {employee.id: departments[employee.department_id] for employee in employees}
    location_of = {employee.id: employee.location for employee in employees}

    rows = []
    if employee_ids:
        rows = (
            db.query(Attendance)
            .filter(
                Attendance.employee_id.in_(employee_ids),
                Attendance.work_date >= start,
                Attendance.work_date <= today,
            )
            .all()
        )

    dept_totals = defaultdict(
        lambda: {"scheduled_days": 0, "unplanned_absent": 0, "leave_days": 0, "remote_days": 0, "present_days": 0}
    )
    monthly = defaultdict(lambda: {"scheduled_days": 0, "unplanned_absent": 0})
    location_monthly = defaultdict(lambda: {"scheduled_days": 0, "unplanned_absent": 0})
    personal_monthly = defaultdict(lambda: {"scheduled_days": 0, "present": 0, "remote": 0, "leave": 0, "unplanned": 0})
    employee_unplanned: dict[int, int] = defaultdict(int)
    employee_scheduled: dict[int, int] = defaultdict(int)
    name_of = {employee.id: employee.full_name for employee in employees}
    number_of = {employee.id: employee.employee_number for employee in employees}

    for row in rows:
        department = dept_of.get(row.employee_id)
        if department is None:
            continue
        bucket = dept_totals[department]
        bucket["scheduled_days"] += 1
        employee_scheduled[row.employee_id] += 1
        month_key = row.work_date.strftime("%Y-%m")
        month_bucket = monthly[(month_key, department)]
        month_bucket["scheduled_days"] += 1
        location = location_of.get(row.employee_id) or "unknown"
        loc_bucket = location_monthly[(month_key, location)]
        loc_bucket["scheduled_days"] += 1
        if actor.role == "employee":
            personal = personal_monthly[month_key]
            personal["scheduled_days"] += 1
        if row.status in UNPLANNED:
            bucket["unplanned_absent"] += 1
            month_bucket["unplanned_absent"] += 1
            loc_bucket["unplanned_absent"] += 1
            employee_unplanned[row.employee_id] += 1
            if actor.role == "employee":
                personal["unplanned"] += 1
        elif row.status == "leave":
            bucket["leave_days"] += 1
            if actor.role == "employee":
                personal["leave"] += 1
        elif row.status == "remote":
            bucket["remote_days"] += 1
            if actor.role == "employee":
                personal["remote"] += 1
        else:
            bucket["present_days"] += 1
            if actor.role == "employee":
                personal["present"] += 1

    department_rows = []
    for name, bucket in sorted(dept_totals.items()):
        scheduled = bucket["scheduled_days"]
        department_rows.append(
            {
                "department": name,
                **bucket,
                "unplanned_absence_rate": _rate(bucket["unplanned_absent"], scheduled),
                "leave_rate": _rate(bucket["leave_days"], scheduled),
                "remote_rate": _rate(bucket["remote_days"], scheduled),
            }
        )

    monthly_rows = []
    for (month, department), bucket in sorted(monthly.items()):
        scheduled = bucket["scheduled_days"]
        monthly_rows.append(
            {
                "month": month,
                "department": department,
                "scheduled_days": scheduled,
                "unplanned_absent": bucket["unplanned_absent"],
                "unplanned_absence_rate": _rate(bucket["unplanned_absent"], scheduled),
            }
        )

    location_monthly_rows = []
    for (month, location), bucket in sorted(location_monthly.items()):
        scheduled = bucket["scheduled_days"]
        location_monthly_rows.append(
            {
                "month": month,
                "location": location,
                "scheduled_days": scheduled,
                "unplanned_absent": bucket["unplanned_absent"],
                "unplanned_absence_rate": _rate(bucket["unplanned_absent"], scheduled),
            }
        )

    anomalies = _anomalies(monthly_rows)
    utilization = _utilization(db, employee_ids, dept_of)
    for row in department_rows:
        row["leave_utilization"] = utilization.get(row["department"], 0)

    scheduled_sum = sum(item["scheduled_days"] for item in department_rows)
    unplanned_sum = sum(item["unplanned_absent"] for item in department_rows)
    org_rate = _rate(unplanned_sum, scheduled_sum)
    insights = _insights(department_rows, anomalies, org_rate, utilization, scope)

    compare_months = complete_month_series(today, 2)
    month_comparison = _month_comparison(monthly_rows, compare_months)
    office_comparison = _office_comparison(location_monthly_rows, compare_months)
    investigation = _investigation_ranking(department_rows, month_comparison, anomalies)
    recommendations = _recommendations(investigation, month_comparison, office_comparison, anomalies, scope)
    priority_employees = []
    for employee_id, unplanned in sorted(employee_unplanned.items(), key=lambda item: (-item[1], name_of.get(item[0], ""))):
        scheduled = employee_scheduled.get(employee_id, 0)
        if unplanned <= 0:
            continue
        priority_employees.append(
            {
                "full_name": name_of.get(employee_id, ""),
                "employee_number": number_of.get(employee_id, ""),
                "department": dept_of.get(employee_id, ""),
                "unplanned_absent": unplanned,
                "scheduled_days": scheduled,
                "unplanned_absence_rate": _rate(unplanned, scheduled),
            }
        )
        if len(priority_employees) >= 12:
            break

    personal_rows = [{"month": month, **bucket} for month, bucket in sorted(personal_monthly.items())]
    return {
        "period": {"start": start.isoformat(), "end": today.isoformat(), "months": months},
        "scope": scope,
        "scope_label": scope_label,
        # IDs used for the attendance query — never an unfiltered org pull for managers.
        "employee_ids": list(employee_ids),
        "method": (
            "Unplanned absence counts attendance statuses absent and sick. "
            "The denominator is weekday attendance rows in the period. "
            "Leave utilization is used annual-leave days divided by entitled days and is a separate metric. "
            "An anomaly is a department-month whose unplanned rate has a z-score of at least 1.5 against the other department-months in this result. "
            "Month comparisons use the last two complete calendar months only."
        ),
        "org_unplanned_absence_rate": org_rate,
        "departments": department_rows,
        "monthly": monthly_rows,
        "location_monthly": location_monthly_rows,
        "month_comparison": month_comparison,
        "office_comparison": office_comparison,
        "investigation": investigation,
        "priority_employees": priority_employees,
        "recommendations": recommendations,
        "anomalies": anomalies,
        "leave_utilization": [
            {"department": name, "utilization": value} for name, value in sorted(utilization.items())
        ],
        "personal_monthly": personal_rows,
        "insights": insights,
        "simulated_system": "simulated_hris",
        "limitations": [
            "Rates use stored attendance rows only; missing weekdays are excluded from the denominator.",
            "Small scheduled-day counts make percentage changes unstable; those departments are flagged.",
            "Statistical outliers are observations against other department-months, not findings of misconduct or medical conditions.",
            "Office comparisons use employee.location on the attendance row's employee record.",
        ],
    }


def _month_keys(periods: list[tuple[date, date, str]]) -> list[str]:
    return [start.strftime("%Y-%m") for start, _end, _label in periods]


def _month_comparison(monthly_rows: list[dict], periods: list[tuple[date, date, str]]) -> dict:
    if len(periods) < 2:
        return {"previous": None, "latest": None, "departments": [], "note": "Need two complete months."}
    previous_key, latest_key = _month_keys(periods)[-2:]
    previous_label = periods[-2][2]
    latest_label = periods[-1][2]
    by_dept: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in monthly_rows:
        by_dept[row["department"]][row["month"]] = row
    departments = []
    for department, months in sorted(by_dept.items()):
        prev = months.get(previous_key) or {
            "unplanned_absent": 0,
            "scheduled_days": 0,
            "unplanned_absence_rate": 0.0,
        }
        latest = months.get(latest_key) or {
            "unplanned_absent": 0,
            "scheduled_days": 0,
            "unplanned_absence_rate": 0.0,
        }
        prev_rate = float(prev.get("unplanned_absence_rate") or 0)
        latest_rate = float(latest.get("unplanned_absence_rate") or 0)
        change_pp = round((latest_rate - prev_rate) * 100, 2)
        small_sample = min(int(prev.get("scheduled_days") or 0), int(latest.get("scheduled_days") or 0)) < 20
        departments.append(
            {
                "department": department,
                "previous_month": previous_label,
                "latest_month": latest_label,
                "previous_absent": int(prev.get("unplanned_absent") or 0),
                "previous_scheduled": int(prev.get("scheduled_days") or 0),
                "previous_rate": prev_rate,
                "latest_absent": int(latest.get("unplanned_absent") or 0),
                "latest_scheduled": int(latest.get("scheduled_days") or 0),
                "latest_rate": latest_rate,
                "change_pp": change_pp,
                "direction": "increase" if change_pp > 0 else "decrease" if change_pp < 0 else "unchanged",
                "small_sample": small_sample,
            }
        )
    departments.sort(key=lambda item: (-item["change_pp"], item["department"]))
    return {
        "previous": {"key": previous_key, "label": previous_label, "start": periods[-2][0].isoformat(), "end": periods[-2][1].isoformat()},
        "latest": {"key": latest_key, "label": latest_label, "start": periods[-1][0].isoformat(), "end": periods[-1][1].isoformat()},
        "metric": "unplanned_absence_rate",
        "numerator": "attendance statuses absent and sick",
        "denominator": "stored weekday attendance rows in each complete calendar month",
        "departments": departments,
    }


def _office_comparison(location_monthly_rows: list[dict], periods: list[tuple[date, date, str]]) -> dict:
    if len(periods) < 2:
        return {"previous": None, "latest": None, "offices": [], "largest_increase": None}
    previous_key, latest_key = _month_keys(periods)[-2:]
    previous_label = periods[-2][2]
    latest_label = periods[-1][2]
    by_office: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in location_monthly_rows:
        by_office[row["location"]][row["month"]] = row
    offices = []
    for location, months in sorted(by_office.items()):
        prev = months.get(previous_key) or {
            "unplanned_absent": 0,
            "scheduled_days": 0,
            "unplanned_absence_rate": 0.0,
        }
        latest = months.get(latest_key) or {
            "unplanned_absent": 0,
            "scheduled_days": 0,
            "unplanned_absence_rate": 0.0,
        }
        prev_rate = float(prev.get("unplanned_absence_rate") or 0)
        latest_rate = float(latest.get("unplanned_absence_rate") or 0)
        change_pp = round((latest_rate - prev_rate) * 100, 2)
        offices.append(
            {
                "location": location,
                "previous_month": previous_label,
                "latest_month": latest_label,
                "previous_rate": prev_rate,
                "latest_rate": latest_rate,
                "change_pp": change_pp,
                "previous_absent": int(prev.get("unplanned_absent") or 0),
                "previous_scheduled": int(prev.get("scheduled_days") or 0),
                "latest_absent": int(latest.get("unplanned_absent") or 0),
                "latest_scheduled": int(latest.get("scheduled_days") or 0),
                "small_sample": min(int(prev.get("scheduled_days") or 0), int(latest.get("scheduled_days") or 0)) < 20,
                "direction": "increase" if change_pp > 0 else "decrease" if change_pp < 0 else "unchanged",
            }
        )
    offices.sort(key=lambda item: (-item["change_pp"], item["location"]))
    largest = next((item for item in offices if item["change_pp"] > 0), offices[0] if offices else None)
    return {
        "previous": {"key": previous_key, "label": previous_label},
        "latest": {"key": latest_key, "label": latest_label},
        "metric": "unplanned_absence_rate",
        "offices": offices,
        "largest_increase": largest,
    }


def _investigation_ranking(
    departments: list[dict],
    month_comparison: dict,
    anomalies: list[dict],
) -> list[dict]:
    anomaly_by_dept: dict[str, list[dict]] = defaultdict(list)
    for item in anomalies:
        anomaly_by_dept[item["department"]].append(item)
    change_by_dept = {
        item["department"]: item for item in (month_comparison.get("departments") or [])
    }
    ranked = []
    for row in departments:
        name = row["department"]
        change = change_by_dept.get(name) or {}
        anomaly_hits = anomaly_by_dept.get(name) or []
        change_pp = float(change.get("change_pp") or 0)
        rate = float(row.get("unplanned_absence_rate") or 0)
        records = int(row.get("unplanned_absent") or 0)
        scheduled = int(row.get("scheduled_days") or 0)
        small_sample = scheduled < 40 or bool(change.get("small_sample"))
        # Transparent weighted score: rate (0-1), positive month change (pp/100), record volume, anomaly count.
        score = round(
            (rate * 40)
            + (max(change_pp, 0) * 0.8)
            + min(records / 10.0, 20)
            + (5 * len(anomaly_hits)),
            2,
        )
        ranked.append(
            {
                "department": name,
                "score": score,
                "unplanned_absence_rate": rate,
                "unplanned_absent": records,
                "scheduled_days": scheduled,
                "month_change_pp": change_pp,
                "anomaly_count": len(anomaly_hits),
                "anomaly_months": [item["month"] for item in anomaly_hits],
                "small_sample": small_sample,
                "criteria": {
                    "unplanned_absence_rate": rate,
                    "month_to_month_change_pp": change_pp,
                    "affected_records": records,
                    "statistical_anomaly_evidence": len(anomaly_hits),
                },
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["department"]))
    return ranked


def _recommendations(
    investigation: list[dict],
    month_comparison: dict,
    office_comparison: dict,
    anomalies: list[dict],
    scope: str,
) -> dict:
    if scope != "organization":
        return {
            "actions": ["Recommendations for organization-wide attendance patterns require HR Admin organization scope."],
            "grounded_in": [],
        }
    actions = []
    grounded = []
    top = investigation[:3] if investigation else []
    for item in top:
        if item["small_sample"]:
            actions.append(
                f"Review {item['department']} cautiously: unplanned rate {item['unplanned_absence_rate']:.1%} "
                f"on only {item['scheduled_days']} weekday rows (small sample)."
            )
        else:
            actions.append(
                f"Investigate {item['department']} further: unplanned rate {item['unplanned_absence_rate']:.1%} "
                f"({item['unplanned_absent']} of {item['scheduled_days']} weekday rows), "
                f"month-to-month change {item['month_change_pp']:+.2f} pp, "
                f"{item['anomaly_count']} statistical outlier month(s)."
            )
        grounded.append(item)
    largest_office = office_comparison.get("largest_increase")
    if largest_office and largest_office.get("change_pp", 0) > 0:
        actions.append(
            f"Compare staffing and coverage at {largest_office['location']}: unplanned absence rose "
            f"{largest_office['change_pp']:+.2f} pp from {largest_office['previous_rate']:.1%} "
            f"({largest_office['previous_absent']}/{largest_office['previous_scheduled']}) to "
            f"{largest_office['latest_rate']:.1%} "
            f"({largest_office['latest_absent']}/{largest_office['latest_scheduled']})."
        )
    if anomalies:
        first = anomalies[0]
        actions.append(
            f"Validate data quality for {first['department']} in {first['month']} "
            f"(z={first['zscore']}, rate {first['unplanned_absence_rate']:.1%}) before any personnel action."
        )
    actions.append(
        "Treat these as operational review prompts only. Do not infer misconduct or medical conditions from attendance rates alone."
    )
    prev = month_comparison.get("previous") or {}
    latest = month_comparison.get("latest") or {}
    return {
        "actions": actions,
        "grounded_in": grounded,
        "date_ranges": {
            "analytics_window": "current absence_analytics period",
            "month_comparison_previous": prev,
            "month_comparison_latest": latest,
        },
        "metric_definitions": {
            "unplanned_absence_rate": "absent + sick / stored weekday attendance rows",
            "month_change_pp": "latest complete-month rate minus previous complete-month rate, in percentage points",
            "anomaly_threshold": "z-score >= 1.5 among department-months with at least 20 scheduled days",
        },
    }


def _anomalies(monthly_rows: list[dict]) -> list[dict]:
    usable = [row for row in monthly_rows if row["scheduled_days"] >= 20]
    if len(usable) < 4:
        return []
    values = [row["unplanned_absence_rate"] for row in usable]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std < 1e-9:
        return []
    found = []
    for row in usable:
        zscore = (row["unplanned_absence_rate"] - mean) / std
        if zscore >= 1.5:
            found.append(
                {
                    "department": row["department"],
                    "month": row["month"],
                    "unplanned_absence_rate": row["unplanned_absence_rate"],
                    "scheduled_days": row["scheduled_days"],
                    "unplanned_absent": row["unplanned_absent"],
                    "zscore": round(zscore, 2),
                    "threshold": 1.5,
                    "comparison_population": "other department-months in this result with >= 20 scheduled days",
                    "note": "Statistical outlier against department-months in this period. The figure does not identify a cause.",
                }
            )
    found.sort(key=lambda item: -item["zscore"])
    return found


def _utilization(db: Session, employee_ids: list[int], dept_of: dict[int, str]) -> dict[str, float]:
    if not employee_ids:
        return {}
    balances = (
        db.query(LeaveBalance)
        .filter(LeaveBalance.employee_id.in_(employee_ids), LeaveBalance.leave_type == "annual", LeaveBalance.year == 2026)
        .all()
    )
    totals = defaultdict(lambda: {"entitled": 0.0, "used": 0.0})
    for balance in balances:
        department = dept_of.get(balance.employee_id)
        if department is None:
            continue
        totals[department]["entitled"] += balance.entitled_days
        totals[department]["used"] += balance.used_days
    return {
        name: round(bucket["used"] / bucket["entitled"], 4) if bucket["entitled"] else 0
        for name, bucket in totals.items()
    }


def _insights(departments: list[dict], anomalies: list[dict], org_rate: float, utilization: dict[str, float], scope: str) -> list[str]:
    if not departments:
        return ["No attendance rows were available for this scope and period."]
    if scope == "self":
        return ["This view is limited to your own attendance. Department comparisons are restricted to managers and HR administrators."]
    ranked = sorted(departments, key=lambda item: item["unplanned_absence_rate"], reverse=True)
    top = ranked[0]
    insights = [
        (
            f"{top['department']} has the highest unplanned absence rate in this result at "
            f"{top['unplanned_absence_rate']:.1%} ({top['unplanned_absent']} of {top['scheduled_days']} weekday records)."
        ),
        f"The scoped unplanned absence rate is {org_rate:.1%}.",
    ]
    if anomalies:
        first = anomalies[0]
        insights.append(
            f"{first['department']} in {first['month']} is an outlier with a z-score of {first['zscore']} and an unplanned rate of {first['unplanned_absence_rate']:.1%}."
        )
    else:
        insights.append("No department-month crossed the 1.5 z-score anomaly threshold in this period.")
    if utilization:
        highest = max(utilization.items(), key=lambda item: item[1])
        insights.append(
            f"Highest annual-leave utilization in this scope is {highest[0]} at {highest[1]:.0%} of entitled days. Utilization is not the same metric as unplanned absence."
        )
    insights.append("These figures describe the synthetic attendance extract. They are not a finding about employee conduct.")
    return insights
