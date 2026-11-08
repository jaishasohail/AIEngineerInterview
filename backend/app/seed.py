"""Deterministic Northstar Global directory. Same seed, same organization."""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy.orm import Session

from app.ai.ingest import ingest_policies
from app.clock import business_days_inclusive, org_today
from app.knowledge.holiday_seed import seed_holidays
from app.models import (
    Approval,
    Attendance,
    Department,
    Employee,
    LeaveBalance,
    LeaveRequest,
    Project,
    ProjectAssignment,
    User,
)
from app.security import hash_password

DEMO_PASSWORD = "Helios#2026"

DEPARTMENTS = [
    ("Engineering", "ENG", "Austin"),
    ("Product", "PRD", "London"),
    ("People Operations", "HR", "Berlin"),
    ("Finance", "FIN", "London"),
    ("Customer Operations", "CX", "Singapore"),
    ("Sales", "SAL", "Austin"),
    ("Legal & Compliance", "LGL", "Berlin"),
    ("Supply Chain", "SCM", "Singapore"),
]

TARGETS = {
    "Engineering": 22,
    "Product": 14,
    "People Operations": 10,
    "Finance": 16,
    "Customer Operations": 20,
    "Sales": 16,
    "Legal & Compliance": 10,
    "Supply Chain": 12,
}

FIRST = [
    "Maya", "Owen", "Lucas", "Emma", "Nina", "Hugo", "Iris", "Felix", "Ade", "Chloe",
    "Jonah", "Leila", "Samir", "Freya", "Theo", "Amina", "Carl", "Yara", "Ivan", "Nora",
    "Peter", "Hana", "Victor", "Ruth", "Oscar", "Dalia", "Quinn", "Mina", "Elliot", "Sara",
    "Malik", "Tessa", "Andre", "Lucia", "Chris", "Anika", "George", "Farah", "Henry", "Zola",
]
LAST = [
    "Bennett", "Owusu", "Keller", "Ibrahim", "Novak", "Silva", "Berg", "Costa", "Walsh", "Park",
    "Mensah", "Duval", "Schneider", "Abbas", "Moreau", "Grant", "Hoff", "Diallo", "Sato", "Reed",
    "Khan", "Vogelmann", "Andersen", "Bose", "Murphy", "Santos", "Nguyen", "Clark", "Haddad", "Meyer",
]


def _email(first: str, last: str, used: set[str]) -> str:
    base = f"{first}.{last}".lower().replace(" ", "").replace("'", "")
    candidate = f"{base}@northstar.example"
    suffix = 2
    while candidate in used:
        candidate = f"{base}{suffix}@northstar.example"
        suffix += 1
    used.add(candidate)
    return candidate


def reset_database(db: Session) -> None:
    for table in reversed(db.get_bind().metadata.sorted_tables if False else []):
        pass
    from app.database import Base

    for table in reversed(Base.metadata.sorted_tables):
        db.execute(table.delete())
    db.commit()


def seed(db: Session) -> dict:
    rng = random.Random(42)
    used_emails: set[str] = set()
    departments = {}
    for name, code, location in DEPARTMENTS:
        row = Department(name=name, code=code, location=location)
        db.add(row)
        departments[name] = row
    db.flush()

    people: dict[str, Employee] = {}
    number = 1000

    def add(key: str, first: str, last: str, dept: str, title: str, location: str, grade: str, salary: int, rating: float, hire: date) -> Employee:
        nonlocal number
        number += 1
        employee = Employee(
            employee_number=f"E-{number}",
            first_name=first,
            last_name=last,
            email=_email(first, last, used_emails),
            department_id=departments[dept].id,
            location=location,
            job_title=title,
            hire_date=hire,
            employment_status="active",
            grade=grade,
            salary_annual=salary,
            performance_rating=rating,
            phone=f"+1-555-01{number - 1000:02d}"[-16:],
        )
        db.add(employee)
        people[key] = employee
        return employee

    add("elena", "Elena", "Rossi", "Engineering", "VP Engineering", "Austin", "L7", 214000, 4.4, date(2017, 4, 3))
    add("james", "James", "Okonkwo", "Engineering", "Engineering Manager", "Austin", "L6", 168000, 4.2, date(2019, 2, 11))
    add("daniel", "Daniel", "Cho", "Engineering", "Engineering Manager", "Singapore", "L6", 164000, 4.1, date(2019, 8, 19))
    add("aisha", "Aisha", "Rahman", "Engineering", "Software Engineer", "Singapore", "L4", 118000, 4.3, date(2022, 3, 14))
    add("noah", "Noah", "Patel", "Engineering", "Software Engineer", "Austin", "L4", 128450, 4.6, date(2021, 6, 1))
    add("sofia", "Sofia", "Alvarez", "Engineering", "Software Engineer", "Austin", "L4", 116000, 4.0, date(2021, 11, 8))
    add("kenji", "Kenji", "Mori", "Engineering", "Software Engineer", "London", "L4", 121000, 3.9, date(2020, 5, 18))
    add("priya", "Priya", "Nair", "Engineering", "Software Engineer", "Singapore", "L5", 132000, 4.4, date(2019, 9, 2))
    add("hannah", "Hannah", "Brooks", "Engineering", "Software Engineer", "Austin", "L3", 98000, 3.8, date(2023, 1, 16))
    add("omar", "Omar", "Farouk", "Engineering", "Backend Engineer", "Berlin", "L4", 119000, 4.0, date(2022, 7, 25))
    add("grace", "Grace", "Liu", "Engineering", "Frontend Engineer", "Singapore", "L3", 101000, 3.9, date(2024, 2, 5))
    add("liam", "Liam", "Chen", "Engineering", "Software Engineer", "London", "L4", 124000, 4.1, date(2020, 1, 20))
    add("helena", "Helena", "Vogel", "People Operations", "HR Director", "Berlin", "L7", 198000, 4.5, date(2016, 9, 12))
    add("marcus", "Marcus", "Hale", "Finance", "Finance Director", "London", "L7", 206000, 4.2, date(2016, 3, 7))

    leadership = {
        "Product": ("Product Director", "Product Manager"),
        "People Operations": ("HR Director", "HR Partner"),
        "Finance": ("Finance Director", "Finance Manager"),
        "Customer Operations": ("CX Director", "CX Manager"),
        "Sales": ("Sales Director", "Sales Manager"),
        "Legal & Compliance": ("General Counsel", "Counsel"),
        "Supply Chain": ("Supply Chain Director", "Logistics Manager"),
    }
    ic_titles = {
        "Engineering": ["Software Engineer", "Backend Engineer", "QA Engineer"],
        "Product": ["Product Designer", "Product Analyst"],
        "People Operations": ["HR Generalist", "People Partner"],
        "Finance": ["Accountant", "Financial Analyst"],
        "Customer Operations": ["Support Lead", "Customer Specialist"],
        "Sales": ["Account Executive", "Sales Associate"],
        "Legal & Compliance": ["Legal Counsel", "Compliance Analyst"],
        "Supply Chain": ["Planner", "Buyer"],
    }
    locations = ["Austin", "Berlin", "London", "Singapore"]

    for dept_name, (head_title, manager_title) in leadership.items():
        if dept_name == "People Operations":
            head_key = "helena"
        elif dept_name == "Finance":
            head_key = "marcus"
        else:
            head_key = f"head-{dept_name}"
            add(head_key, rng.choice(FIRST), rng.choice(LAST), dept_name, head_title, departments[dept_name].location, "L7", 190000 + rng.randint(0, 8) * 1000, 4.2, date(2018, 1, 15))
        manager_key = f"mgr-{dept_name}"
        add(manager_key, rng.choice(FIRST), rng.choice(LAST), dept_name, manager_title, departments[dept_name].location, "L6", 150000 + rng.randint(0, 10) * 1000, 4.0, date(2019, 5, 6))
        people[f"lead-{dept_name}"] = people[head_key]

    def roster() -> list[Employee]:
        unique: dict[int, Employee] = {}
        for person in people.values():
            unique[id(person)] = person
        return list(unique.values())

    def count_in(dept: str) -> int:
        return sum(1 for person in roster() if person.department_id == departments[dept].id)

    for dept_name, target in TARGETS.items():
        while count_in(dept_name) < target:
            first = rng.choice(FIRST)
            last = rng.choice(LAST)
            title = rng.choice(ic_titles[dept_name])
            add(
                f"ic-{dept_name}-{count_in(dept_name)}",
                first,
                last,
                dept_name,
                title,
                rng.choice(locations),
                rng.choice(["L3", "L4", "L5"]),
                90000 + rng.randint(0, 40) * 1000,
                round(rng.uniform(3.2, 4.6), 1),
                date(2019, 1, 1) + timedelta(days=rng.randint(0, 2200)),
            )
    db.flush()

    james_team = ["aisha", "noah", "sofia", "kenji", "priya", "hannah", "omar", "grace"]
    for key in james_team:
        people[key].manager_id = people["james"].id
        people[key].approval_manager_id = people["james"].id
    people["james"].manager_id = people["elena"].id
    people["james"].approval_manager_id = people["elena"].id
    people["daniel"].manager_id = people["elena"].id
    people["daniel"].approval_manager_id = people["elena"].id
    people["liam"].manager_id = people["daniel"].id
    people["liam"].approval_manager_id = people["marcus"].id
    people["liam"].prior_department_id = departments["Finance"].id
    people["liam"].transfer_effective_date = date(2026, 10, 1)

    engineering_ids = {people[key].id for key in ["elena", "james", "daniel", *james_team, "liam"]}
    for person in list(people.values()):
        if person.department_id != departments["Engineering"].id or person.id in engineering_ids:
            continue
        if person.manager_id:
            continue
        person.manager_id = people["daniel"].id
        person.approval_manager_id = people["daniel"].id

    for dept_name in TARGETS:
        if dept_name == "Engineering":
            departments[dept_name].head_employee_id = people["elena"].id
            continue
        head = people[f"lead-{dept_name}"]
        manager = people[f"mgr-{dept_name}"]
        departments[dept_name].head_employee_id = head.id
        if manager.manager_id is None and manager.id != head.id:
            manager.manager_id = head.id
            manager.approval_manager_id = head.id
        for person in people.values():
            if person.department_id != departments[dept_name].id:
                continue
            if person.id in {head.id, manager.id}:
                continue
            if person.manager_id is None:
                person.manager_id = manager.id
                person.approval_manager_id = manager.id
    people["helena"].manager_id = None
    people["helena"].approval_manager_id = None
    people["marcus"].manager_id = None
    people["marcus"].approval_manager_id = None
    people["elena"].manager_id = None
    people["elena"].approval_manager_id = None
    db.flush()

    direct_reports: dict[int, int] = {}
    for person in roster():
        if person.manager_id:
            direct_reports[person.manager_id] = direct_reports.get(person.manager_id, 0) + 1
    for person in roster():
        if person.email == "helena.vogel@northstar.example":
            role = "hr_admin"
        elif direct_reports.get(person.id, 0) > 0:
            role = "manager"
        else:
            role = "employee"
        db.add(
            User(
                email=person.email,
                password_hash=hash_password(DEMO_PASSWORD),
                role=role,
                employee_id=person.id,
                is_active=True,
            )
        )

    special_used = {
        "aisha": (6, 0),
        "sofia": (3, None),
        "kenji": (2, None),
        "priya": (4, None),
    }
    pending_windows = {
        "sofia": (date(2026, 12, 7), date(2026, 12, 11)),
        "kenji": (date(2026, 12, 9), date(2026, 12, 15)),
        "priya": (date(2026, 12, 7), date(2026, 12, 18)),
    }
    for key, (start, end) in pending_windows.items():
        used, _pending = special_used[key]
        special_used[key] = (used, business_days_inclusive(start, end))

    for person in roster():
        used = rng.randint(0, 8)
        pending = 0
        for key, employee in people.items():
            if employee.id == person.id and key in special_used:
                used, pending_value = special_used[key]
                pending = 0 if pending_value is None else pending_value
                break
        db.add(LeaveBalance(employee_id=person.id, leave_type="annual", year=2026, entitled_days=24, used_days=used, pending_days=pending))
        db.add(LeaveBalance(employee_id=person.id, leave_type="sick", year=2026, entitled_days=10, used_days=rng.randint(0, 3), pending_days=0))
    db.flush()

    def project(code: str, name: str, dept: str, deadline: date, criticality: str, description: str) -> Project:
        row = Project(code=code, name=name, department_id=departments[dept].id, deadline=deadline, criticality=criticality, description=description, status="active")
        db.add(row)
        db.flush()
        return row

    atlas = project("ATL-1", "Project Atlas", "Engineering", date(2026, 12, 11), "critical", "Customer billing cutover. Deadline is fixed for 11 December 2026.")
    project("HEL-1", "Helios Platform", "Product", date(2027, 2, 20), "medium", "Internal platform release. Not on the Atlas critical path.")
    project("FIN-1", "Q1 Close", "Finance", date(2027, 1, 15), "critical", "Quarter close. Engineering is not assigned.")
    project("CX-1", "CX Stabilization", "Customer Operations", date(2026, 10, 20), "high", "Support backlog program that ended in October.")
    project("SAL-1", "Northstar Renewal", "Sales", date(2027, 3, 31), "high", "Enterprise renewal cycle.")
    for key, allocation, role_name in [
        ("james", 20, "Eng lead"),
        ("aisha", 80, "Engineer"),
        ("noah", 80, "Engineer"),
        ("sofia", 70, "Engineer"),
        ("kenji", 70, "Engineer"),
        ("priya", 60, "Engineer"),
        ("hannah", 50, "Engineer"),
    ]:
        db.add(ProjectAssignment(project_id=atlas.id, employee_id=people[key].id, role_name=role_name, allocation_pct=allocation))

    # Leave submission times are anchored to the organization clock so wall-clock
    # seeding never outranks newer org-clock submissions in "latest request" lookups.
    now = datetime.combine(org_today(), time(12, 0, 0), tzinfo=timezone.utc)
    users = {row.employee_id: row for row in db.query(User).all()}
    aisha_leave = LeaveRequest(
        employee_id=people["aisha"].id,
        leave_type="annual",
        start_date=date(2026, 8, 17),
        end_date=date(2026, 8, 24),
        business_days=6,
        status="approved",
        reason="Summer break",
        routed_manager_id=people["james"].id,
        hr_required=False,
        submitted_by_user_id=users[people["aisha"].id].id,
        submitted_at=now,
        resolved_at=now,
    )
    db.add(aisha_leave)

    for key, (start, end) in pending_windows.items():
        days = business_days_inclusive(start, end)
        request = LeaveRequest(
            employee_id=people[key].id,
            leave_type="annual",
            start_date=start,
            end_date=end,
            business_days=days,
            status="pending_manager",
            reason="Personal leave",
            routed_manager_id=people["james"].id,
            hr_required=True,
            submitted_by_user_id=users[people[key].id].id,
            submitted_at=now,
        )
        db.add(request)
        db.flush()
        db.add(
            Approval(
                action_type="leave",
                leave_request_id=request.id,
                subject_employee_id=people[key].id,
                stage=1,
                approver_employee_id=people["james"].id,
                approver_role="manager",
                status="pending",
                payload_json={"start": start.isoformat(), "end": end.isoformat(), "business_days": days},
                requested_by_user_id=users[people[key].id].id,
                created_at=now,
            )
        )

    upcoming = []
    for dept_name in ["Product", "Sales", "Customer Operations", "Finance", "Legal & Compliance", "Supply Chain"]:
        for person in people.values():
            if person.department_id == departments[dept_name].id and person.manager_id == people[f"mgr-{dept_name}"].id:
                upcoming.append(person)
                break
    for person in upcoming:
        request = LeaveRequest(
            employee_id=person.id,
            leave_type="annual",
            start_date=date(2026, 11, 9),
            end_date=date(2026, 11, 13),
            business_days=5,
            status="approved",
            reason="Planned leave",
            routed_manager_id=person.approval_manager_id,
            hr_required=False,
            submitted_by_user_id=users[person.id].id,
            submitted_at=now,
            resolved_at=now,
        )
        db.add(request)
        balance = db.query(LeaveBalance).filter(LeaveBalance.employee_id == person.id, LeaveBalance.leave_type == "annual").one()
        balance.used_days = max(balance.used_days, 5)

    db.flush()
    leave_days: dict[int, set[date]] = {}

    def mark(employee_id: int, start: date, end: date) -> None:
        cursor = start
        while cursor <= end:
            if cursor.weekday() < 5:
                leave_days.setdefault(employee_id, set()).add(cursor)
            cursor += timedelta(days=1)

    mark(people["aisha"].id, date(2026, 8, 17), date(2026, 8, 24))
    for person in upcoming:
        mark(person.id, date(2026, 11, 9), date(2026, 11, 13))

    rows = []
    cursor = date(2026, 8, 3)
    end = date(2026, 11, 6)
    employees = roster()
    dept_by_id = {department.id: department.name for department in departments.values()}
    while cursor <= end:
        if cursor.weekday() < 5:
            for index, person in enumerate(employees):
                local = random.Random(42_000 + person.id * 100 + cursor.toordinal())
                if cursor in leave_days.get(person.id, set()):
                    status = "leave"
                else:
                    dept = dept_by_id[person.department_id]
                    chance = 0.025
                    if dept == "Customer Operations" and date(2026, 9, 1) <= cursor <= date(2026, 10, 31):
                        chance = 0.12
                    elif dept == "Customer Operations":
                        chance = 0.04
                    roll = local.random()
                    if roll < chance:
                        status = "sick" if local.random() < 0.6 else "absent"
                    elif dept in {"Engineering", "Product"} and local.random() < 0.35:
                        status = "remote"
                    else:
                        status = "present"
                hours = 0 if status in {"leave", "absent", "sick"} else 8
                rows.append({"employee_id": person.id, "work_date": cursor, "status": status, "hours": hours})
        cursor += timedelta(days=1)
    db.bulk_insert_mappings(Attendance, rows)
    seed_holidays(db)
    db.commit()
    chunks = ingest_policies(db)
    return {
        "employees": len(roster()),
        "departments": len(departments),
        "attendance_rows": len(rows),
        "policy_chunks": chunks,
        "clock": org_today().isoformat(),
    }


def ensure_seeded(db: Session) -> None:
    if db.query(Employee).count() == 0:
        seed(db)
        return
    from app.models import Holiday

    if db.query(Holiday).count() == 0:
        seed_holidays(db)
        db.commit()
