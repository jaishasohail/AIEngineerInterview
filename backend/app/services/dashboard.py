from datetime import timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.access import employee_for_user
from app.clock import org_today
from app.models import Approval, Attendance, Department, Employee, LeaveRequest, User


def build_dashboard(db: Session, actor: User) -> dict:
    today = org_today()
    horizon = today + timedelta(days=14)
    actor_employee = employee_for_user(db, actor)
    employee_query = db.query(Employee).filter(Employee.employment_status == "active")
    if actor.role == "manager":
        employee_query = employee_query.filter(Employee.department_id == actor_employee.department_id)
    elif actor.role == "employee":
        employee_query = employee_query.filter(Employee.department_id == actor_employee.department_id)
    visible_employees = employee_query.all()
    visible_ids = [person.id for person in visible_employees]
    headcount = len(visible_ids) if actor.role != "hr_admin" else db.query(Employee).filter(Employee.employment_status == "active").count()

    leave_query = db.query(LeaveRequest).filter(
        LeaveRequest.status == "approved",
        LeaveRequest.start_date <= horizon,
        LeaveRequest.end_date >= today,
    )
    if actor.role == "manager":
        report_ids = [row[0] for row in db.query(Employee.id).filter(Employee.manager_id == actor_employee.id).all()]
        leave_query = leave_query.filter(LeaveRequest.employee_id.in_(report_ids or [0]))
    elif actor.role == "employee":
        leave_query = leave_query.filter(LeaveRequest.employee_id == actor_employee.id)
    out_rows = leave_query.all()

    approval_query = db.query(Approval).filter(Approval.status == "pending")
    if actor.role == "manager":
        approval_query = approval_query.filter(Approval.approver_employee_id == actor_employee.id)
    elif actor.role == "employee":
        approval_query = approval_query.filter(Approval.requested_by_user_id == actor.id)
    elif actor.role != "hr_admin":
        approval_query = approval_query.filter(Approval.id == -1)
    pending_count = approval_query.count()
    pending = approval_query.order_by(Approval.created_at.desc()).limit(6).all()

    start = today - timedelta(days=30)
    attendance_ids = visible_ids if actor.role != "hr_admin" else [row[0] for row in db.query(Employee.id).all()]
    attendance = (
        db.query(Attendance.status, func.count(Attendance.id))
        .filter(Attendance.employee_id.in_(attendance_ids or [0]), Attendance.work_date >= start, Attendance.work_date <= today)
        .group_by(Attendance.status)
        .all()
    )
    counts = {status: count for status, count in attendance}
    scheduled = sum(counts.values()) or 1
    presentish = counts.get("present", 0) + counts.get("remote", 0)
    departments = []
    dept_rows = db.query(Department).order_by(Department.name).all()
    for department in dept_rows:
        if actor.role != "hr_admin" and department.id != actor_employee.department_id:
            continue
        count = db.query(Employee).filter(Employee.department_id == department.id, Employee.employment_status == "active").count()
        departments.append({"department": department.name, "headcount": count, "location": department.location})

    month_expr = func.strftime("%Y-%m", Attendance.work_date) if db.get_bind().dialect.name == "sqlite" else func.to_char(Attendance.work_date, "YYYY-MM")
    monthly = (
        db.query(month_expr, Attendance.status, func.count(Attendance.id))
        .filter(Attendance.employee_id.in_(attendance_ids or [0]), Attendance.work_date >= today.replace(day=1) - timedelta(days=100), Attendance.work_date <= today)
        .group_by(month_expr, Attendance.status)
        .all()
    )
    month_bucket: dict[str, dict[str, int]] = {}
    for month, status, count in monthly:
        month_bucket.setdefault(month, {})
        month_bucket[month][status] = count
    monthly_absence = []
    for month, bucket in sorted(month_bucket.items()):
        total = sum(bucket.values()) or 1
        unplanned = bucket.get("absent", 0) + bucket.get("sick", 0)
        monthly_absence.append({"month": month, "unplanned_absence_rate": round(unplanned / total, 4), "scheduled_days": total})

    preview = []
    for approval in pending:
        subject = db.get(Employee, approval.subject_employee_id) if approval.subject_employee_id else None
        preview.append(
            {
                "id": approval.id,
                "action_type": approval.action_type,
                "status": approval.status,
                "subject": None if subject is None else subject.full_name,
                "created_at": approval.created_at.isoformat(),
            }
        )
    return {
        "org_clock": today.isoformat(),
        "scope": actor.role,
        "headcount": db.query(Employee).filter(Employee.employment_status == "active").count() if actor.role == "hr_admin" else headcount,
        "out_of_office_next_14_days": len({row.employee_id for row in out_rows}),
        "pending_approvals": pending_count,
        "attendance_rate": round(presentish / scheduled, 4),
        "departments": departments,
        "monthly_absence": monthly_absence,
        "pending_preview": preview,
        "simulated_system": "simulated_hris",
    }
