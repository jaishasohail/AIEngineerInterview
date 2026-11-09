"""Authorization rules. These are enforced in Python services, never by the model."""

from sqlalchemy.orm import Session

from app.models import Approval, Employee, LeaveRequest, User

SENSITIVE_SALARY = "salary"
SENSITIVE_PERFORMANCE = "performance"
SENSITIVE_PERSONAL = "personal"
SENSITIVE_ATTENDANCE = "attendance"

# Arguments an LLM tool call must never be allowed to override.
FORBIDDEN_TOOL_ARG_KEYS = frozenset(
    {
        "employee_id",
        "user_id",
        "actor_user_id",
        "owner_user_id",
        "role",
        "actor_role",
        "access_token",
        "password",
        "jwt",
        "api_key",
        "secret",
        "conversation_id",
        "pending_action_id",
    }
)


class AuthorizationError(Exception):
    def __init__(self, message: str, *, resource: str = "", action: str = "read") -> None:
        super().__init__(message)
        self.message = message
        self.resource = resource
        self.action = action


def employee_for_user(db: Session, user: User) -> Employee:
    employee = db.get(Employee, user.employee_id)
    if employee is None:
        raise AuthorizationError("Your account is not linked to an employee record.", resource="user")
    return employee


def is_manager_of(db: Session, manager: Employee, subject: Employee) -> bool:
    cursor = subject
    seen: set[int] = set()
    while cursor.manager_id and cursor.manager_id not in seen:
        if cursor.manager_id == manager.id:
            return True
        seen.add(cursor.id)
        parent = db.get(Employee, cursor.manager_id)
        if parent is None:
            return False
        cursor = parent
    return False


def direct_report_ids(db: Session, manager_id: int) -> list[int]:
    rows = db.query(Employee.id).filter(Employee.manager_id == manager_id).all()
    return [row[0] for row in rows]


def sanitize_tool_arguments(arguments: dict | None) -> dict:
    """Drop identity/privilege keys from LLM-supplied tool arguments."""
    if not arguments:
        return {}
    cleaned: dict = {}
    for key, value in arguments.items():
        if str(key).lower() in FORBIDDEN_TOOL_ARG_KEYS:
            continue
        cleaned[key] = value
    return cleaned


def can_view_employee_record(db: Session, actor: User, actor_employee: Employee, subject: Employee) -> bool:
    """Private employee profile access (not the public directory card)."""
    if actor.role == "hr_admin" or actor.employee_id == subject.id:
        return True
    if actor.role == "manager" and is_manager_of(db, actor_employee, subject):
        return True
    return False


def can_view_attendance(db: Session, actor: User, actor_employee: Employee, subject: Employee) -> bool:
    if actor.role == "hr_admin" or actor.employee_id == subject.id:
        return True
    if actor.role == "manager" and is_manager_of(db, actor_employee, subject):
        return True
    return False


def can_decide_approval(actor: User, approval: Approval) -> bool:
    """Deterministic approver check — never trust the model for this.

    Rules (API, chatbot authorize, tools, and confirmation all share this):
    - Employees never approve or reject (including their own leave).
    - Nobody may decide an approval where they are the subject employee.
    - HR Admin may decide HR-stage items and may override manager-stage leave decisions.
    - Managers may only decide items assigned to them as approver_employee_id.
    """
    if actor.role == "employee":
        return False
    if approval.subject_employee_id and actor.employee_id == approval.subject_employee_id:
        return False
    if approval.approver_role == "hr_admin":
        return actor.role == "hr_admin"
    if actor.role == "hr_admin":
        return True
    if actor.role != "manager":
        return False
    return actor.employee_id is not None and actor.employee_id == approval.approver_employee_id


def resolve_leave_approval_reference(
    db: Session,
    *,
    request_id: int | None = None,
    approval_id: int | None = None,
    pending_only: bool = False,
) -> Approval | None:
    """Resolve an Approval row for approve/reject flows.

    When ``request_id`` is set it always refers to ``leave_requests.id``, never an approval id.
    """
    if approval_id is not None:
        row = db.get(Approval, int(approval_id))
        if row is None:
            return None
        if pending_only and row.status != "pending":
            return None
        return row
    if request_id is None:
        return None
    leave = db.get(LeaveRequest, int(request_id))
    if leave is None:
        return None
    pending = (
        db.query(Approval)
        .filter(Approval.leave_request_id == leave.id, Approval.status == "pending")
        .order_by(Approval.stage.asc(), Approval.id.asc())
        .first()
    )
    if pending is not None:
        return pending
    if pending_only:
        return None
    return (
        db.query(Approval)
        .filter(Approval.leave_request_id == leave.id)
        .order_by(Approval.id.desc())
        .first()
    )


def approval_decision_access(
    db: Session,
    actor: User,
    *,
    request_id: int | None = None,
    approval_id: int | None = None,
) -> tuple[bool, str]:
    """Authoritative gate for chat and REST before preparing or executing a decision.

    When both ids are omitted the caller may still need clarification (allowed).
    """
    if actor.role == "employee":
        return (
            False,
            "Employees cannot approve or reject leave requests, including their own. "
            "Approval decisions are limited to the assigned manager or an HR administrator.",
        )
    if actor.role not in {"manager", "hr_admin"}:
        return False, "Only managers and HR administrators can approve or reject leave requests."
    if request_id is None and approval_id is None:
        return True, ""
    approval = resolve_leave_approval_reference(
        db, request_id=request_id, approval_id=approval_id, pending_only=True
    )
    if approval is None:
        if request_id is not None:
            leave = db.get(LeaveRequest, int(request_id))
            if leave is None:
                return False, f"Leave request {request_id} was not found."
        return False, "No pending approval was found for that leave request."
    if approval.status != "pending":
        return False, f"Approval {approval.id} is already {approval.status} and cannot be decided again."
    if not can_decide_approval(actor, approval):
        return (
            False,
            "You are not authorized to approve or reject this leave request. "
            "Managers may only decide requests assigned to them within their reporting scope.",
        )
    return True, ""


def scoped_employee_ids(db: Session, actor: User, actor_employee: Employee) -> list[int] | None:
    """Employee ids visible for attendance/analytics aggregates.

    None means organization-wide (HR Admin only).
    """
    if actor.role == "hr_admin":
        return None
    if actor.role == "manager":
        # Manager analytics/attendance aggregates are limited to authorized direct reports.
        # Do not include the manager's own id in the aggregate scope.
        return sorted(direct_report_ids(db, actor_employee.id))
    return [actor_employee.id]


def can_view_directory(actor: User, _subject: Employee) -> bool:
    return actor.is_active


def can_view_salary(actor: User, subject: Employee) -> bool:
    if actor.role == "hr_admin":
        return True
    return actor.employee_id == subject.id


def can_view_performance(db: Session, actor: User, actor_employee: Employee, subject: Employee) -> bool:
    if actor.role == "hr_admin" or actor.employee_id == subject.id:
        return True
    if actor.role == "manager" and is_manager_of(db, actor_employee, subject):
        return True
    return False


def can_view_personal(actor: User, subject: Employee) -> bool:
    return actor.role == "hr_admin" or actor.employee_id == subject.id


def can_view_leave(db: Session, actor: User, actor_employee: Employee, subject: Employee) -> bool:
    if actor.role == "hr_admin" or actor.employee_id == subject.id:
        return True
    if actor.role == "manager" and is_manager_of(db, actor_employee, subject):
        return True
    return False


def can_view_team_availability(db: Session, actor: User, actor_employee: Employee, subject: Employee) -> bool:
    """Dates only, for people who share a manager. Reasons stay with the manager and HR."""
    if can_view_leave(db, actor, actor_employee, subject):
        return True
    if actor_employee.manager_id and actor_employee.manager_id == subject.manager_id:
        return True
    if subject.id == actor_employee.manager_id:
        return True
    return False


def redact_employee(
    db: Session,
    actor: User,
    actor_employee: Employee,
    subject: Employee,
    department_name: str,
    *,
    include_private: bool,
) -> dict:
    manager = db.get(Employee, subject.manager_id) if subject.manager_id else None
    approval_manager = db.get(Employee, subject.approval_manager_id) if subject.approval_manager_id else None
    profile = {
        "id": subject.id,
        "employee_number": subject.employee_number,
        "full_name": subject.full_name,
        "first_name": subject.first_name,
        "last_name": subject.last_name,
        "email": subject.email,
        "job_title": subject.job_title,
        "department": department_name,
        "department_id": subject.department_id,
        "location": subject.location,
        "employment_status": subject.employment_status,
    }
    if not include_private:
        return profile

    if actor.employee_id == subject.id or actor.role in {"manager", "hr_admin"}:
        profile.update(
            {
                "hire_date": subject.hire_date.isoformat(),
                "grade": subject.grade,
                "manager": None
                if manager is None
                else {"id": manager.id, "full_name": manager.full_name, "job_title": manager.job_title},
                "approval_manager": None
                if approval_manager is None
                else {
                    "id": approval_manager.id,
                    "full_name": approval_manager.full_name,
                    "job_title": approval_manager.job_title,
                },
                "routing_consistent": subject.manager_id == subject.approval_manager_id,
                "prior_department_id": subject.prior_department_id,
                "transfer_effective_date": subject.transfer_effective_date.isoformat()
                if subject.transfer_effective_date
                else None,
            }
        )
    if can_view_personal(actor, subject):
        profile["phone"] = subject.phone
    if can_view_performance(db, actor, actor_employee, subject):
        profile["performance_rating"] = subject.performance_rating
    if can_view_salary(actor, subject):
        profile["salary_annual"] = subject.salary_annual
    return profile


def assert_sensitive_access(
    db: Session,
    actor: User,
    actor_employee: Employee,
    subject: Employee,
    kinds: set[str],
) -> None:
    if SENSITIVE_SALARY in kinds and not can_view_salary(actor, subject):
        raise AuthorizationError(
            "You are not allowed to view that employee's salary.",
            resource=f"employee:{subject.id}",
            action="read_salary",
        )
    if SENSITIVE_PERFORMANCE in kinds and not can_view_performance(db, actor, actor_employee, subject):
        raise AuthorizationError(
            "You are not allowed to view that employee's performance rating.",
            resource=f"employee:{subject.id}",
            action="read_performance",
        )
    if SENSITIVE_PERSONAL in kinds and not can_view_personal(actor, subject):
        raise AuthorizationError(
            "You are not allowed to view that employee's personal contact details.",
            resource=f"employee:{subject.id}",
            action="read_personal",
        )
    if SENSITIVE_ATTENDANCE in kinds and not can_view_attendance(db, actor, actor_employee, subject):
        raise AuthorizationError(
            "You are not allowed to view that employee's attendance records.",
            resource=f"employee:{subject.id}",
            action="read_attendance",
        )


def assert_employee_record_access(
    db: Session,
    actor: User,
    actor_employee: Employee,
    subject: Employee,
) -> None:
    if not can_view_employee_record(db, actor, actor_employee, subject):
        raise AuthorizationError(
            "You cannot open that employee record.",
            resource=f"employee:{subject.id}",
            action="read",
        )
