"""Authorization rules. These are enforced in Python services, never by the model."""

from sqlalchemy.orm import Session

from app.models import Employee, User

SENSITIVE_SALARY = "salary"
SENSITIVE_PERFORMANCE = "performance"
SENSITIVE_PERSONAL = "personal"


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
