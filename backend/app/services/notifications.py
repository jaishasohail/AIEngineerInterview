"""In-app HR workflow notifications.

Helios does not integrate an external email, SMS, or WhatsApp provider.
These records are durable in-app signals for managers/employees/HR that a
consequential leave workflow event occurred. delivery_status is always
'recorded' for in_app channel — never claim external delivery.
"""

from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.audit import write_audit
from app.clock import utcnow
from app.models import Employee, HrNotification, User


def record_notification(
    db: Session,
    *,
    recipient_employee_id: int,
    recipient_role: str,
    event_type: str,
    resource_type: str,
    resource_id: str | int,
    title: str,
    body: str,
    payload: dict | None = None,
    actor_user_id: int | None = None,
    ip_address: str | None = None,
) -> HrNotification | None:
    """Create an in-app notification, or return the existing duplicate."""
    resource_id_s = str(resource_id)
    existing = (
        db.query(HrNotification)
        .filter(
            HrNotification.event_type == event_type,
            HrNotification.resource_type == resource_type,
            HrNotification.resource_id == resource_id_s,
            HrNotification.recipient_employee_id == recipient_employee_id,
        )
        .one_or_none()
    )
    if existing is not None:
        return existing

    row = HrNotification(
        recipient_employee_id=recipient_employee_id,
        recipient_role=recipient_role,
        event_type=event_type,
        resource_type=resource_type,
        resource_id=resource_id_s,
        title=title[:180],
        body=body,
        channel="in_app",
        delivery_status="recorded",
        payload_json=payload or {},
        created_at=utcnow(),
    )
    # Savepoint so a duplicate unique conflict cannot roll back the parent HR write.
    try:
        with db.begin_nested():
            db.add(row)
            db.flush()
    except IntegrityError:
        return (
            db.query(HrNotification)
            .filter(
                HrNotification.event_type == event_type,
                HrNotification.resource_type == resource_type,
                HrNotification.resource_id == resource_id_s,
                HrNotification.recipient_employee_id == recipient_employee_id,
            )
            .one_or_none()
        )

    write_audit(
        db,
        actor_user_id=actor_user_id,
        action="hr_notification_recorded",
        resource_type="hr_notification",
        resource_id=str(row.id),
        outcome="success",
        details={
            "event_type": event_type,
            "recipient_employee_id": recipient_employee_id,
            "channel": "in_app",
            "delivery_status": "recorded",
            "external_provider": None,
        },
        ip_address=ip_address,
        commit=False,
    )
    return row


def notify_leave_submitted(
    db: Session,
    *,
    leave_request_id: int,
    employee: Employee,
    manager: Employee,
    business_days: float,
    actor_user_id: int | None,
    ip_address: str | None = None,
) -> list[HrNotification]:
    rows: list[HrNotification] = []
    manager_note = record_notification(
        db,
        recipient_employee_id=manager.id,
        recipient_role="manager",
        event_type="leave_submitted_manager",
        resource_type="leave_request",
        resource_id=leave_request_id,
        title=f"Leave request {leave_request_id} awaits your approval",
        body=(
            f"{employee.full_name} submitted a leave request for {business_days} business day(s). "
            "Open the Approvals inbox to review it. This is an in-app Helios notification; "
            "no external email was sent."
        ),
        payload={
            "leave_request_id": leave_request_id,
            "employee_id": employee.id,
            "business_days": business_days,
        },
        actor_user_id=actor_user_id,
        ip_address=ip_address,
    )
    if manager_note:
        rows.append(manager_note)
    return rows


def notify_leave_decision(
    db: Session,
    *,
    leave_request_id: int,
    employee: Employee,
    decision: str,
    status: str,
    actor_user_id: int | None,
    ip_address: str | None = None,
    hr_required_next: bool = False,
) -> list[HrNotification]:
    rows: list[HrNotification] = []
    employee_note = record_notification(
        db,
        recipient_employee_id=employee.id,
        recipient_role="employee",
        event_type=f"leave_{status}",
        resource_type="leave_request",
        resource_id=leave_request_id,
        title=f"Leave request {leave_request_id} is {status}",
        body=(
            f"Your leave request {leave_request_id} was marked {status} after a manager/HR {decision} decision. "
            "This is an in-app Helios notification; no external email was sent."
        ),
        payload={"leave_request_id": leave_request_id, "decision": decision, "status": status},
        actor_user_id=actor_user_id,
        ip_address=ip_address,
    )
    if employee_note:
        rows.append(employee_note)

    if hr_required_next:
        # Notify active HR admins that co-approval is required.
        hr_users = db.query(User).filter(User.role == "hr_admin", User.is_active.is_(True)).all()
        for hr in hr_users:
            note = record_notification(
                db,
                recipient_employee_id=hr.employee_id,
                recipient_role="hr_admin",
                event_type="leave_pending_hr",
                resource_type="leave_request",
                resource_id=leave_request_id,
                title=f"Leave request {leave_request_id} needs HR co-approval",
                body=(
                    f"Leave request {leave_request_id} for {employee.full_name} is pending People Operations "
                    "co-approval. This is an in-app Helios notification; no external email was sent."
                ),
                payload={"leave_request_id": leave_request_id, "employee_id": employee.id},
                actor_user_id=actor_user_id,
                ip_address=ip_address,
            )
            if note:
                rows.append(note)
    return rows


def list_notifications(
    db: Session,
    *,
    recipient_employee_id: int,
    limit: int = 50,
    unread_only: bool = False,
) -> list[dict]:
    query = db.query(HrNotification).filter(HrNotification.recipient_employee_id == recipient_employee_id)
    if unread_only:
        query = query.filter(HrNotification.read_at.is_(None))
    rows = query.order_by(HrNotification.created_at.desc()).limit(max(1, min(limit, 200))).all()
    return [
        {
            "id": row.id,
            "event_type": row.event_type,
            "resource_type": row.resource_type,
            "resource_id": row.resource_id,
            "title": row.title,
            "body": row.body,
            "channel": row.channel,
            "delivery_status": row.delivery_status,
            "external_provider": None,
            "payload": row.payload_json,
            "created_at": row.created_at.isoformat(),
            "read_at": row.read_at.isoformat() if row.read_at else None,
        }
        for row in rows
    ]
