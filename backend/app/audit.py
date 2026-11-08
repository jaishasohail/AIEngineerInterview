from sqlalchemy.orm import Session

from app.clock import utcnow
from app.models import AuditLog


def write_audit(
    db: Session,
    *,
    actor_user_id: int | None,
    action: str,
    resource_type: str,
    resource_id: str | None,
    outcome: str,
    details: dict | None = None,
    ip_address: str | None = None,
    commit: bool = True,
) -> AuditLog:
    safe_details = _scrub(details or {})
    row = AuditLog(
        actor_user_id=actor_user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome,
        details_json=safe_details,
        ip_address=ip_address,
        created_at=utcnow(),
    )
    db.add(row)
    if commit:
        db.commit()
    else:
        db.flush()
    return row


def _scrub(details: dict) -> dict:
    blocked = {"salary_annual", "password", "password_hash", "token", "authorization"}
    cleaned = {}
    for key, value in details.items():
        if key.lower() in blocked:
            cleaned[key] = "[redacted]"
        elif isinstance(value, dict):
            cleaned[key] = _scrub(value)
        else:
            cleaned[key] = value
    return cleaned
