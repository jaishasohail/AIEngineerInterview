"""Resolve manager natural-language approve/reject targets to leave request ids."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.access import can_decide_approval, direct_report_ids
from app.models import Approval, Employee, LeaveRequest, User


def nl_approval_resolution(
    db: Session,
    actor: User,
    manager: Employee,
    message: str,
    subject_employee_ids: list[int],
) -> tuple[dict | None, dict]:
    """Parse the message and optionally resolve a named direct-report target."""
    from app.ai.context import parse_approval_decision

    parsed = parse_approval_decision(message) or {}
    if not parsed.get("decision"):
        return None, parsed
    request_id = parsed.get("request_id")
    if request_id is not None:
        return {"status": "resolved", "request_id": int(request_id)}, parsed
    if parsed.get("resolve_direct_report") and not subject_employee_ids:
        auto_id = first_decidable_direct_report_leave_request(db, actor, manager)
        if auto_id is not None:
            return {"status": "resolved", "request_id": auto_id}, parsed
        return {
            "status": "not_found",
            "message": (
                "No eligible pending leave request was found for your direct reports "
                "that you are assigned to approve or reject."
            ),
        }, parsed
    if not parsed.get("resolve_by_name") and not subject_employee_ids:
        return None, parsed
    resolution = resolve_manager_approval_target(
        db,
        actor,
        manager,
        subject_employee_ids=subject_employee_ids,
        prefer_latest=bool(parsed.get("prefer_latest")),
    )
    return resolution, parsed


def _decidable_pending_for_subject(
    db: Session,
    actor: User,
    manager: Employee,
    subject_id: int,
) -> list[tuple[LeaveRequest, Approval]]:
    if actor.role == "manager" and subject_id not in direct_report_ids(db, manager.id):
        return []
    rows: list[tuple[LeaveRequest, Approval]] = []
    leaves = (
        db.query(LeaveRequest)
        .filter(
            LeaveRequest.employee_id == subject_id,
            LeaveRequest.status.in_(("pending_manager", "pending_hr")),
        )
        .order_by(LeaveRequest.submitted_at.desc(), LeaveRequest.id.desc())
        .all()
    )
    for leave in leaves:
        approval = (
            db.query(Approval)
            .filter(
                Approval.leave_request_id == leave.id,
                Approval.status == "pending",
                Approval.action_type == "leave",
            )
            .order_by(Approval.stage.asc(), Approval.id.asc())
            .first()
        )
        if approval is not None and can_decide_approval(actor, approval):
            rows.append((leave, approval))
    return rows


def first_decidable_direct_report_leave_request(
    db: Session,
    actor: User,
    manager: Employee,
) -> int | None:
    """Most recent pending leave the manager may decide for any direct report."""
    if actor.role != "manager":
        return None
    for subject_id in sorted(direct_report_ids(db, manager.id), reverse=True):
        candidates = _decidable_pending_for_subject(db, actor, manager, subject_id)
        if candidates:
            leave, _approval = candidates[0]
            return int(leave.id)
    return None


def first_decidable_leave_request(
    db: Session,
    actor: User,
    manager: Employee,
) -> int | None:
    """Most recent pending leave request the manager may approve or reject."""
    if actor.role != "manager":
        return None
    scoped = first_decidable_direct_report_leave_request(db, actor, manager)
    if scoped is not None:
        return scoped
    leaves = (
        db.query(LeaveRequest)
        .filter(LeaveRequest.status.in_(("pending_manager", "pending_hr")))
        .order_by(LeaveRequest.submitted_at.desc(), LeaveRequest.id.desc())
        .limit(40)
        .all()
    )
    for leave in leaves:
        approval = (
            db.query(Approval)
            .filter(
                Approval.leave_request_id == leave.id,
                Approval.status == "pending",
                Approval.action_type == "leave",
            )
            .order_by(Approval.stage.asc(), Approval.id.asc())
            .first()
        )
        if approval is not None and can_decide_approval(actor, approval):
            return int(leave.id)
    return None


def resolve_manager_approval_target(
    db: Session,
    actor: User,
    manager: Employee,
    *,
    subject_employee_ids: list[int],
    request_id: int | None = None,
    prefer_latest: bool = False,
) -> dict:
    """Map a named direct report (and optional disambiguation) to one leave request id."""
    if request_id is not None:
        return {"status": "resolved", "request_id": int(request_id)}
    if not subject_employee_ids:
        return {
            "status": "clarify",
            "message": "Name the employee whose leave request you want to approve or reject.",
        }
    if len(subject_employee_ids) > 1:
        return {
            "status": "clarify",
            "message": "Several employees were mentioned. Name one direct report whose leave you want to decide.",
        }
    subject_id = int(subject_employee_ids[0])
    subject = db.get(Employee, subject_id)
    if subject is None:
        return {"status": "not_found", "message": "That employee record was not found."}
    if actor.role == "manager" and subject_id not in direct_report_ids(db, manager.id):
        return {
            "status": "unauthorized",
            "message": (
                "You are not authorized to approve or reject leave for that employee. "
                "Managers may only decide pending leave for their direct reports."
            ),
        }
    candidates = _decidable_pending_for_subject(db, actor, manager, subject_id)
    if not candidates:
        return {
            "status": "not_found",
            "message": (
                f"No eligible pending leave request was found for {subject.full_name} "
                "that you are assigned to approve or reject."
            ),
        }
    if len(candidates) == 1 or prefer_latest:
        leave, _approval = candidates[0]
        return {
            "status": "resolved",
            "request_id": leave.id,
            "approval_id": _approval.id,
            "subject_employee_id": subject_id,
        }
    options = [
        {
            "request_id": leave.id,
            "approval_id": approval.id,
            "start": leave.start_date.isoformat(),
            "end": leave.end_date.isoformat(),
            "leave_type": leave.leave_type,
            "business_days": leave.business_days,
        }
        for leave, approval in candidates
    ]
    lines = [f"{item['request_id']}: {item['start']} to {item['end']} ({item['leave_type']})" for item in options]
    return {
        "status": "clarify",
        "message": (
            f"{subject.full_name} has {len(candidates)} pending leave requests. "
            f"Specify the request id. Pending: " + "; ".join(lines)
        ),
        "candidates": options,
        "subject_employee_id": subject_id,
    }
