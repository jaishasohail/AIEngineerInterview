"""State-changing HR actions. Every function re-checks authorization and business rules."""

from datetime import date

from sqlalchemy.orm import Session

from app.access import AuthorizationError, can_decide_approval, employee_for_user
from app.audit import write_audit
from app.clock import org_today, utcnow
from app.models import Approval, Employee, LeaveBalance, LeaveRequest, User
from app.services.leave_engine import LEAVE_YEAR, assess_window, get_balance, investigate_routing
from app.services.notifications import notify_leave_decision, notify_leave_submitted


def _reserve(balance: LeaveBalance, days: float) -> None:
    balance.pending_days = round(balance.pending_days + days, 2)


def _release(balance: LeaveBalance, days: float) -> None:
    balance.pending_days = round(max(0, balance.pending_days - days), 2)


def _consume(balance: LeaveBalance, days: float) -> None:
    _release(balance, days)
    balance.used_days = round(balance.used_days + days, 2)


def submit_leave(
    db: Session,
    actor: User,
    *,
    start: date,
    end: date,
    leave_type: str = "annual",
    reason: str = "Planned annual leave",
    ip_address: str | None = None,
) -> dict:
    employee = employee_for_user(db, actor)
    if actor.role not in {"employee", "manager", "hr_admin"}:
        raise AuthorizationError("This account cannot submit leave.", resource="leave", action="submit")
    assessment = assess_window(db, employee, start, end, leave_type=leave_type)
    if assessment["hard_blocks"]:
        write_audit(
            db,
            actor_user_id=actor.id,
            action="leave_submit_blocked",
            resource_type="leave_request",
            resource_id=None,
            outcome="denied",
            details={"reasons": assessment["hard_blocks"], "start": start.isoformat(), "end": end.isoformat()},
            ip_address=ip_address,
        )
        return {"status": "blocked", "message": " ".join(assessment["hard_blocks"]), "summary": "The request was not created.", "assessment": assessment}

    balance = get_balance(db, employee.id, leave_type, LEAVE_YEAR)
    if balance is None:
        raise AuthorizationError("No leave balance is recorded for this year.", resource="leave_balance")
    manager = db.get(Employee, employee.approval_manager_id)
    if manager is None:
        raise AuthorizationError("No approval manager is available.", resource="employee")

    now = utcnow()
    request = LeaveRequest(
        employee_id=employee.id,
        leave_type=leave_type,
        start_date=start,
        end_date=end,
        business_days=assessment["business_days"],
        status="pending_manager",
        reason=reason.strip() or "Planned annual leave",
        routed_manager_id=manager.id,
        hr_required=assessment["hr_required"],
        impact_json=assessment,
        submitted_by_user_id=actor.id,
        submitted_at=now,
    )
    db.add(request)
    db.flush()
    approval = Approval(
        action_type="leave",
        leave_request_id=request.id,
        subject_employee_id=employee.id,
        stage=1,
        approver_employee_id=manager.id,
        approver_role="manager",
        status="pending",
        payload_json={"start": start.isoformat(), "end": end.isoformat(), "business_days": assessment["business_days"]},
        impact_json=assessment,
        requested_by_user_id=actor.id,
        created_at=now,
    )
    db.add(approval)
    _reserve(balance, assessment["business_days"])
    notifications = notify_leave_submitted(
        db,
        leave_request_id=request.id,
        employee=employee,
        manager=manager,
        business_days=assessment["business_days"],
        actor_user_id=actor.id,
        ip_address=ip_address,
    )
    write_audit(
        db,
        actor_user_id=actor.id,
        action="leave_submitted",
        resource_type="leave_request",
        resource_id=str(request.id),
        outcome="success",
        details={
            "routed_manager_id": manager.id,
            "hr_required": assessment["hr_required"],
            "business_days": assessment["business_days"],
            "system": "simulated_hris",
            "manager_notification_ids": [item.id for item in notifications],
            "notification_channel": "in_app",
        },
        ip_address=ip_address,
        commit=False,
    )
    db.commit()
    message = (
        f"Leave request {request.id} is submitted in the simulated HRIS and routed to {manager.full_name}. "
        f"Status is pending manager approval. "
        f"An in-app manager notification was recorded (no external email provider is configured)."
    )
    if assessment["hr_required"]:
        message += " People Operations co-approval will be required after the manager decision because of the policy rules."
    return {
        "status": "submitted",
        "message": message,
        "summary": f"Submitted request {request.id} for {assessment['business_days']} business days. No approval was granted by the assistant.",
        "leave_request_id": request.id,
        "approval_id": approval.id,
        "routed_to": manager.full_name,
        "hr_required": assessment["hr_required"],
        "notifications": [{"id": item.id, "event_type": item.event_type, "channel": item.channel} for item in notifications],
    }


def file_routing_correction(
    db: Session,
    actor: User,
    *,
    correction: dict | None = None,
    subject_employee_id: int | None = None,
    ip_address: str | None = None,
) -> dict:
    """File a pending People Ops data_correction for a routing mismatch.

    HR admins may file for a subject employee identified in the pending payload.
    Non-HR actors may only file against their own employee record.
    """
    actor_employee = employee_for_user(db, actor)
    subject_id = subject_employee_id or (correction or {}).get("subject_employee_id")
    if subject_id is None:
        subject_id = actor_employee.id
    subject_id = int(subject_id)
    if actor.role != "hr_admin" and subject_id != actor_employee.id:
        raise AuthorizationError(
            "You can only file a routing correction for your own employee record.",
            resource="employee",
            action="update",
        )
    subject = db.get(Employee, subject_id)
    if subject is None:
        return {
            "status": "noop",
            "message": "The subject employee record no longer exists. Nothing was filed.",
            "summary": "Subject employee missing.",
        }
    finding = investigate_routing(db, subject)
    live_correction = finding.get("correction")
    if finding["consistent"] or live_correction is None:
        return {
            "status": "noop",
            "message": (
                f"Approval routing for {subject.full_name} already matches the current line manager. "
                "Nothing was filed."
            ),
            "summary": "No correction was required.",
            "subject_employee_id": subject.id,
        }
    # Prefer the live DB correction; still require the pending payload to target the same subject/to-manager.
    if correction:
        pending_subject = correction.get("subject_employee_id")
        pending_to = correction.get("to_employee_id")
        if pending_subject is not None and int(pending_subject) != subject.id:
            return {
                "status": "noop",
                "message": "The confirmed correction target no longer matches the pending subject. Nothing was filed.",
                "summary": "Pending subject mismatch.",
                "subject_employee_id": subject.id,
            }
        if pending_to is not None and int(pending_to) != int(live_correction.get("to_employee_id") or 0):
            return {
                "status": "noop",
                "message": (
                    "The recommended approval manager changed since this confirmation was prepared. "
                    "Nothing was filed. Re-run the investigation and confirm again."
                ),
                "summary": "Stale correction payload.",
                "subject_employee_id": subject.id,
            }
    correction = live_correction
    existing = (
        db.query(Approval)
        .filter(
            Approval.action_type == "data_correction",
            Approval.subject_employee_id == subject.id,
            Approval.status == "pending",
        )
        .one_or_none()
    )
    if existing:
        return {
            "status": "submitted",
            "message": (
                f"A correction request is already waiting for People Operations (approval {existing.id}). "
                "The employee record was not changed."
            ),
            "summary": "Existing correction request reused.",
            "approval_id": existing.id,
            "subject_employee_id": subject.id,
        }
    approval = Approval(
        action_type="data_correction",
        subject_employee_id=subject.id,
        stage=1,
        approver_employee_id=None,
        approver_role="hr_admin",
        status="pending",
        payload_json=correction,
        impact_json=finding,
        requested_by_user_id=actor.id,
        created_at=utcnow(),
    )
    db.add(approval)
    db.flush()
    write_audit(
        db,
        actor_user_id=actor.id,
        action="routing_correction_requested",
        resource_type="approval",
        resource_id=str(approval.id),
        outcome="success",
        details={
            "employee_id": subject.id,
            "from": correction["from_employee_id"],
            "to": correction["to_employee_id"],
            "requested_by_role": actor.role,
        },
        ip_address=ip_address,
        commit=False,
    )
    db.commit()
    return {
        "status": "submitted",
        "message": (
            f"Correction request {approval.id} was sent to People Operations for {subject.full_name}. "
            f"It asks to move approval routing from {correction['from_name']} to {correction['to_name']}. "
            "The official record is unchanged until an HR administrator approves it."
        ),
        "summary": "HR correction filed. Record not modified.",
        "approval_id": approval.id,
        "subject_employee_id": subject.id,
    }


def _can_decide(actor: User, approval: Approval) -> bool:
    return can_decide_approval(actor, approval)


def decide_approval(
    db: Session,
    actor: User,
    approval_id: int,
    decision: str,
    note: str = "",
    *,
    ip_address: str | None = None,
) -> dict:
    if decision not in {"approve", "reject"}:
        raise AuthorizationError("Decision must be approve or reject.", resource="approval", action="decide")
    approval = db.get(Approval, approval_id)
    if approval is None:
        raise AuthorizationError("Approval not found.", resource=f"approval:{approval_id}", action="decide")
    if approval.status != "pending":
        raise AuthorizationError("This approval is no longer pending.", resource=f"approval:{approval_id}", action="decide")
    if not _can_decide(actor, approval):
        write_audit(
            db,
            actor_user_id=actor.id,
            action="approval_decide",
            resource_type="approval",
            resource_id=str(approval.id),
            outcome="denied",
            details={"decision": decision},
            ip_address=ip_address,
        )
        raise AuthorizationError("You are not the assigned approver for this request.", resource=f"approval:{approval.id}", action="decide")

    now = utcnow()
    approval.status = "approved" if decision == "approve" else "rejected"
    approval.decision_note = note.strip() or None
    approval.decided_by_user_id = actor.id
    approval.decided_at = now

    if approval.action_type == "leave":
        result = _decide_leave(db, actor, approval, decision, now, ip_address=ip_address)
    elif approval.action_type == "data_correction":
        result = _decide_correction(db, actor, approval, decision)
    else:
        raise AuthorizationError("Unsupported approval type.", resource=f"approval:{approval.id}")

    write_audit(
        db,
        actor_user_id=actor.id,
        action=f"approval_{decision}",
        resource_type="approval",
        resource_id=str(approval.id),
        outcome="success",
        details={
            "action_type": approval.action_type,
            "result": result["status"],
            "leave_request_id": result.get("leave_request_id") or approval.leave_request_id,
            "system": "simulated_hris",
            "notification_ids": result.get("notification_ids") or [],
            "notification_channel": "in_app",
        },
        ip_address=ip_address,
        commit=False,
    )
    db.commit()
    return result


def _decide_leave(
    db: Session,
    actor: User,
    approval: Approval,
    decision: str,
    now,
    *,
    ip_address: str | None = None,
) -> dict:
    request = db.get(LeaveRequest, approval.leave_request_id)
    if request is None:
        raise AuthorizationError("Leave request missing.", resource="leave_request")
    balance = get_balance(db, request.employee_id, request.leave_type, LEAVE_YEAR)
    if balance is None:
        raise AuthorizationError("Leave balance missing.", resource="leave_balance")
    employee = db.get(Employee, request.employee_id)
    if employee is None:
        raise AuthorizationError("Employee record missing.", resource="employee")
    if decision == "reject":
        request.status = "rejected"
        request.resolved_at = now
        _release(balance, request.business_days)
        notes = notify_leave_decision(
            db,
            leave_request_id=request.id,
            employee=employee,
            decision=decision,
            status="rejected",
            actor_user_id=actor.id,
            ip_address=ip_address,
        )
        _write_leave_decision_audit(
            db,
            actor=actor,
            request=request,
            approval=approval,
            decision=decision,
            status="rejected",
            notification_ids=[item.id for item in notes],
            ip_address=ip_address,
        )
        return {
            "status": "rejected",
            "message": (
                f"Leave request {request.id} was rejected. Reserved days were released. "
                "The assistant did not make this decision. "
                "An in-app employee notification was recorded (no external email provider is configured)."
            ),
            "leave_request_id": request.id,
            "notification_ids": [item.id for item in notes],
        }
    if approval.stage == 1 and request.hr_required:
        request.status = "pending_hr"
        db.add(
            Approval(
                action_type="leave",
                leave_request_id=request.id,
                subject_employee_id=request.employee_id,
                stage=2,
                approver_employee_id=None,
                approver_role="hr_admin",
                status="pending",
                payload_json=approval.payload_json,
                impact_json=request.impact_json,
                requested_by_user_id=request.submitted_by_user_id,
                created_at=now,
            )
        )
        notes = notify_leave_decision(
            db,
            leave_request_id=request.id,
            employee=employee,
            decision=decision,
            status="pending_hr",
            actor_user_id=actor.id,
            ip_address=ip_address,
            hr_required_next=True,
        )
        _write_leave_decision_audit(
            db,
            actor=actor,
            request=request,
            approval=approval,
            decision=decision,
            status="pending_hr",
            notification_ids=[item.id for item in notes],
            ip_address=ip_address,
        )
        return {
            "status": "pending_hr",
            "message": (
                f"Manager approval recorded for request {request.id}. It now waits for People Operations "
                "because a policy rule requires co-approval. Balance stays reserved. "
                "In-app employee and HR notifications were recorded (no external email provider is configured)."
            ),
            "leave_request_id": request.id,
            "notification_ids": [item.id for item in notes],
        }
    request.status = "approved"
    request.resolved_at = now
    _consume(balance, request.business_days)
    notes = notify_leave_decision(
        db,
        leave_request_id=request.id,
        employee=employee,
        decision=decision,
        status="approved",
        actor_user_id=actor.id,
        ip_address=ip_address,
    )
    _write_leave_decision_audit(
        db,
        actor=actor,
        request=request,
        approval=approval,
        decision=decision,
        status="approved",
        notification_ids=[item.id for item in notes],
        ip_address=ip_address,
    )
    return {
        "status": "approved",
        "message": (
            f"Leave request {request.id} is approved in the simulated HRIS. "
            f"{request.business_days} days moved from pending to used. "
            "An in-app employee notification was recorded (no external email provider is configured)."
        ),
        "leave_request_id": request.id,
        "notification_ids": [item.id for item in notes],
    }


def _write_leave_decision_audit(
    db: Session,
    *,
    actor: User,
    request: LeaveRequest,
    approval: Approval,
    decision: str,
    status: str,
    notification_ids: list[int],
    ip_address: str | None = None,
) -> None:
    """Dedicated leave_request audit for approve/reject outcomes (alongside approval_*)."""
    write_audit(
        db,
        actor_user_id=actor.id,
        action=f"leave_{status}",
        resource_type="leave_request",
        resource_id=str(request.id),
        outcome="success",
        details={
            "decision": decision,
            "status": status,
            "employee_id": request.employee_id,
            "leave_request_id": request.id,
            "approval_id": approval.id,
            "business_days": request.business_days,
            "system": "simulated_hris",
            "notification_ids": notification_ids,
            "notification_channel": "in_app",
        },
        ip_address=ip_address,
        commit=False,
    )


def _decide_correction(db: Session, actor: User, approval: Approval, decision: str) -> dict:
    if actor.role != "hr_admin":
        raise AuthorizationError("Only HR administrators can change official reporting routes.", resource="employee", action="update")
    employee = db.get(Employee, approval.subject_employee_id)
    if employee is None:
        raise AuthorizationError("Employee record missing.", resource="employee")
    if decision == "reject":
        return {"status": "rejected", "message": f"Correction {approval.id} was rejected. {employee.full_name}'s record was not changed.", "approval_id": approval.id}
    payload = approval.payload_json or {}
    target_id = payload.get("to_employee_id")
    target = db.get(Employee, target_id) if target_id else None
    if target is None:
        raise AuthorizationError("Correction target no longer exists.", resource="employee")
    employee.approval_manager_id = target.id
    open_requests = (
        db.query(LeaveRequest)
        .filter(LeaveRequest.employee_id == employee.id, LeaveRequest.status.in_(("pending_manager", "pending_hr")))
        .all()
    )
    for request in open_requests:
        if request.status == "pending_manager":
            request.routed_manager_id = target.id
            pending = (
                db.query(Approval)
                .filter(Approval.leave_request_id == request.id, Approval.status == "pending", Approval.approver_role == "manager")
                .all()
            )
            for item in pending:
                item.approver_employee_id = target.id
    return {
        "status": "approved",
        "message": f"Approval routing for {employee.full_name} now points to {target.full_name}. Open manager-stage requests were moved with it.",
        "approval_id": approval.id,
    }
