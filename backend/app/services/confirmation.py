"""Human-in-the-loop pending actions.

Persists consequential HR operations until the owning user explicitly confirms.
Execution always re-checks authorization and current database state.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.access import AuthorizationError, employee_for_user
from app.audit import write_audit
from app.clock import utcnow
from app.models import Approval, Conversation, Employee, LeaveRequest, PendingAction, User
from app.services.actions import decide_approval, file_routing_correction, submit_leave
from app.services.leave_engine import assess_window

PENDING_TTL_MINUTES = 30
ACTIVE_STATUSES = ("awaiting_confirmation",)
TERMINAL_STATUSES = ("executed", "cancelled", "expired", "failed", "blocked")


class ConfirmationError(Exception):
    def __init__(self, message: str, *, status: str = "blocked", details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.details = details or {}


def _expiry(now: datetime | None = None) -> datetime:
    return (now or utcnow()) + timedelta(minutes=PENDING_TTL_MINUTES)


def idempotency_for_submit(*, employee_id: int, start: str, end: str, leave_type: str) -> str:
    return f"submit_leave:{employee_id}:{start}:{end}:{leave_type}"


def idempotency_for_decision(*, approval_id: int, decision: str) -> str:
    return f"decide_approval:{approval_id}:{decision}"


def idempotency_for_routing(*, employee_id: int, to_manager_id: int) -> str:
    return f"file_routing_correction:{employee_id}:{to_manager_id}"


def public_pending(row: PendingAction) -> dict:
    payload = dict(row.payload_json or {})
    return {
        **payload,
        "pending_action_id": row.id,
        "action_type": row.action_type,
        "status": row.status,
        "owner_user_id": row.owner_user_id,
        "conversation_id": row.conversation_id,
        "idempotency_key": row.idempotency_key,
        "expires_at": row.expires_at.isoformat(),
        "workflow_state": payload.get("workflow_state") or "AWAITING_CONFIRMATION",
    }


def get_pending(db: Session, pending_id: int | None) -> PendingAction | None:
    if pending_id is None:
        return None
    return db.get(PendingAction, int(pending_id))


def expire_if_needed(db: Session, row: PendingAction, *, ip: str | None = None) -> PendingAction:
    if row.status in ACTIVE_STATUSES and row.expires_at <= utcnow():
        row.status = "expired"
        row.updated_at = utcnow()
        write_audit(
            db,
            actor_user_id=row.owner_user_id,
            action="pending_action_expired",
            resource_type="pending_action",
            resource_id=str(row.id),
            outcome="expired",
            details={"action_type": row.action_type, "idempotency_key": row.idempotency_key},
            ip_address=ip,
            commit=False,
        )
        db.flush()
    return row


def assert_owner(row: PendingAction, user: User) -> None:
    if row.owner_user_id != user.id:
        raise ConfirmationError(
            "You cannot confirm another user's pending action.",
            status="denied",
            details={"owner_user_id": row.owner_user_id, "actor_user_id": user.id},
        )


def create_or_reuse_pending(
    db: Session,
    *,
    user: User,
    conversation_id: int | None,
    action_type: str,
    payload: dict,
    idempotency_key: str,
    thread_id: str | None = None,
    ip: str | None = None,
) -> tuple[PendingAction, bool]:
    """Create a pending action or reuse an active/executed duplicate.

    Returns (row, created_new).
    """
    existing = (
        db.query(PendingAction)
        .filter(PendingAction.idempotency_key == idempotency_key)
        .order_by(PendingAction.id.desc())
        .first()
    )
    now = utcnow()
    if existing is not None:
        existing = expire_if_needed(db, existing, ip=ip)
        if existing.status == "awaiting_confirmation":
            assert_owner(existing, user)
            # Refresh payload/conversation linkage for the active draft.
            existing.payload_json = payload
            existing.conversation_id = conversation_id or existing.conversation_id
            existing.thread_id = thread_id or existing.thread_id
            existing.expires_at = _expiry(now)
            existing.updated_at = now
            db.flush()
            _link_conversation(db, conversation_id, existing.id)
            return existing, False
        if existing.status == "executed" and existing.result_json:
            # Duplicate prevention: surface the prior successful result.
            return existing, False
        # Re-arm cancelled/expired/failed/blocked rows under the same idempotency key.
        if existing.status in {"cancelled", "expired", "failed", "blocked"}:
            assert_owner(existing, user)
            existing.status = "awaiting_confirmation"
            existing.payload_json = payload
            existing.result_json = None
            existing.conversation_id = conversation_id or existing.conversation_id
            existing.thread_id = thread_id or existing.thread_id
            existing.expires_at = _expiry(now)
            existing.confirmed_at = None
            existing.executed_at = None
            existing.updated_at = now
            db.flush()
            _link_conversation(db, conversation_id, existing.id)
            return existing, False

    row = PendingAction(
        owner_user_id=user.id,
        conversation_id=conversation_id,
        action_type=action_type,
        status="awaiting_confirmation",
        idempotency_key=idempotency_key,
        payload_json=payload,
        thread_id=thread_id,
        expires_at=_expiry(now),
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    db.flush()
    _link_conversation(db, conversation_id, row.id)
    write_audit(
        db,
        actor_user_id=user.id,
        action="pending_action_prepared",
        resource_type="pending_action",
        resource_id=str(row.id),
        outcome="success",
        details={"action_type": action_type, "idempotency_key": idempotency_key},
        ip_address=ip,
        commit=False,
    )
    db.flush()
    return row, True


def _link_conversation(db: Session, conversation_id: int | None, pending_id: int) -> None:
    if conversation_id is None:
        return
    conversation = db.get(Conversation, conversation_id)
    if conversation is not None:
        conversation.pending_action_id = pending_id
        conversation.status = "awaiting_confirmation"
        conversation.updated_at = utcnow()


def clear_conversation_pending(db: Session, conversation_id: int | None) -> None:
    if conversation_id is None:
        return
    conversation = db.get(Conversation, conversation_id)
    if conversation is not None:
        conversation.pending_action_id = None
        if conversation.status == "awaiting_confirmation":
            conversation.status = "active"
        conversation.updated_at = utcnow()


def cancel_pending(
    db: Session,
    row: PendingAction,
    *,
    user: User,
    ip: str | None = None,
    reason: str = "user_cancelled",
) -> dict:
    assert_owner(row, user)
    row = expire_if_needed(db, row, ip=ip)
    if row.status == "expired":
        clear_conversation_pending(db, row.conversation_id)
        return {
            "status": "expired",
            "message": "That pending action expired before confirmation. No records were changed.",
            "summary": "Pending action expired.",
            "pending_action_id": row.id,
        }
    if row.status != "awaiting_confirmation":
        return {
            "status": row.status,
            "message": f"That action is already {row.status}. No additional change was made.",
            "summary": f"Pending action already {row.status}.",
            "pending_action_id": row.id,
            "result": row.result_json,
        }
    row.status = "cancelled"
    row.updated_at = utcnow()
    clear_conversation_pending(db, row.conversation_id)
    write_audit(
        db,
        actor_user_id=user.id,
        action="pending_action_cancelled",
        resource_type="pending_action",
        resource_id=str(row.id),
        outcome="cancelled",
        details={"reason": reason, "action_type": row.action_type},
        ip_address=ip,
        commit=False,
    )
    db.flush()
    return {
        "status": "cancelled",
        "message": "I did not submit anything. Official records are unchanged.",
        "summary": "Confirmation was declined.",
        "pending_action_id": row.id,
    }


def revalidate_pending(db: Session, user: User, row: PendingAction) -> None:
    """Re-check authorization and current DB state before execution."""
    assert_owner(row, user)
    row = expire_if_needed(db, row)
    if row.status == "expired":
        raise ConfirmationError(
            "That pending action expired before confirmation. Prepare the action again.",
            status="expired",
        )
    if row.status == "executed":
        raise ConfirmationError(
            "That action was already executed. Helios will not repeat it.",
            status="duplicate",
            details={"result": row.result_json},
        )
    if row.status != "awaiting_confirmation":
        raise ConfirmationError(
            f"That pending action is {row.status} and cannot be confirmed.",
            status=row.status,
        )

    payload = row.payload_json or {}
    if row.action_type == "submit_leave":
        option = _selected_option(payload)
        start = date.fromisoformat(str(option["start"])[:10])
        end = date.fromisoformat(str(option["end"])[:10])
        leave_type = option.get("leave_type") or "annual"
        employee = employee_for_user(db, user)
        assessment = assess_window(db, employee, start, end, leave_type=leave_type)
        if assessment.get("hard_blocks"):
            raise ConfirmationError(
                " ".join(assessment["hard_blocks"]),
                status="blocked",
                details={"hard_blocks": assessment["hard_blocks"]},
            )
        # Duplicate open request for the same window.
        clash = (
            db.query(LeaveRequest)
            .filter(
                LeaveRequest.employee_id == employee.id,
                LeaveRequest.start_date == start,
                LeaveRequest.end_date == end,
                LeaveRequest.leave_type == leave_type,
                LeaveRequest.status.in_(("pending_manager", "pending_hr", "approved")),
            )
            .first()
        )
        if clash is not None:
            raise ConfirmationError(
                f"Leave request {clash.id} already covers {start.isoformat()} to {end.isoformat()} "
                f"(status {clash.status}). Helios will not create a duplicate.",
                status="duplicate",
                details={"leave_request_id": clash.id, "status": clash.status},
            )
        return

    if row.action_type == "decide_approval":
        approval_id = int(payload.get("approval_id"))
        decision = payload.get("decision")
        approval = db.get(Approval, approval_id)
        if approval is None:
            raise ConfirmationError("The approval record no longer exists.", status="not_found")
        if approval.status != "pending":
            raise ConfirmationError(
                f"Approval {approval_id} is already {approval.status}. It cannot be decided again.",
                status="duplicate",
                details={"approval_id": approval_id, "status": approval.status},
            )
        # Soft auth probe via decide path helpers without writing.
        from app.services.actions import _can_decide  # noqa: PLC0415

        if not _can_decide(user, approval):
            raise ConfirmationError(
                "You are not the assigned approver for this request.",
                status="denied",
            )
        if decision == "reject" and not str(payload.get("reason") or "").strip():
            raise ConfirmationError("A rejection reason is required.", status="needs_reason")
        return

    if row.action_type == "file_routing_correction":
        if user.role not in {"employee", "manager", "hr_admin"}:
            raise ConfirmationError("This account cannot file routing corrections.", status="denied")
        correction = payload.get("correction") or {}
        subject_id = correction.get("subject_employee_id") or user.employee_id
        if user.role != "hr_admin" and int(subject_id or 0) != int(user.employee_id or 0):
            raise ConfirmationError(
                "You can only confirm a routing correction for your own employee record.",
                status="denied",
            )
        subject = db.get(Employee, int(subject_id)) if subject_id else None
        if subject is None:
            raise ConfirmationError("The subject employee record no longer exists.", status="not_found")
        from app.services.leave_engine import investigate_routing  # noqa: PLC0415

        finding = investigate_routing(db, subject)
        live = finding.get("correction")
        if finding.get("consistent") or live is None:
            raise ConfirmationError(
                f"Approval routing for {subject.full_name} is already consistent. Nothing to file.",
                status="duplicate",
                details={"subject_employee_id": subject.id},
            )
        pending_to = correction.get("to_employee_id")
        if pending_to is not None and int(pending_to) != int(live.get("to_employee_id") or 0):
            raise ConfirmationError(
                "The recommended approval manager changed since this confirmation was prepared.",
                status="blocked",
                details={
                    "subject_employee_id": subject.id,
                    "pending_to": pending_to,
                    "live_to": live.get("to_employee_id"),
                },
            )
        return

    raise ConfirmationError(f"Unsupported pending action type '{row.action_type}'.", status="error")


def _selected_option(payload: dict, option_id: str | None = None) -> dict:
    options = payload.get("options") or {}
    chosen = option_id or payload.get("default_option") or "primary"
    option = options.get(chosen) or options.get("primary")
    if not option:
        raise ConfirmationError("The pending leave option is missing.", status="error")
    return option


def execute_pending(
    db: Session,
    user: User,
    row: PendingAction,
    *,
    option_id: str | None = None,
    ip: str | None = None,
) -> dict:
    try:
        revalidate_pending(db, user, row)
    except ConfirmationError as exc:
        row.status = "failed" if exc.status not in {"expired", "duplicate", "denied", "blocked", "needs_reason"} else (
            "expired" if exc.status == "expired" else row.status
        )
        if exc.status in {"blocked", "denied", "needs_reason", "not_found"}:
            row.status = "blocked"
        if exc.status == "duplicate" and row.result_json:
            clear_conversation_pending(db, row.conversation_id)
            return {
                "status": "duplicate",
                "message": exc.message,
                "summary": "Duplicate action prevented.",
                "pending_action_id": row.id,
                "result": row.result_json,
            }
        row.updated_at = utcnow()
        row.result_json = {"status": exc.status, "message": exc.message, "details": exc.details}
        clear_conversation_pending(db, row.conversation_id)
        write_audit(
            db,
            actor_user_id=user.id,
            action="pending_action_blocked",
            resource_type="pending_action",
            resource_id=str(row.id),
            outcome=exc.status,
            details={"message": exc.message, **exc.details},
            ip_address=ip,
            commit=False,
        )
        db.flush()
        return {
            "status": exc.status,
            "message": exc.message,
            "summary": f"Action could not proceed ({exc.status}).",
            "pending_action_id": row.id,
            "details": exc.details,
        }

    payload = dict(row.payload_json or {})
    now = utcnow()
    row.confirmed_at = now
    try:
        if row.action_type == "submit_leave":
            option = _selected_option(payload, option_id)
            leave_type = option.get("leave_type") or "annual"
            result = submit_leave(
                db,
                user,
                start=date.fromisoformat(str(option["start"])[:10]),
                end=date.fromisoformat(str(option["end"])[:10]),
                leave_type=leave_type,
                reason=(
                    "Planned sick leave prepared with Helios"
                    if leave_type == "sick"
                    else "Planned annual leave prepared with Helios"
                ),
                ip_address=ip,
            )
        elif row.action_type == "decide_approval":
            result = decide_approval(
                db,
                user,
                int(payload["approval_id"]),
                str(payload["decision"]),
                str(payload.get("reason") or payload.get("note") or ""),
                ip_address=ip,
            )
        elif row.action_type == "file_routing_correction":
            correction = payload.get("correction") or {}
            result = file_routing_correction(
                db,
                user,
                correction=correction,
                subject_employee_id=correction.get("subject_employee_id"),
                ip_address=ip,
            )
        else:
            raise ConfirmationError(f"Unsupported pending action type '{row.action_type}'.", status="error")
    except AuthorizationError as exc:
        row.status = "blocked"
        row.updated_at = utcnow()
        row.result_json = {"status": "denied", "message": str(exc)}
        clear_conversation_pending(db, row.conversation_id)
        write_audit(
            db,
            actor_user_id=user.id,
            action="pending_action_denied",
            resource_type="pending_action",
            resource_id=str(row.id),
            outcome="denied",
            details={"message": str(exc)},
            ip_address=ip,
            commit=False,
        )
        db.flush()
        return {
            "status": "denied",
            "message": str(exc),
            "summary": "Authorization failed on revalidation.",
            "pending_action_id": row.id,
        }
    except Exception as exc:  # noqa: BLE001
        row.status = "failed"
        row.updated_at = utcnow()
        row.result_json = {"status": "error", "message": f"{exc.__class__.__name__}: {exc}"}
        clear_conversation_pending(db, row.conversation_id)
        write_audit(
            db,
            actor_user_id=user.id,
            action="pending_action_failed",
            resource_type="pending_action",
            resource_id=str(row.id),
            outcome="error",
            details={"error": exc.__class__.__name__, "message": str(exc)[:300]},
            ip_address=ip,
            commit=False,
        )
        db.flush()
        return {
            "status": "error",
            "message": f"The operation failed and was not reported as successful ({exc.__class__.__name__}).",
            "summary": "Execution failed.",
            "pending_action_id": row.id,
        }

    success_statuses = {"submitted", "approved", "rejected", "pending_hr", "noop"}
    status = str(result.get("status") or "error")
    if status not in success_statuses and status != "blocked":
        # Do not mark executed for unexpected outcomes.
        row.status = "failed" if status == "error" else "blocked"
        row.result_json = result
        row.updated_at = utcnow()
        clear_conversation_pending(db, row.conversation_id)
        write_audit(
            db,
            actor_user_id=user.id,
            action="pending_action_not_executed",
            resource_type="pending_action",
            resource_id=str(row.id),
            outcome=status,
            details={"result": result},
            ip_address=ip,
            commit=False,
        )
        db.flush()
        return {
            **result,
            "pending_action_id": row.id,
            "summary": result.get("summary") or result.get("message") or status,
        }

    row.status = "executed"
    row.executed_at = utcnow()
    row.updated_at = utcnow()
    row.result_json = result
    clear_conversation_pending(db, row.conversation_id)
    write_audit(
        db,
        actor_user_id=user.id,
        action="pending_action_executed",
        resource_type="pending_action",
        resource_id=str(row.id),
        outcome="success",
        details={"action_type": row.action_type, "result_status": status},
        ip_address=ip,
        commit=False,
    )
    db.flush()
    return {
        **result,
        "pending_action_id": row.id,
        "summary": result.get("summary") or result.get("message") or f"Executed {row.action_type}.",
    }


def prepare_submit_leave_pending(
    db: Session,
    user: User,
    *,
    conversation_id: int | None,
    options: dict,
    default_option: str = "primary",
    confirmation_prompt: str | None = None,
    thread_id: str | None = None,
    ip: str | None = None,
) -> dict:
    primary = options.get(default_option) or options.get("primary") or {}
    key = idempotency_for_submit(
        employee_id=user.employee_id,
        start=str(primary.get("start")),
        end=str(primary.get("end")),
        leave_type=str(primary.get("leave_type") or "annual"),
    )
    payload = {
        "type": "submit_leave",
        "confirmation_prompt": confirmation_prompt
        or "Confirm which leave request I should file. I will not approve it.",
        "options": options,
        "default_option": default_option,
        "workflow_state": "AWAITING_CONFIRMATION",
    }
    row, _created = create_or_reuse_pending(
        db,
        user=user,
        conversation_id=conversation_id,
        action_type="submit_leave",
        payload=payload,
        idempotency_key=key,
        thread_id=thread_id,
        ip=ip,
    )
    if row.status == "executed" and row.result_json:
        return {
            **public_pending(row),
            "already_executed": True,
            "prior_result": row.result_json,
        }
    return public_pending(row)


def prepare_decision_pending(
    db: Session,
    user: User,
    *,
    conversation_id: int | None,
    approval_id: int,
    request_id: int | None,
    decision: str,
    reason: str = "",
    thread_id: str | None = None,
    ip: str | None = None,
) -> dict:
    key = idempotency_for_decision(approval_id=approval_id, decision=decision)
    label = f"{decision.title()} leave request {request_id or approval_id}"
    payload = {
        "type": "decide_approval",
        "confirmation_prompt": (
            f"Confirm that I should {decision} leave request {request_id or approval_id}. "
            "I will re-check authorization and the current approval status before writing."
        ),
        "options": {
            "primary": {
                "id": "primary",
                "label": label,
                "approval_id": approval_id,
                "request_id": request_id,
                "decision": decision,
                "reason": reason,
            }
        },
        "default_option": "primary",
        "approval_id": approval_id,
        "request_id": request_id,
        "decision": decision,
        "reason": reason,
        "workflow_state": "AWAITING_CONFIRMATION",
    }
    row, _created = create_or_reuse_pending(
        db,
        user=user,
        conversation_id=conversation_id,
        action_type="decide_approval",
        payload=payload,
        idempotency_key=key,
        thread_id=thread_id,
        ip=ip,
    )
    if row.status == "executed" and row.result_json:
        return {
            **public_pending(row),
            "already_executed": True,
            "prior_result": row.result_json,
        }
    return public_pending(row)


def prepare_routing_pending(
    db: Session,
    user: User,
    *,
    conversation_id: int | None,
    correction: dict,
    thread_id: str | None = None,
    ip: str | None = None,
) -> dict:
    subject_id = int(correction.get("subject_employee_id") or user.employee_id or 0)
    key = idempotency_for_routing(
        employee_id=subject_id,
        to_manager_id=int(correction.get("to_employee_id") or 0),
    )
    to_name = correction.get("to_name") or "the current manager"
    from_name = correction.get("from_name") or "the previous manager"
    payload = {
        "type": "file_routing_correction",
        "confirmation_prompt": (
            f"File this routing correction for People Operations on employee #{subject_id}? "
            f"It would move approval_manager_id from {from_name} to {to_name}. "
            "The official record will not change until HR approves it."
        ),
        "options": {"primary": {"id": "primary", "label": "File the HR correction"}},
        "default_option": "primary",
        "correction": correction,
        "subject_employee_id": subject_id,
        "owner_user_id": user.id,
        "workflow_state": "AWAITING_CONFIRMATION",
    }
    row, _created = create_or_reuse_pending(
        db,
        user=user,
        conversation_id=conversation_id,
        action_type="file_routing_correction",
        payload=payload,
        idempotency_key=key,
        thread_id=thread_id,
        ip=ip,
    )
    return public_pending(row)
