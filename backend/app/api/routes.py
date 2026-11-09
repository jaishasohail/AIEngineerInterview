from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.access import (
    AuthorizationError,
    can_view_employee_record,
    can_view_leave,
    employee_for_user,
    redact_employee,
)
from app.ai.eval import evaluate_retrieval
from app.ai.runtime import add_message, confirm_turn, json_safe, run_turn, stream_turn
from app.api.deps import client_ip, get_current_user, get_db
from app.audit import write_audit
from app.clock import org_today, utcnow
from app.config import get_settings
from app.models import Approval, AuditLog, Conversation, Department, Employee, LeaveBalance, LeaveRequest, Message, PolicyDocument, Project, ProjectAssignment, User
from app.security import create_token, verify_password
from app.services.actions import decide_approval, submit_leave
from app.services.analytics_engine import absence_analytics
from app.services.dashboard import build_dashboard
from app.services.leave_engine import balance_payload, get_balance
from app.services.notifications import list_notifications

router = APIRouter()

from app.api.holiday_routes import router as holiday_router  # noqa: E402
from app.api.knowledge_routes import router as knowledge_router  # noqa: E402

router.include_router(knowledge_router)
router.include_router(holiday_router)


class LoginIn(BaseModel):
    email: str
    password: str


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = None


class ConfirmIn(BaseModel):
    conversation_id: int
    confirmed: bool
    option_id: str = "primary"


class LeaveIn(BaseModel):
    start_date: date
    end_date: date
    leave_type: str = "annual"
    reason: str = "Planned annual leave"


class DecisionIn(BaseModel):
    decision: str
    note: str = ""


def _user_payload(db: Session, user: User) -> dict:
    employee = employee_for_user(db, user)
    department = db.get(Department, employee.department_id)
    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "employee_id": employee.id,
        "full_name": employee.full_name,
        "job_title": employee.job_title,
        "department": department.name if department else None,
        "location": employee.location,
    }


@router.post("/auth/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)) -> dict:
    user = db.query(User).filter(User.email == body.email.strip().lower()).one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        write_audit(
            db,
            actor_user_id=None if user is None else user.id,
            action="login",
            resource_type="user",
            resource_id=body.email.strip().lower(),
            outcome="denied",
            details={},
            ip_address=client_ip(request),
        )
        raise HTTPException(status_code=401, detail="Email or password is incorrect")
    write_audit(db, actor_user_id=user.id, action="login", resource_type="user", resource_id=str(user.id), outcome="success", ip_address=client_ip(request))
    token = create_token(user_id=user.id, role=user.role, employee_id=user.employee_id)
    return {"access_token": token, "token_type": "bearer", "user": _user_payload(db, user)}


@router.get("/auth/me")
def me(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    return _user_payload(db, user)


@router.get("/dashboard")
def dashboard(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    return build_dashboard(db, user)


@router.get("/directory")
def directory(q: str = "", user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    query = db.query(Employee, Department).join(Department, Department.id == Employee.department_id).filter(Employee.employment_status == "active")
    if q.strip():
        like = f"%{q.strip()}%"
        query = query.filter((Employee.first_name.ilike(like)) | (Employee.last_name.ilike(like)) | (Employee.email.ilike(like)) | (Employee.job_title.ilike(like)))
    rows = query.order_by(Employee.last_name).limit(200).all()
    return {
        "results": [
            {
                "id": employee.id,
                "full_name": employee.full_name,
                "email": employee.email,
                "job_title": employee.job_title,
                "department": department.name,
                "location": employee.location,
                "employee_number": employee.employee_number,
            }
            for employee, department in rows
        ]
    }


@router.get("/employees/me")
def my_employee(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    employee = employee_for_user(db, user)
    department = db.get(Department, employee.department_id)
    profile = redact_employee(db, user, employee, employee, department.name if department else "", include_private=True)
    profile["balances"] = [balance_payload(row) for row in db.query(LeaveBalance).filter(LeaveBalance.employee_id == employee.id).all()]
    return profile


@router.get("/employees/{employee_id}")
def get_employee(employee_id: int, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    actor_employee = employee_for_user(db, user)
    subject = db.get(Employee, employee_id)
    if subject is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    department = db.get(Department, subject.department_id)
    private = can_view_employee_record(db, user, actor_employee, subject)
    if not private:
        write_audit(
            db,
            actor_user_id=user.id,
            action="employee_read",
            resource_type="employee",
            resource_id=str(subject.id),
            outcome="denied",
            details={"reason": "private profile"},
            ip_address=client_ip(request),
        )
        raise HTTPException(status_code=403, detail="You cannot open that employee record")
    if user.employee_id != subject.id:
        write_audit(
            db,
            actor_user_id=user.id,
            action="employee_read",
            resource_type="employee",
            resource_id=str(subject.id),
            outcome="success",
            details={},
            ip_address=client_ip(request),
        )
    profile = redact_employee(db, user, actor_employee, subject, department.name if department else "", include_private=True)
    if can_view_leave(db, user, actor_employee, subject):
        profile["balances"] = [balance_payload(row) for row in db.query(LeaveBalance).filter(LeaveBalance.employee_id == subject.id).all()]
    return profile


@router.get("/leave/balances")
def leave_balances(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    employee = employee_for_user(db, user)
    rows = db.query(LeaveBalance).filter(LeaveBalance.employee_id == employee.id).all()
    return {"balances": [balance_payload(row) for row in rows]}


def _leave_payload(db: Session, request_row: LeaveRequest) -> dict:
    employee = db.get(Employee, request_row.employee_id)
    manager = db.get(Employee, request_row.routed_manager_id) if request_row.routed_manager_id else None
    return {
        "id": request_row.id,
        "employee_id": request_row.employee_id,
        "employee_name": None if employee is None else employee.full_name,
        "leave_type": request_row.leave_type,
        "start_date": request_row.start_date.isoformat(),
        "end_date": request_row.end_date.isoformat(),
        "business_days": request_row.business_days,
        "status": request_row.status,
        "reason": request_row.reason,
        "routed_to": None if manager is None else manager.full_name,
        "hr_required": request_row.hr_required,
        "impact": request_row.impact_json,
        "system": request_row.system_label,
    }


@router.get("/leave/requests")
def leave_requests(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    actor_employee = employee_for_user(db, user)
    query = db.query(LeaveRequest)
    if user.role == "employee":
        query = query.filter(LeaveRequest.employee_id == actor_employee.id)
    elif user.role == "manager":
        report_ids = [row[0] for row in db.query(Employee.id).filter(Employee.manager_id == actor_employee.id).all()]
        query = query.filter(LeaveRequest.employee_id.in_(report_ids + [actor_employee.id]))
    rows = query.order_by(LeaveRequest.start_date.desc()).limit(100).all()
    return {"requests": [_leave_payload(db, row) for row in rows]}


@router.post("/leave/requests")
def create_leave(body: LeaveIn, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    if body.leave_type != "annual":
        raise HTTPException(status_code=422, detail="This proof of concept submits annual leave. Other types stay on the balance record.")
    result = submit_leave(
        db,
        user,
        start=body.start_date,
        end=body.end_date,
        leave_type=body.leave_type,
        reason=body.reason,
        ip_address=client_ip(request),
    )
    if result["status"] == "blocked":
        raise HTTPException(status_code=422, detail=result["message"])
    return result


@router.get("/team")
def team(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    actor = employee_for_user(db, user)
    if user.role == "hr_admin":
        members = db.query(Employee).filter(Employee.employment_status == "active").order_by(Employee.last_name).limit(150).all()
    elif user.role == "manager":
        members = db.query(Employee).filter(Employee.manager_id == actor.id).order_by(Employee.last_name).all()
    else:
        members = db.query(Employee).filter(Employee.manager_id == actor.manager_id, Employee.id != actor.id).order_by(Employee.last_name).all() if actor.manager_id else []
    today = org_today()
    payload = []
    for member in members:
        department = db.get(Department, member.department_id)
        manager = db.get(Employee, member.manager_id) if member.manager_id else None
        leaves = (
            db.query(LeaveRequest)
            .filter(LeaveRequest.employee_id == member.id, LeaveRequest.status.in_(("pending_manager", "pending_hr", "approved")), LeaveRequest.end_date >= today)
            .all()
        )
        show_reason = user.role in {"hr_admin", "manager"}
        projects = (
            db.query(Project, ProjectAssignment)
            .join(ProjectAssignment, ProjectAssignment.project_id == Project.id)
            .filter(ProjectAssignment.employee_id == member.id)
            .all()
        )
        payload.append(
            {
                "id": member.id,
                "full_name": member.full_name,
                "job_title": member.job_title,
                "department": department.name if department else None,
                "location": member.location,
                "manager": None if manager is None else manager.full_name,
                "leave": [
                    {
                        "start_date": row.start_date.isoformat(),
                        "end_date": row.end_date.isoformat(),
                        "status": row.status,
                        "business_days": row.business_days,
                        "reason": row.reason if show_reason else None,
                    }
                    for row in leaves
                ],
                "projects": [
                    {"name": project.name, "deadline": project.deadline.isoformat(), "criticality": project.criticality, "allocation_pct": assignment.allocation_pct}
                    for project, assignment in projects
                ],
            }
        )
    return {"scope": user.role, "members": payload}


@router.get("/analytics")
def analytics(months: int = 3, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    return absence_analytics(db, user, months=months)


@router.get("/policies")
def policies(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    rows = db.query(PolicyDocument).order_by(PolicyDocument.title, PolicyDocument.version).all()
    allowed = {"public", "internal", "hr_confidential"} if user.role == "hr_admin" else {"public", "internal"}
    visible = []
    for row in rows:
        if row.access not in allowed:
            continue
        visible.append(
            {
                "slug": row.slug,
                "title": row.title,
                "version": row.version,
                "doc_type": row.doc_type,
                "status": row.status,
                "effective_date": row.effective_date.isoformat(),
                "expiry_date": row.expiry_date.isoformat() if row.expiry_date else None,
                "department": row.department,
                "location": row.location,
                "access": row.access,
                "conflict_group": row.conflict_group,
                "supersedes": row.supersedes_slug,
                "index_status": row.index_status,
                "index_error": row.index_error,
                "embedding_model": row.embedding_model,
                "chunk_count": row.chunk_count,
            }
        )
    return {"policies": visible}


@router.get("/policies/{slug}")
def policy_detail(slug: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    row = db.query(PolicyDocument).filter(PolicyDocument.slug == slug).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    if row.access == "hr_confidential" and user.role != "hr_admin":
        raise HTTPException(status_code=403, detail="That policy is restricted to People Operations")
    return {
        "slug": row.slug,
        "title": row.title,
        "version": row.version,
        "status": row.status,
        "effective_date": row.effective_date.isoformat(),
        "expiry_date": row.expiry_date.isoformat() if row.expiry_date else None,
        "access": row.access,
        "conflict_group": row.conflict_group,
        "body": row.body,
    }


def _approval_visible(user: User, actor: Employee, approval: Approval) -> bool:
    if user.role == "hr_admin":
        return True
    if approval.approver_employee_id == actor.id:
        return True
    if approval.requested_by_user_id == user.id:
        return True
    return False


@router.get("/approvals")
def approvals(status: str = "pending", user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    actor = employee_for_user(db, user)
    query = db.query(Approval)
    if status != "all":
        query = query.filter(Approval.status == status)
    rows = []
    for approval in query.order_by(Approval.created_at.desc()).limit(200).all():
        if _approval_visible(user, actor, approval):
            subject = db.get(Employee, approval.subject_employee_id) if approval.subject_employee_id else None
            leave = db.get(LeaveRequest, approval.leave_request_id) if approval.leave_request_id else None
            rows.append(
                {
                    "id": approval.id,
                    "action_type": approval.action_type,
                    "stage": approval.stage,
                    "status": approval.status,
                    "approver_role": approval.approver_role,
                    "subject": None if subject is None else subject.full_name,
                    "subject_employee_id": approval.subject_employee_id,
                    "leave": None if leave is None else _leave_payload(db, leave),
                    "payload": approval.payload_json,
                    "impact": approval.impact_json or (None if leave is None else leave.impact_json),
                    "decision_note": approval.decision_note,
                    "created_at": approval.created_at.isoformat(),
                    "can_decide": approval.status == "pending" and (
                        user.role == "hr_admin" or (approval.approver_role == "manager" and approval.approver_employee_id == actor.id)
                    ),
                }
            )
    return {"approvals": rows}


@router.post("/approvals/{approval_id}/decide")
def decide(approval_id: int, body: DecisionIn, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    try:
        return decide_approval(db, user, approval_id, body.decision, body.note, ip_address=client_ip(request))
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=exc.message) from exc


@router.get("/audit")
def audit(limit: int = 100, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    limit = max(1, min(limit, 300))
    query = db.query(AuditLog)
    if user.role == "employee":
        query = query.filter(AuditLog.actor_user_id == user.id)
    elif user.role == "manager":
        query = query.filter(AuditLog.actor_user_id == user.id)
    rows = query.order_by(AuditLog.created_at.desc()).limit(limit).all()
    users = {item.id: item.email for item in db.query(User).filter(User.id.in_([row.actor_user_id for row in rows if row.actor_user_id])).all()} if rows else {}
    return {
        "logs": [
            {
                "id": row.id,
                "actor": users.get(row.actor_user_id),
                "action": row.action,
                "resource_type": row.resource_type,
                "resource_id": row.resource_id,
                "outcome": row.outcome,
                "details": row.details_json,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
    }


@router.get("/notifications")
def notifications(
    limit: int = 50,
    unread_only: bool = False,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """List in-app HR workflow notifications for the signed-in employee.

    Helios has no external email/SMS/WhatsApp provider. These rows are recorded
    in-app only; delivery_status 'recorded' does not mean an external message was sent.
    """
    actor = employee_for_user(db, user)
    rows = list_notifications(db, recipient_employee_id=actor.id, limit=limit, unread_only=unread_only)
    return {
        "notifications": rows,
        "channel": "in_app",
        "external_provider": None,
        "note": "Notifications are persisted in Helios only. No external email provider is configured.",
    }


@router.get("/settings")
def settings(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    config = get_settings()
    balance = get_balance(db, user.employee_id)
    # Never return secrets, connection strings, API keys, or JWT material.
    return {
        "profile": _user_payload(db, user),
        "org_clock": config.org_clock,
        "llm_enabled": config.llm_enabled,
        "model": config.openrouter_model if config.llm_enabled else None,
        "embedding_mode": f"{config.embedding_provider}:{config.embedding_model}",
        "embedding_dimensions": config.embedding_dimensions,
        "rerank_enabled": config.rerank_enabled,
        "database": "postgresql" if config.database_url.startswith("postgresql") else "sqlite",
        "annual_remaining": None if balance is None else balance_payload(balance)["remaining"],
        "demo_password_hint": "Use the password from the README. It is not returned by the API.",
        "secrets_exposed": False,
    }


@router.post("/retrieval/evaluate")
def evaluate(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    if user.role != "hr_admin":
        raise HTTPException(status_code=403, detail="Retrieval evaluation is limited to HR administrators")
    return evaluate_retrieval(db, role="hr_admin")


def _conversation_for(db: Session, user: User, conversation_id: int | None, message: str) -> Conversation:
    if conversation_id is not None:
        conversation = db.get(Conversation, conversation_id)
        if conversation is None or conversation.user_id != user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    now = utcnow()
    conversation = Conversation(user_id=user.id, title=message[:80] or "New conversation", status="active", created_at=now, updated_at=now)
    db.add(conversation)
    db.flush()
    return conversation


@router.get("/assistant/conversations")
def conversations(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    rows = db.query(Conversation).filter(Conversation.user_id == user.id).order_by(Conversation.updated_at.desc()).limit(40).all()
    return {
        "conversations": [
            {"id": row.id, "title": row.title, "status": row.status, "updated_at": row.updated_at.isoformat()}
            for row in rows
        ]
    }


@router.get("/assistant/conversations/{conversation_id}")
def conversation_detail(conversation_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = db.query(Message).filter(Message.conversation_id == conversation.id).order_by(Message.created_at.asc(), Message.id.asc()).all()
    return {
        "id": conversation.id,
        "title": conversation.title,
        "status": conversation.status,
        "messages": [
            {"id": row.id, "role": row.role, "content": row.content, "payload": row.payload_json, "created_at": row.created_at.isoformat()}
            for row in messages
        ],
    }


@router.post("/assistant/chat")
def chat(body: ChatIn, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    conversation = _conversation_for(db, user, body.conversation_id, body.message)
    add_message(db, conversation, "user", body.message.strip())
    payload = run_turn(db, user, conversation, body.message.strip(), ip=client_ip(request))
    add_message(db, conversation, "assistant", payload["message"], json_safe(payload))
    db.commit()
    return payload


@router.post("/assistant/chat/stream")
def chat_stream(body: ChatIn, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> StreamingResponse:
    conversation = _conversation_for(db, user, body.conversation_id, body.message)
    add_message(db, conversation, "user", body.message.strip())
    db.commit()

    def generate():
        import json

        final = None
        for event in stream_turn(db, user, conversation, body.message.strip(), ip=client_ip(request)):
            if event["type"] == "final":
                final = event["payload"]
            yield f"data: {json.dumps(event, default=str)}\n\n"
        if final:
            add_message(db, conversation, "assistant", final["message"], json_safe(final))
            db.commit()

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/assistant/confirm")
def confirm(body: ConfirmIn, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    conversation = db.get(Conversation, body.conversation_id)
    if conversation is None or conversation.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    phrase = "cancel" if not body.confirmed else ("yes, submit the alternative" if body.option_id == "alternative" else "yes, submit")
    add_message(db, conversation, "user", phrase)
    try:
        payload = confirm_turn(db, user, conversation, confirmed=body.confirmed, option_id=body.option_id, ip=client_ip(request))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    add_message(db, conversation, "assistant", payload["message"], json_safe(payload))
    db.commit()
    return payload
