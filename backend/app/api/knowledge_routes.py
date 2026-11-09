from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import client_ip, get_current_user, get_db
from app.audit import write_audit
from app.config import get_settings
from app.models import IndexJob, PolicyDocument, User
from app.services.indexing import (
    content_hash,
    delete_document,
    extract_text,
    find_duplicate,
    index_document,
    mark_superseded,
    parse_optional_date,
    reindex_all,
    slugify,
    sync_policy_lifecycle,
    verify_index,
)

router = APIRouter()


def _require_hr(user: User) -> None:
    if user.role != "hr_admin":
        raise HTTPException(status_code=403, detail="Only People Operations can change the policy index")


def _document_payload(row: PolicyDocument) -> dict:
    return {
        "id": row.id,
        "slug": row.slug,
        "title": row.title,
        "version": row.version,
        "status": row.status,
        "doc_type": row.doc_type,
        "policy_family": row.policy_family,
        "effective_date": row.effective_date.isoformat(),
        "expiry_date": row.expiry_date.isoformat() if row.expiry_date else None,
        "department": row.department,
        "location": row.location,
        "access": row.access,
        "conflict_group": row.conflict_group,
        "supersedes": row.supersedes_slug,
        "source_format": row.source_format,
        "content_hash": row.content_hash,
        "index_status": row.index_status,
        "index_error": row.index_error,
        "embedding_model": row.embedding_model,
        "chunk_count": row.chunk_count,
        "indexed_at": row.indexed_at.isoformat() if row.indexed_at else None,
    }


@router.get("/policies/index")
def policy_index(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    _require_hr(user)
    settings = get_settings()
    documents = db.query(PolicyDocument).order_by(PolicyDocument.title, PolicyDocument.version).all()
    jobs = db.query(IndexJob).order_by(IndexJob.id.desc()).limit(30).all()
    health = verify_index(db)
    return {
        "embedding": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
            "batch_size": settings.embedding_batch_size,
            "rerank_enabled": settings.rerank_enabled,
            "rerank_model": settings.rerank_model if settings.rerank_enabled else None,
        },
        "health": health,
        "documents": [_document_payload(row) for row in documents],
        "jobs": [
            {
                "id": job.id,
                "document_id": job.document_id,
                "action": job.action,
                "status": job.status,
                "detail": job.detail,
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            }
            for job in jobs
        ],
    }


@router.post("/policies/reindex")
def reindex_policies(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    _require_hr(user)
    lifecycle = sync_policy_lifecycle(db)
    results = reindex_all(db, force=True)
    health = verify_index(db)
    write_audit(
        db,
        actor_user_id=user.id,
        action="policy_reindex",
        resource_type="policy_index",
        resource_id="all",
        outcome="success" if health.get("healthy") else "partial",
        details={"results": results, "lifecycle": lifecycle, "healthy": health.get("healthy")},
        ip_address=client_ip(request),
    )
    return {"results": results, "lifecycle": lifecycle, "health": health}


@router.post("/policies/upload")
async def upload_policy(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(...),
    version: str = Form(...),
    policy_family: str = Form(...),
    effective_date: date = Form(...),
    doc_type: str = Form("policy"),
    expiry_date: str = Form(""),
    supersedes_slug: str = Form(""),
    department: str = Form(""),
    location: str = Form("global"),
    access: str = Form("internal"),
    conflict_group: str = Form(""),
    status: str = Form("current"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    _require_hr(user)
    if access not in {"public", "internal", "hr_confidential"}:
        raise HTTPException(status_code=422, detail="Access must be public, internal, or hr_confidential")
    if status not in {"current", "expired", "superseded", "draft"}:
        raise HTTPException(status_code=422, detail="Status must be current, expired, superseded, or draft")
    payload = await file.read()
    if len(payload) > 10_000_000:
        raise HTTPException(status_code=413, detail="Policy files must be 10 MB or smaller")
    try:
        text, source_format = extract_text(file.filename or "policy.txt", payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"The file could not be read: {exc}") from exc
    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from that file")
    digest = content_hash(text)
    duplicate = find_duplicate(db, digest)
    if duplicate is not None:
        raise HTTPException(status_code=409, detail=f"This file matches the indexed document {duplicate.slug}")
    slug = slugify(title, version)
    suffix = 2
    while db.query(PolicyDocument).filter(PolicyDocument.slug == slug).one_or_none() is not None:
        slug = slugify(f"{title}-{suffix}", version)
        suffix += 1
    try:
        expiry = parse_optional_date(expiry_date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="expiry_date must be YYYY-MM-DD") from exc
    document = PolicyDocument(
        slug=slug,
        title=title.strip(),
        doc_type=doc_type.strip() or "policy",
        version=version.strip(),
        policy_family=policy_family.strip(),
        effective_date=effective_date,
        expiry_date=expiry,
        supersedes_slug=supersedes_slug.strip() or None,
        department=department.strip() or None,
        location=location.strip() or "global",
        access=access,
        conflict_group=conflict_group.strip() or None,
        status=status,
        body=text,
        source_path=file.filename or slug,
        source_format=source_format,
        content_hash="",
        index_status="pending",
    )
    db.add(document)
    db.flush()
    if status == "current":
        mark_superseded(db, document.supersedes_slug)
    outcome = index_document(db, document, force=True)
    db.commit()
    write_audit(
        db,
        actor_user_id=user.id,
        action="policy_upload",
        resource_type="policy_document",
        resource_id=document.slug,
        outcome="success" if outcome == "indexed" else "failed",
        details={"index_status": document.index_status, "error": document.index_error},
        ip_address=client_ip(request),
    )
    if outcome != "indexed":
        raise HTTPException(status_code=502, detail=document.index_error or "The document was stored but embedding failed")
    return _document_payload(document)


@router.post("/policies/{slug}/reindex")
def reindex_one(slug: str, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    _require_hr(user)
    document = db.query(PolicyDocument).filter(PolicyDocument.slug == slug).one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    outcome = index_document(db, document, force=True)
    db.commit()
    write_audit(
        db,
        actor_user_id=user.id,
        action="policy_reindex",
        resource_type="policy_document",
        resource_id=slug,
        outcome="success" if outcome == "indexed" else "failed",
        details={"outcome": outcome, "error": document.index_error},
        ip_address=client_ip(request),
    )
    return _document_payload(document)


@router.delete("/policies/{slug}")
def remove_policy(slug: str, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    _require_hr(user)
    document = db.query(PolicyDocument).filter(PolicyDocument.slug == slug).one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    delete_document(db, document)
    write_audit(
        db,
        actor_user_id=user.id,
        action="policy_delete",
        resource_type="policy_document",
        resource_id=slug,
        outcome="success",
        details={},
        ip_address=client_ip(request),
    )
    return {"deleted": slug}
