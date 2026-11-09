"""Idempotent policy ingestion, embedding, and reindexing.

Structured HR entities (employees, balances, attendance, approvals) stay in
PostgreSQL tables. pgvector stores searchable chunks of textual HR knowledge only.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from datetime import date, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.ai.embeddings import Embedder, EmbeddingUnavailable, get_embedder, validate_embedder
from app.ai.retrieval import chunk_text, tokenize
from app.clock import org_today, utcnow
from app.config import get_settings
from app.models import IndexJob, PolicyChunk, PolicyDocument

logger = logging.getLogger("helios.index")
SLUG = re.compile(r"[^a-z0-9]+")
REQUIRED_META_KEYS = {
    "document_id",
    "chunk_index",
    "source",
    "document_type",
    "employee_id",
    "department",
    "access",
    "visibility",
    "policy_version",
    "effective_date",
    "expiration_date",
    "updated_at",
}


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(title: str, version: str) -> str:
    raw = SLUG.sub("-", f"{title}-{version}".lower()).strip("-")
    return raw[:80] or "policy"


def extract_text(filename: str, data: bytes) -> tuple[str, str]:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".txt", ".md"}:
        try:
            return data.decode("utf-8"), suffix.lstrip(".")
        except UnicodeError as exc:
            raise ValueError("Text files must be UTF-8.") from exc
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(part.strip() for part in pages if part.strip()), "pdf"
    if suffix == ".docx":
        from docx import Document

        document = Document(io.BytesIO(data))
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        return "\n\n".join(paragraphs), "docx"
    raise ValueError("Upload a PDF, DOCX, TXT, or Markdown file.")


def _visibility(access: str) -> str:
    if access == "hr_confidential":
        return "hr_admin"
    if access == "internal":
        return "authenticated"
    return "public"


def chunk_metadata(
    document: PolicyDocument,
    *,
    chunk_index: int,
    chunk_hash: str,
    updated_at: datetime,
    chunk_id: int | None = None,
) -> dict:
    """Authorization and provenance metadata stored with each vector chunk."""
    payload = {
        "document_id": document.id,
        "document_slug": document.slug,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "chunk_hash": chunk_hash,
        "source": document.source_path,
        "source_format": document.source_format,
        "document_type": document.doc_type,
        "policy_family": document.policy_family,
        "title": document.title,
        "employee_id": None,
        "department": document.department,
        "location": document.location,
        "access": document.access,
        "visibility": _visibility(document.access),
        "policy_version": document.version,
        "effective_date": document.effective_date.isoformat(),
        "expiration_date": document.expiry_date.isoformat() if document.expiry_date else None,
        "status": document.status,
        "conflict_group": document.conflict_group,
        "supersedes_slug": document.supersedes_slug,
        "updated_at": updated_at.isoformat(),
    }
    return payload


def _embed_batches(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    batch_size = max(1, int(getattr(settings, "embedding_batch_size", 32) or 32))
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(embedder.embed_documents(batch))
    if len(vectors) != len(texts):
        raise EmbeddingUnavailable("The embedding provider returned a different number of vectors than chunks.")
    for vector in vectors:
        if len(vector) != embedder.dimensions:
            raise EmbeddingUnavailable(
                f"The embedding provider returned dimension {len(vector)}. The active model expects {embedder.dimensions}."
            )
    return vectors


def _chunks_are_current(db: Session, document: PolicyDocument, embedder: Embedder, digest: str) -> bool:
    if (
        document.content_hash != digest
        or document.index_status != "indexed"
        or document.embedding_model != embedder.model_id
        or (document.chunk_count or 0) <= 0
    ):
        return False
    rows = (
        db.query(PolicyChunk)
        .filter(PolicyChunk.policy_id == document.id)
        .order_by(PolicyChunk.chunk_index.asc())
        .all()
    )
    if len(rows) != document.chunk_count:
        return False
    for row in rows:
        if row.embedding_model != embedder.model_id or row.embedding_dimensions != embedder.dimensions:
            return False
        if not row.embedding or len(row.embedding) != embedder.dimensions:
            return False
        meta = row.meta_json or {}
        if not REQUIRED_META_KEYS.issubset(meta.keys()):
            return False
        if meta.get("document_id") != document.id:
            return False
        if meta.get("access") != document.access:
            return False
    return True


def needs_reindex(db: Session, document: PolicyDocument, embedder: Embedder, *, force: bool = False) -> bool:
    if force:
        return True
    digest = content_hash(document.body or "")
    return not _chunks_are_current(db, document, embedder, digest)


def index_document(db: Session, document: PolicyDocument, *, embedder: Embedder | None = None, force: bool = False) -> str:
    """Regenerate chunk embeddings. Unchanged content for the active model is skipped."""
    try:
        embedder = embedder or get_embedder()
        validate_embedder(embedder, probe=False)
    except EmbeddingUnavailable as exc:
        document.index_status = "failed"
        document.index_error = str(exc)[:500]
        db.flush()
        _job(db, document.id, "reindex", "failed", str(exc))
        return "failed"

    digest = content_hash(document.body or "")
    if not force and _chunks_are_current(db, document, embedder, digest):
        _job(db, document.id, "reindex", "skipped", "Content and embedding model are unchanged.")
        return "skipped"

    if not (document.body or "").strip():
        document.index_status = "failed"
        document.index_error = "The document has no extractable text."
        db.flush()
        _job(db, document.id, "reindex", "failed", document.index_error)
        return "failed"

    chunks = chunk_text(document.body)
    texts = [f"{document.title}\n{chunk}" for chunk in chunks]
    try:
        vectors = _embed_batches(embedder, texts)
    except EmbeddingUnavailable as exc:
        document.index_status = "failed"
        document.index_error = str(exc)[:500]
        db.flush()
        _job(db, document.id, "reindex", "failed", str(exc))
        logger.warning("Embedding failed for %s: %s", document.slug, exc)
        return "failed"

    started = utcnow()
    with db.begin_nested():
        db.query(PolicyChunk).filter(PolicyChunk.policy_id == document.id).delete()
        rows: list[PolicyChunk] = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_hash = content_hash(chunk)
            row = PolicyChunk(
                policy_id=document.id,
                chunk_index=index,
                content=chunk,
                embedding=vector,
                embedding_model=embedder.model_id,
                embedding_dimensions=embedder.dimensions,
                token_count=len(tokenize(chunk)),
                meta_json=chunk_metadata(
                    document,
                    chunk_index=index,
                    chunk_hash=chunk_hash,
                    updated_at=started,
                ),
            )
            db.add(row)
            rows.append(row)
        db.flush()
        for row in rows:
            meta = dict(row.meta_json or {})
            meta["chunk_id"] = row.id
            meta["document_id"] = document.id
            row.meta_json = meta
        document.content_hash = digest
        document.index_status = "indexed"
        document.index_error = None
        document.embedding_model = embedder.model_id
        document.chunk_count = len(chunks)
        document.indexed_at = started
    db.flush()
    _job(db, document.id, "reindex", "succeeded", f"Indexed {len(chunks)} chunks with {embedder.model_id}.")
    return "indexed"


def reindex_all(db: Session, *, embedder: Embedder | None = None, force: bool = False) -> list[dict]:
    results = []
    try:
        embedder = embedder or get_embedder()
        validate_embedder(embedder)
    except EmbeddingUnavailable as exc:
        documents = db.query(PolicyDocument).order_by(PolicyDocument.id).all()
        for document in documents:
            document.index_status = "failed"
            document.index_error = str(exc)[:500]
            results.append({"slug": document.slug, "outcome": "failed", "error": document.index_error})
        db.commit()
        return results

    sync_policy_lifecycle(db)
    documents = db.query(PolicyDocument).order_by(PolicyDocument.id).all()
    for document in documents:
        outcome = index_document(db, document, embedder=embedder, force=force)
        results.append({"slug": document.slug, "outcome": outcome, "error": document.index_error})
    db.commit()
    return results


def refresh_stale_policies(db: Session) -> list[dict]:
    """Incrementally reindex documents whose content, model, dimensions, or metadata drifted."""
    try:
        embedder = get_embedder()
        validate_embedder(embedder)
    except EmbeddingUnavailable:
        raise
    sync_policy_lifecycle(db)
    stale: list[PolicyDocument] = []
    for document in db.query(PolicyDocument).order_by(PolicyDocument.id).all():
        if needs_reindex(db, document, embedder, force=False):
            stale.append(document)
    if not stale:
        return []
    results = []
    for document in stale:
        outcome = index_document(db, document, embedder=embedder, force=False)
        results.append({"slug": document.slug, "outcome": outcome, "error": document.index_error})
    db.commit()
    return results


def sync_policy_lifecycle(db: Session) -> dict:
    """Keep document status aligned with effective/expiry dates and supersession links.

    Expired and superseded policies remain stored for audit, but must not stay marked current.
    """
    today = org_today()
    expired = 0
    superseded = 0
    documents = db.query(PolicyDocument).all()
    by_slug = {document.slug: document for document in documents}

    for document in documents:
        if document.expiry_date and document.expiry_date < today and document.status == "current":
            document.status = "expired"
            expired += 1

    for document in documents:
        prior_slug = document.supersedes_slug
        if not prior_slug or document.status != "current":
            continue
        if document.effective_date > today:
            continue
        prior = by_slug.get(prior_slug)
        if prior is None:
            continue
        if prior.status == "current":
            prior.status = "superseded"
            superseded += 1

    db.flush()
    return {"expired": expired, "superseded": superseded, "as_of": today.isoformat()}


def delete_document(db: Session, document: PolicyDocument) -> None:
    db.query(PolicyChunk).filter(PolicyChunk.policy_id == document.id).delete()
    _job(db, document.id, "delete", "succeeded", f"Removed {document.slug} and its vectors.")
    db.delete(document)
    db.commit()


def find_duplicate(db: Session, digest: str) -> PolicyDocument | None:
    if not digest:
        return None
    return (
        db.query(PolicyDocument)
        .filter(PolicyDocument.content_hash == digest)
        .order_by(PolicyDocument.id.asc())
        .first()
    )


def mark_superseded(db: Session, slug: str | None) -> None:
    if not slug:
        return
    previous = db.query(PolicyDocument).filter(PolicyDocument.slug == slug).one_or_none()
    if previous is not None and previous.status == "current":
        previous.status = "superseded"


def verify_index(db: Session, *, embedder: Embedder | None = None) -> dict:
    """Read-only health report for the pgvector policy index."""
    settings = get_settings()
    embedder_error = None
    model_id = settings.embedding_model
    dimensions = settings.embedding_dimensions
    try:
        embedder = embedder or get_embedder()
        validate_embedder(embedder)
        model_id = embedder.model_id
        dimensions = embedder.dimensions
    except EmbeddingUnavailable as exc:
        embedder_error = str(exc)

    documents = db.query(PolicyDocument).order_by(PolicyDocument.id).all()
    chunks = db.query(PolicyChunk).all()
    by_policy: dict[int, list[PolicyChunk]] = {}
    for chunk in chunks:
        by_policy.setdefault(chunk.policy_id, []).append(chunk)

    today = org_today()
    issues: list[dict] = []
    indexed = 0
    failed = 0
    pending = 0
    current_docs = 0
    expired_or_superseded = 0

    for document in documents:
        if document.status == "current" and not (document.expiry_date and document.expiry_date < today):
            current_docs += 1
        else:
            expired_or_superseded += 1
        if document.index_status == "indexed":
            indexed += 1
        elif document.index_status == "failed":
            failed += 1
        else:
            pending += 1

        rows = by_policy.get(document.id, [])
        if document.index_status == "indexed" and len(rows) != (document.chunk_count or 0):
            issues.append({"slug": document.slug, "issue": "chunk_count_mismatch", "expected": document.chunk_count, "actual": len(rows)})
        for row in rows:
            if embedder_error is None and (row.embedding_model != model_id or row.embedding_dimensions != dimensions):
                issues.append({"slug": document.slug, "issue": "embedding_model_mismatch", "chunk_id": row.id})
            if not row.embedding or len(row.embedding) != (row.embedding_dimensions or 0):
                issues.append({"slug": document.slug, "issue": "invalid_embedding", "chunk_id": row.id})
            meta = row.meta_json or {}
            missing = sorted(REQUIRED_META_KEYS - set(meta.keys()))
            if missing:
                issues.append({"slug": document.slug, "issue": "incomplete_metadata", "chunk_id": row.id, "missing": missing})
            elif meta.get("access") != document.access:
                issues.append({"slug": document.slug, "issue": "access_metadata_drift", "chunk_id": row.id})

    # Dense retrieval probe against current, non-confidential corpus when embeddings work.
    retrieval_probe = None
    if embedder_error is None:
        from app.ai.retrieval import retrieve

        probe = retrieve(
            db,
            "How many days of annual leave are employees entitled to each year?",
            role="employee",
            department=None,
            location="global",
            embedder=embedder,
        )
        ids = [item.get("policy_id") for item in probe.get("citations") or []]
        retrieval_probe = {
            "mode": probe.get("retrieval_mode"),
            "confidence": probe.get("confidence"),
            "citations": ids,
            "includes_current_annual_leave": "annual-leave-v3" in ids,
            "excludes_expired_annual_leave": "annual-leave-v2" not in ids,
            "embedding_error": probe.get("embedding_error"),
        }
        if not retrieval_probe["includes_current_annual_leave"]:
            issues.append({"slug": "annual-leave-v3", "issue": "not_retrieved_by_probe"})
        if not retrieval_probe["excludes_expired_annual_leave"]:
            issues.append({"slug": "annual-leave-v2", "issue": "expired_policy_retrieved"})

    healthy = embedder_error is None and not issues and failed == 0 and indexed == len(documents)
    return {
        "healthy": healthy,
        "embedding": {
            "provider": settings.embedding_provider,
            "model": model_id,
            "dimensions": dimensions,
            "batch_size": getattr(settings, "embedding_batch_size", 32),
            "error": embedder_error,
        },
        "documents": {
            "total": len(documents),
            "indexed": indexed,
            "failed": failed,
            "pending": pending,
            "current": current_docs,
            "expired_or_superseded": expired_or_superseded,
        },
        "chunks": {
            "total": len(chunks),
            "with_embeddings": sum(1 for row in chunks if row.embedding),
        },
        "issues": issues,
        "retrieval_probe": retrieval_probe,
        "note": "Structured employee/leave/attendance records remain in PostgreSQL tables and are never substituted by embeddings.",
    }


def _job(db: Session, document_id: int | None, action: str, status: str, detail: str) -> None:
    now: datetime = utcnow()
    db.add(
        IndexJob(
            document_id=document_id,
            action=action,
            status=status,
            detail=detail[:500],
            started_at=now,
            finished_at=now,
        )
    )


def parse_optional_date(value: str | None) -> date | None:
    cleaned = (value or "").strip()
    if cleaned.lower() in {"", "none", "null"}:
        return None
    return date.fromisoformat(cleaned)
