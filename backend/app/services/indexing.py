"""Idempotent policy ingestion, embedding, and reindexing."""

from __future__ import annotations

import hashlib
import io
import logging
import re
from datetime import date, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.ai.embeddings import Embedder, EmbeddingUnavailable, get_embedder
from app.ai.retrieval import chunk_text, tokenize
from app.clock import utcnow
from app.models import IndexJob, PolicyChunk, PolicyDocument

logger = logging.getLogger("helios.index")
SLUG = re.compile(r"[^a-z0-9]+")


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


def _chunk_metadata(document: PolicyDocument) -> dict:
    return {
        "slug": document.slug,
        "title": document.title,
        "version": document.version,
        "doc_type": document.doc_type,
        "effective_date": document.effective_date.isoformat(),
        "access": document.access,
        "department": document.department,
        "location": document.location,
        "status": document.status,
        "conflict_group": document.conflict_group,
    }


def index_document(db: Session, document: PolicyDocument, *, embedder: Embedder | None = None, force: bool = False) -> str:
    """Regenerate chunk embeddings. Unchanged content for the active model is skipped."""
    try:
        embedder = embedder or get_embedder()
    except EmbeddingUnavailable as exc:
        document.index_status = "failed"
        document.index_error = str(exc)[:500]
        db.flush()
        _job(db, document.id, "reindex", "failed", str(exc))
        return "failed"
    digest = content_hash(document.body or "")
    if (
        not force
        and document.content_hash == digest
        and document.index_status == "indexed"
        and document.embedding_model == embedder.model_id
        and (document.chunk_count or 0) > 0
    ):
        _job(db, document.id, "reindex", "skipped", "Content and embedding model are unchanged.")
        return "skipped"
    if not (document.body or "").strip():
        document.index_status = "failed"
        document.index_error = "The document has no extractable text."
        db.flush()
        _job(db, document.id, "reindex", "failed", document.index_error)
        return "failed"
    chunks = chunk_text(document.body)
    try:
        vectors = embedder.embed_documents([f"{document.title}\n{chunk}" for chunk in chunks])
        if len(vectors) != len(chunks):
            raise EmbeddingUnavailable("The embedding provider returned a different number of vectors than chunks.")
        for vector in vectors:
            if len(vector) != embedder.dimensions:
                raise EmbeddingUnavailable(
                    f"The embedding provider returned dimension {len(vector)}. The active model expects {embedder.dimensions}."
                )
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
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            db.add(
                PolicyChunk(
                    policy_id=document.id,
                    chunk_index=index,
                    content=chunk,
                    embedding=vector,
                    embedding_model=embedder.model_id,
                    embedding_dimensions=embedder.dimensions,
                    token_count=len(tokenize(chunk)),
                    meta_json=_chunk_metadata(document),
                )
            )
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
    documents = db.query(PolicyDocument).order_by(PolicyDocument.id).all()
    for document in documents:
        outcome = index_document(db, document, embedder=embedder, force=force)
        results.append({"slug": document.slug, "outcome": outcome, "error": document.index_error})
    db.commit()
    return results


def refresh_stale_policies(db: Session) -> list[dict]:
    try:
        embedder = get_embedder()
    except EmbeddingUnavailable:
        raise
    stale = (
        db.query(PolicyDocument)
        .filter(
            (PolicyDocument.index_status != "indexed")
            | (PolicyDocument.embedding_model.is_(None))
            | (PolicyDocument.embedding_model != embedder.model_id)
        )
        .count()
    )
    if stale == 0:
        return []
    return reindex_all(db, embedder=embedder, force=False)


def delete_document(db: Session, document: PolicyDocument) -> None:
    db.query(PolicyChunk).filter(PolicyChunk.policy_id == document.id).delete()
    _job(db, document.id, "delete", "succeeded", f"Removed {document.slug} and its vectors.")
    db.delete(document)
    db.commit()


def find_duplicate(db: Session, digest: str) -> PolicyDocument | None:
    if not digest:
        return None
    return db.query(PolicyDocument).filter(PolicyDocument.content_hash == digest).one_or_none()


def mark_superseded(db: Session, slug: str | None) -> None:
    if not slug:
        return
    previous = db.query(PolicyDocument).filter(PolicyDocument.slug == slug).one_or_none()
    if previous is not None and previous.status == "current":
        previous.status = "superseded"


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
