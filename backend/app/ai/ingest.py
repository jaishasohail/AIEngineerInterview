from datetime import date

from sqlalchemy.orm import Session

from app.knowledge.catalog import POLICY_DIR, materialize_policies, parse_policy_file
from app.models import PolicyDocument
from app.services.indexing import index_document


def _none(value: str) -> str | None:
    cleaned = (value or "").strip()
    if cleaned.lower() in {"", "none", "null"}:
        return None
    return cleaned


def ingest_policies(db: Session) -> int:
    """Upsert the seeded corpus and embed any document whose content or model changed."""
    materialize_policies()
    count = 0
    for path in sorted(POLICY_DIR.glob("*.md")):
        meta, body = parse_policy_file(path)
        expiry = _none(meta.get("expiry_date", ""))
        document = db.query(PolicyDocument).filter(PolicyDocument.slug == meta["id"]).one_or_none()
        fields = dict(
            title=meta["title"],
            doc_type=meta["doc_type"],
            version=meta["version"],
            policy_family=meta["policy_family"],
            effective_date=date.fromisoformat(meta["effective_date"]),
            expiry_date=date.fromisoformat(expiry) if expiry else None,
            supersedes_slug=_none(meta.get("supersedes", "")),
            department=_none(meta.get("department", "")),
            location=meta.get("location") or "global",
            access=meta["access"],
            conflict_group=_none(meta.get("conflict_group", "")),
            status=meta["status"],
            body=body,
            source_path=str(path.name),
            source_format="md",
        )
        if document is None:
            document = PolicyDocument(slug=meta["id"], **fields)
            db.add(document)
            db.flush()
        else:
            for key, value in fields.items():
                setattr(document, key, value)
            db.flush()
        index_document(db, document)
        count += document.chunk_count or 0
    db.commit()
    return count
