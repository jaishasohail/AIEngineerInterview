"""Hybrid policy retrieval.

BM25 and dense vectors are ranked separately and combined with reciprocal rank
fusion. Dense vectors come only from the configured embedding provider. If that
provider fails, the result says so and keyword search is the only signal.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from app.ai.embeddings import Embedder, EmbeddingUnavailable, get_embedder, rerank_passages
from app.ai.fusion import reciprocal_rank_fusion
from app.clock import org_today
from app.models import PolicyChunk, PolicyDocument

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "for", "to", "in", "on", "and", "or", "with", "from", "by", "at",
    "what", "how", "does", "do", "can", "i", "my", "me", "we", "our", "about",
    "this", "that", "it", "as", "if", "when", "who", "which", "after", "many",
}
ACCESS_BY_ROLE = {
    "employee": {"public", "internal"},
    "manager": {"public", "internal"},
    "hr_admin": {"public", "internal", "hr_confidential"},
}
RRF_K = 60
LEXICAL_POOL = 20
DENSE_POOL = 20
DENSE_FLOOR = 0.34


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN.findall(text.lower()) if token not in STOPWORDS and len(token) > 1]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))


def chunk_text(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 2 <= size:
            buffer = (buffer + "\n\n" + paragraph).strip()
            continue
        if buffer:
            chunks.append(buffer)
        if len(paragraph) <= size:
            buffer = paragraph
            continue
        step = max(size - overlap, 1)
        for index in range(0, len(paragraph), step):
            chunks.append(paragraph[index : index + size])
        buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks or [text.strip()]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in docs]
        self.avgdl = (sum(self.doc_len) / len(docs)) if docs else 0
        self.df: Counter[str] = Counter()
        self.tf = []
        for doc in docs:
            counts = Counter(doc)
            self.tf.append(counts)
            for token in counts:
                self.df[token] += 1
        self.n = len(docs)

    def score(self, query: list[str], index: int) -> float:
        if not self.n or self.avgdl == 0:
            return 0.0
        score = 0.0
        frequencies = self.tf[index]
        length = self.doc_len[index] or 1
        for token in query:
            if token not in frequencies:
                continue
            df = self.df[token]
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            freq = frequencies[token]
            denom = freq + self.k1 * (1 - self.b + self.b * length / self.avgdl)
            score += idf * (freq * (self.k1 + 1)) / denom
        return score


def active_documents(db: Session, role: str, department: str | None, location: str | None) -> tuple[list[PolicyDocument], int]:
    today = org_today()
    allowed = ACCESS_BY_ROLE.get(role, {"public"})
    rows = db.query(PolicyDocument).all()
    current = []
    for row in rows:
        if row.status != "current":
            continue
        if row.effective_date > today:
            continue
        if row.expiry_date and row.expiry_date < today:
            continue
        current.append(row)
    superseded = {row.supersedes_slug for row in current if row.supersedes_slug}
    current = [row for row in current if row.slug not in superseded]
    withheld = 0
    visible = []
    for row in current:
        if row.access not in allowed:
            withheld += 1
            continue
        if row.department and department and row.department != department and role != "hr_admin":
            withheld += 1
            continue
        if row.location not in {"global", "none"} and location and row.location != location and role != "hr_admin":
            withheld += 1
            continue
        visible.append(row)
    return visible, withheld


def retrieve(
    db: Session,
    query: str,
    *,
    role: str,
    department: str | None = None,
    location: str | None = None,
    k: int = 4,
    embedder: Embedder | None = None,
) -> dict:
    documents, withheld = active_documents(db, role, department, location)
    if not documents:
        return _empty(withheld, "none", None, None)
    by_id = {document.id: document for document in documents}
    chunks = db.query(PolicyChunk).filter(PolicyChunk.policy_id.in_(list(by_id))).all()
    if not chunks:
        return _empty(withheld, "none", None, "No chunks are indexed for the visible policies.")
    try:
        embedder = embedder or get_embedder()
    except EmbeddingUnavailable as exc:
        embedder = None
        embedding_error = str(exc)
        model_id = None
        dimensions = None
    else:
        embedding_error = None
        model_id = embedder.model_id
        dimensions = embedder.dimensions

    tokenized = [tokenize(chunk.content) for chunk in chunks]
    search_query = _expanded_search(query)
    query_tokens = tokenize(search_query)
    bm25 = BM25(tokenized)
    lexical_scores = [bm25.score(query_tokens, index) for index in range(len(chunks))]
    lexical_order = sorted(range(len(chunks)), key=lambda index: lexical_scores[index], reverse=True)
    lexical_ids = [chunks[index].id for index in lexical_order if lexical_scores[index] > 0][:LEXICAL_POOL]

    retrieval_mode = "hybrid"
    dense_scores: dict[int, float] = {}
    dense_ids: list[int] = []
    mismatched = 0
    if embedder is None:
        retrieval_mode = "degraded_bm25"
    else:
        try:
            query_vector = embedder.embed_query(search_query)
        except EmbeddingUnavailable as exc:
            retrieval_mode = "degraded_bm25"
            embedding_error = str(exc)
            query_vector = None
        if query_vector is not None:
            compatible = []
            for chunk in chunks:
                if chunk.embedding_model != embedder.model_id or chunk.embedding_dimensions != embedder.dimensions or not chunk.embedding:
                    mismatched += 1
                    continue
                compatible.append(chunk)
            if not compatible:
                retrieval_mode = "degraded_bm25"
                embedding_error = embedding_error or (
                    "No chunks use the active embedding model. Reindex before dense retrieval can run."
                )
            else:
                policy_ids = list({chunk.policy_id for chunk in compatible})
                ranked = _dense_search(db, compatible, policy_ids, query_vector, embedder)
                dense_scores = {chunk_id: score for chunk_id, score in ranked}
                if ranked:
                    best_semantic = ranked[0][1]
                    cutoff = max(DENSE_FLOOR, best_semantic * 0.82)
                    dense_ids = [chunk_id for chunk_id, score in ranked if score >= cutoff][:DENSE_POOL]

    rankings = [lexical_ids]
    if dense_ids:
        rankings.append(dense_ids)
    if retrieval_mode == "hybrid" and not dense_scores and embedding_error:
        retrieval_mode = "degraded_bm25"
    fused = reciprocal_rank_fusion(rankings, k=RRF_K)
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    lexical_by_id = {chunks[index].id: lexical_scores[index] for index in range(len(chunks))}

    grouped: dict[int, dict] = {}
    for chunk_id, rrf_score in fused:
        chunk = chunk_by_id[chunk_id]
        lexical = lexical_by_id.get(chunk_id, 0.0)
        semantic = dense_scores.get(chunk_id, 0.0)
        if lexical <= 0 and semantic < DENSE_FLOOR:
            continue
        document = by_id[chunk.policy_id]
        current = grouped.get(document.id)
        if current is None or rrf_score > current["score"]:
            grouped[document.id] = _citation(document, chunk, rrf_score, lexical, semantic)
    ordered = sorted(grouped.values(), key=lambda item: -item["score"])
    if not ordered:
        return _result(
            "weak",
            withheld,
            retrieval_mode,
            model_id,
            embedding_error,
            [],
            [],
            mismatched,
            0,
        )
    top_score = ordered[0]["score"] or 1.0
    citations = [item for item in ordered if item["score"] >= top_score * 0.62][:k]
    citations = _include_conflict_partners(citations, documents, chunks, top_score)
    for citation in citations:
        chunk = chunk_by_id.get(citation.get("chunk_id"))
        if chunk is not None:
            citation["excerpt"] = _excerpt(chunk.content, query=query)
    citations = _keep_supporting(query, citations)
    for citation in citations:
        citation["score"] = round(citation["score"], 4)
        citation["lexical_score"] = round(citation["lexical_score"], 4)
        citation["semantic_score"] = round(citation["semantic_score"], 4)
    conflicts = _conflicts(citations)
    confidence = "conflict" if conflicts else ("strong" if citations else "weak")
    rerank_status = "disabled"
    if citations and retrieval_mode == "hybrid":
        scores, rerank_status = rerank_passages(query, [item["excerpt"] for item in citations])
        if scores is not None:
            citations = [item for _, item in sorted(zip(scores, citations), key=lambda pair: -pair[0])]
            for item, score in zip(citations, sorted(scores, reverse=True)):
                item["rerank_score"] = round(score, 4)
    return _result(
        confidence,
        withheld,
        retrieval_mode,
        model_id,
        embedding_error,
        citations,
        conflicts,
        mismatched,
        round(top_score, 4),
        rerank_status,
    )


def _dense_search(
    db: Session,
    chunks: list[PolicyChunk],
    policy_ids: list[int],
    query_vector: list[float],
    embedder: Embedder,
) -> list[tuple[int, float]]:
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        return _dense_search_pgvector(db, policy_ids, query_vector, embedder)
    scored = []
    for chunk in chunks:
        scored.append((chunk.id, cosine(query_vector, chunk.embedding or [])))
    scored.sort(key=lambda item: -item[1])
    return scored[:DENSE_POOL]


def _dense_search_pgvector(db: Session, policy_ids: list[int], query_vector: list[float], embedder: Embedder) -> list[tuple[int, float]]:
    if not policy_ids:
        return []
    literal = "[" + ",".join(format(value, ".8f") for value in query_vector) + "]"
    statement = text(
        """
        SELECT id, 1 - (embedding <=> CAST(:query AS vector)) AS cosine
        FROM policy_chunks
        WHERE policy_id IN :policy_ids
          AND embedding_model = :model
          AND embedding_dimensions = :dimensions
        ORDER BY embedding <=> CAST(:query AS vector)
        LIMIT :limit
        """
    ).bindparams(bindparam("policy_ids", expanding=True))
    rows = db.execute(
        statement,
        {
            "query": literal,
            "policy_ids": policy_ids,
            "model": embedder.model_id,
            "dimensions": embedder.dimensions,
            "limit": DENSE_POOL,
        },
    ).all()
    return [(int(row.id), float(row.cosine)) for row in rows]


def _document_owner(document: PolicyDocument) -> str:
    """Issuing department named in the document text. The department column is an access scope."""
    issued = re.search(r"issued by ([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)", document.body or "")
    if issued:
        return issued.group(1).strip()
    return ""


def _citation(document: PolicyDocument, chunk: PolicyChunk, score: float, lexical: float, semantic: float) -> dict:
    return {
        "score": score,
        "lexical_score": lexical,
        "semantic_score": semantic,
        "policy_id": document.slug,
        "title": document.title,
        "version": document.version,
        "doc_type": document.doc_type,
        "effective_date": document.effective_date.isoformat(),
        "status": document.status,
        "access": document.access,
        "conflict_group": document.conflict_group,
        "department": document.department,
        "owner": _document_owner(document),
        "excerpt": _excerpt(chunk.content),
        "chunk_id": chunk.id,
    }


def _include_conflict_partners(
    citations: list[dict],
    documents: list[PolicyDocument],
    chunks: list[PolicyChunk],
    top_score: float,
) -> list[dict]:
    """Surface the other side of a conflict only when one side is actually on topic.

    A low-ranked document that merely shares a conflict group must not turn an
    unrelated question into a conflict. The partner is attached when a retrieved
    member is close to the top fused score and itself has a lexical or dense hit.
    """
    present = {item["policy_id"] for item in citations}
    groups = set()
    for item in citations:
        relevant = item.get("lexical_score", 0) > 0 or item.get("semantic_score", 0) >= DENSE_FLOOR
        near_top = top_score <= 0 or item.get("score", 0) >= top_score * 0.85
        if item.get("conflict_group") and relevant and near_top:
            groups.add(item["conflict_group"])
    first_chunk: dict[int, PolicyChunk] = {}
    for chunk in chunks:
        first_chunk.setdefault(chunk.policy_id, chunk)
    for document in documents:
        if not document.conflict_group or document.conflict_group not in groups or document.slug in present:
            continue
        chunk = first_chunk.get(document.id)
        if chunk is None:
            continue
        citations.append(_citation(document, chunk, 0.0, 0.0, 0.0))
        present.add(document.slug)
    return citations


GENERIC_TOPIC = STOPWORDS | {
    "leave", "policy", "policies", "employee", "employees", "staff", "people",
    "person", "day", "days", "work", "working", "worked", "paid", "pay", "current",
    "year", "yearly", "annually", "calendar", "company", "northstar", "global",
    "manager", "time", "request", "requests", "rule", "rules", "please", "tell",
    "say", "says", "said", "granted", "get", "many", "much", "need", "required",
    "using", "used", "about", "does", "each", "per", "into", "over", "than",
    "before", "after", "during", "without", "within", "their", "them", "your",
    "have", "has", "had", "not", "may", "can", "will", "would", "should",
}


def _focus_tokens(query: str) -> list[str]:
    seen = []
    for token in tokenize(query):
        if token in GENERIC_TOPIC or token in seen:
            continue
        seen.append(token)
    return seen


def _token_coverage(tokens: list[str], excerpt: str) -> float:
    if not tokens:
        return 0.0
    present = set(tokenize(excerpt))
    return sum(1 for token in tokens if token in present) / len(tokens)


_REQUIRED_TOPICS = (
    ("resign", "resignation"),
    ("terminate", "termination"),
    ("encash", "encashment"),
    ("settlement",),
)


def _expanded_search(query: str) -> str:
    """Add settlement vocabulary to the search text without changing the question used for support."""
    if re.search(r"\bnotice\b", query, re.IGNORECASE) and not re.search(
        r"unused|encash|settlement", query, re.IGNORECASE
    ):
        return query
    if re.search(r"resign|resignation|terminat|encash|settlement", query, re.IGNORECASE):
        return (
            query
            + " resignation termination final settlement leave encashment unused leave provisions"
        )
    return query


def _missing_required_topic(query: str, citations: list[dict]) -> bool:
    text = query.lower()
    excerpts = " ".join((item.get("excerpt") or "") for item in citations).lower()
    groups = _REQUIRED_TOPICS
    if re.search(r"\bnotice\b", text) and not re.search(r"unused|encash|settlement", text):
        groups = tuple(group for group in _REQUIRED_TOPICS if group[0] not in {"resign", "terminate"})
    for group in groups:
        if any(term in text for term in group) and not any(term in excerpts for term in group):
            return True
    return False


def _keep_supporting(query: str, citations: list[dict]) -> list[dict]:
    """Keep documents that support the question, including both sides of a real conflict.

    Near neighbors from the same broad topic stay in the ranked list until this
    step. They are dropped when their text does not cover the question's
    distinctive terms. A paraphrase with no shared terms keeps the leading
    semantic hit, plus the other current side of that document's conflict.
    """
    if not citations:
        return []
    focus = _focus_tokens(query)
    coverages = [_token_coverage(focus, item.get("excerpt") or "") for item in citations]
    best = max(coverages) if coverages else 0.0
    if best > 0:
        cutoff = best * 0.8
        kept = [item for item, cover in zip(citations, coverages) if cover >= cutoff]
        groups = {item.get("conflict_group") for item in kept if item.get("conflict_group")}
        for item, cover in zip(citations, coverages):
            if item in kept or not item.get("conflict_group"):
                continue
            if item["conflict_group"] in groups and cover > 0:
                kept.append(item)
        if _missing_required_topic(query, kept):
            return []
        return kept
    ranked = sorted(
        citations,
        key=lambda item: (item.get("semantic_score") or 0.0, item.get("lexical_score") or 0.0, item.get("score") or 0.0),
        reverse=True,
    )
    top = ranked[0]
    semantic = top.get("semantic_score") or 0.0
    lexical = top.get("lexical_score") or 0.0
    if semantic < DENSE_FLOOR and lexical <= 0:
        return []
    chosen = [top]
    if len(ranked) > 1 and top.get("conflict_group") and ranked[1].get("conflict_group") == top.get("conflict_group"):
        second = ranked[1]
        gap = semantic - (second.get("semantic_score") or 0.0)
        lexical_ratio = (second.get("lexical_score") or 0.0) / lexical if lexical else 0.0
        second_semantic = second.get("semantic_score") or 0.0
        if (second_semantic >= DENSE_FLOOR and gap < 0.08) or lexical_ratio >= 0.45:
            chosen.append(second)
    if _missing_required_topic(query, chosen):
        return []
    return chosen


def _conflicts(citations: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for citation in citations:
        if citation.get("conflict_group"):
            groups[citation["conflict_group"]].append(citation)
    found = []
    for group, items in groups.items():
        distinct = {item["policy_id"]: item for item in items}
        if len(distinct) < 2:
            continue
        found.append(
            {
                "group": group,
                "policies": [
                    {
                        "policy_id": item["policy_id"],
                        "title": item["title"],
                        "version": item["version"],
                        "excerpt": item["excerpt"],
                        "effective_date": item.get("effective_date") or "",
                        "department": item.get("department") or "",
                        "owner": item.get("owner") or "",
                    }
                    for item in distinct.values()
                ],
                "summary": "Two current documents in this conflict group were both retrieved. No single rule is selected.",
            }
        )
    return found


def _excerpt(content: str, limit: int = 320, query: str = "") -> str:
    compact = " ".join(content.split())
    if len(compact) <= limit:
        return compact
    focus = _focus_tokens(query)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", compact) if part.strip()]
    best = ""
    best_cover = 0.0
    if focus and len(sentences) > 1:
        for start in range(len(sentences)):
            piece = ""
            for sentence in sentences[start:]:
                candidate = (piece + " " + sentence).strip()
                if len(candidate) > limit:
                    break
                piece = candidate
                cover = _token_coverage(focus, piece)
                if cover > best_cover or (cover == best_cover and len(piece) > len(best)):
                    best = piece
                    best_cover = cover
    if best and best_cover > 0:
        return best
    return compact[: limit - 1].rstrip() + "…"


def _empty(withheld: int, mode: str, model: str | None, error: str | None) -> dict:
    return _result("weak", withheld, mode, model, error, [], [], 0, 0)


def _result(
    confidence: str,
    withheld: int,
    mode: str,
    model: str | None,
    error: str | None,
    citations: list[dict],
    conflicts: list[dict],
    mismatched: int,
    top_score: float,
    rerank_status: str = "disabled",
) -> dict:
    return {
        "confidence": confidence,
        "withheld_documents": withheld,
        "retrieval_mode": mode,
        "fusion": "rrf",
        "embedding_model": model,
        "embedding_error": error,
        "mismatched_chunks": mismatched,
        "rerank_status": rerank_status,
        "citations": citations,
        "conflicts": conflicts,
        "top_score": top_score,
    }


def conflict_groups_in_force(db: Session) -> dict[str, list[str]]:
    rows = db.query(PolicyDocument).filter(PolicyDocument.status == "current", PolicyDocument.conflict_group.isnot(None)).all()
    groups = defaultdict(list)
    for row in rows:
        groups[row.conflict_group].append(row.slug)
    return {key: value for key, value in groups.items() if len(value) > 1}
