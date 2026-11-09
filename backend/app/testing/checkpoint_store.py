"""LangGraph checkpoint storage helpers (optional PostgreSQL prune)."""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger("helios.checkpoint_store")

CHECKPOINT_TABLES = ("checkpoints", "checkpoint_writes", "checkpoint_blobs")


def database_uses_postgres_checkpoints(database_url: str) -> bool:
    return database_url.startswith("postgresql")


def prune_postgres_checkpoint_threads(db: Session, thread_ids: list[str]) -> dict[str, int]:
    """Delete LangGraph checkpoint rows for explicit thread ids only."""
    cleaned: dict[str, int] = {}
    ids = [tid.strip() for tid in thread_ids if tid and tid.strip()]
    if not ids:
        return cleaned
    bind = db.get_bind()
    if bind is None or bind.dialect.name != "postgresql":
        logger.warning("checkpoint prune skipped: not PostgreSQL")
        return cleaned
    for table in CHECKPOINT_TABLES:
        result = db.execute(
            text(f"DELETE FROM {table} WHERE thread_id = ANY(:ids)"),
            {"ids": ids},
        )
        cleaned[table] = int(result.rowcount or 0)
    return cleaned


def thread_ids_for_conversations(db: Session, conversation_ids: list[int]) -> list[str]:
    if not conversation_ids:
        return []
    rows = db.execute(
        text(
            """
            SELECT DISTINCT active_thread_id
            FROM conversations
            WHERE id = ANY(:ids) AND active_thread_id IS NOT NULL
            """
        ),
        {"ids": conversation_ids},
    ).fetchall()
    return [str(row[0]) for row in rows if row[0]]
