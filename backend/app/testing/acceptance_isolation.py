"""Disposable acceptance-test records with explicit ownership tracking.

Uses seeded demo users only. Creates tagged conversations; tracks ids created during
a run; cleans up only test-owned rows after verification. Does not delete audit history,
approvals, or leave requests unless they were created in the run and still owned.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from app.models import Conversation, Message, PendingAction
from app.testing.checkpoint_store import (
    database_uses_postgres_checkpoints,
    prune_postgres_checkpoint_threads,
    thread_ids_for_conversations,
)

ACCEPTANCE_TITLE_PREFIX = "[acceptance:"
RUN_ID_RE = re.compile(r"\[acceptance:([^\]]+)\]", re.IGNORECASE)


@dataclass
class AcceptanceRun:
    run_id: str
    conversation_ids: list[int] = field(default_factory=list)
    thread_ids: list[str] = field(default_factory=list)
    pending_action_ids: list[int] = field(default_factory=list)
    leave_request_ids: list[int] = field(default_factory=list)
    approval_ids: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, label: str | None = None) -> AcceptanceRun:
        suffix = (label or "run").strip().lower().replace(" ", "-")[:40]
        run_id = f"{suffix}-{uuid.uuid4().hex[:10]}"
        return cls(run_id=run_id)

    def title_for(self, prompt: str) -> str:
        base = (prompt or "scenario").strip()[:72]
        return f"{ACCEPTANCE_TITLE_PREFIX}{self.run_id}] {base}"

    def register_chat(self, payload: dict) -> None:
        cid = payload.get("conversation_id")
        if cid is not None and int(cid) not in self.conversation_ids:
            self.conversation_ids.append(int(cid))
        pending = payload.get("pending_action") or {}
        paid = pending.get("pending_action_id")
        if paid is not None and int(paid) not in self.pending_action_ids:
            self.pending_action_ids.append(int(paid))

    def register_thread(self, thread_id: str | None) -> None:
        if thread_id and thread_id not in self.thread_ids:
            self.thread_ids.append(thread_id)

    def register_leave_request(self, request_id: int) -> None:
        if request_id not in self.leave_request_ids:
            self.leave_request_ids.append(int(request_id))

    def register_approval(self, approval_id: int) -> None:
        if approval_id not in self.approval_ids:
            self.approval_ids.append(int(approval_id))


def conversations_owned_by_run(db: Session, run_id: str) -> list[Conversation]:
    pattern = f"{ACCEPTANCE_TITLE_PREFIX}{run_id}]%"
    return list(db.scalars(select(Conversation).where(Conversation.title.ilike(pattern))).all())


def verify_ownership(db: Session, run: AcceptanceRun) -> dict:
    """Return counts of rows tagged to this run without mutating data."""
    convs = conversations_owned_by_run(db, run.run_id)
    conv_ids = [c.id for c in convs]
    msg_count = 0
    if conv_ids:
        msg_count = int(
            db.execute(
                text("SELECT count(*) FROM messages WHERE conversation_id = ANY(:ids)"),
                {"ids": conv_ids},
            ).scalar()
            or 0
        )
    pending = []
    if run.pending_action_ids:
        pending = list(
            db.scalars(select(PendingAction).where(PendingAction.id.in_(run.pending_action_ids))).all()
        )
    return {
        "run_id": run.run_id,
        "conversation_ids": conv_ids,
        "conversation_count": len(conv_ids),
        "message_count": msg_count,
        "thread_ids_tracked": list(run.thread_ids),
        "pending_actions_tracked": [p.id for p in pending],
    }


def cleanup_run(
    db: Session,
    run: AcceptanceRun,
    *,
    database_url: str,
    delete_conversations: bool = True,
    prune_checkpoints: bool = True,
    cancel_pending_actions: bool = True,
) -> dict:
    """Remove only rows owned by this acceptance run. Preserves shared audit logs."""
    convs = conversations_owned_by_run(db, run.run_id)
    conv_ids = [c.id for c in convs]
    if not conv_ids and run.conversation_ids:
        # Fallback: only ids registered in-run and title still matches marker
        conv_ids = [
            cid
            for cid in run.conversation_ids
            if db.get(Conversation, cid) and RUN_ID_RE.search(db.get(Conversation, cid).title or "")
        ]
    threads = list(run.thread_ids)
    threads.extend(thread_ids_for_conversations(db, conv_ids))
    threads = sorted({t for t in threads if t})

    stats: dict[str, int | dict] = {
        "conversations": 0,
        "messages": 0,
        "pending_actions_cancelled": 0,
        "checkpoint_prune": {},
    }

    if cancel_pending_actions and run.pending_action_ids:
        rows = list(
            db.scalars(
                select(PendingAction).where(
                    PendingAction.id.in_(run.pending_action_ids),
                    PendingAction.status == "awaiting_confirmation",
                )
            ).all()
        )
        for row in rows:
            row.status = "cancelled"
            row.updated_at = datetime.now(timezone.utc)
        stats["pending_actions_cancelled"] = len(rows)

    if delete_conversations and conv_ids:
        stats["messages"] = int(
            db.execute(
                text("SELECT count(*) FROM messages WHERE conversation_id = ANY(:ids)"),
                {"ids": conv_ids},
            ).scalar()
            or 0
        )
        db.execute(delete(Message).where(Message.conversation_id.in_(conv_ids)))
        db.execute(delete(Conversation).where(Conversation.id.in_(conv_ids)))
        stats["conversations"] = len(conv_ids)

    if prune_checkpoints and threads and database_uses_postgres_checkpoints(database_url):
        stats["checkpoint_prune"] = prune_postgres_checkpoint_threads(db, threads)
    db.commit()
    return stats
