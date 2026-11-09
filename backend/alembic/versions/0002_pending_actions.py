"""Durable pending actions for human-in-the-loop confirmation."""

from alembic import op
import sqlalchemy as sa

revision = "0002_pending_actions"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "pending_actions" not in tables:
        op.create_table(
            "pending_actions",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("owner_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("conversation_id", sa.Integer(), sa.ForeignKey("conversations.id"), nullable=True),
            sa.Column("action_type", sa.String(length=60), nullable=False),
            sa.Column("status", sa.String(length=30), nullable=False),
            sa.Column("idempotency_key", sa.String(length=180), nullable=False),
            sa.Column("payload_json", sa.JSON(), nullable=False),
            sa.Column("result_json", sa.JSON(), nullable=True),
            sa.Column("thread_id", sa.String(length=80), nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint("idempotency_key", name="uq_pending_actions_idempotency"),
        )
        op.create_index("ix_pending_actions_owner_user_id", "pending_actions", ["owner_user_id"])
        op.create_index("ix_pending_actions_conversation_id", "pending_actions", ["conversation_id"])
        op.create_index("ix_pending_actions_action_type", "pending_actions", ["action_type"])
        op.create_index("ix_pending_actions_status", "pending_actions", ["status"])
        op.create_index("ix_pending_actions_expires_at", "pending_actions", ["expires_at"])

    if "conversations" in tables:
        columns = {column["name"] for column in inspector.get_columns("conversations")}
        if "pending_action_id" not in columns:
            op.add_column("conversations", sa.Column("pending_action_id", sa.Integer(), nullable=True))
            op.create_index("ix_conversations_pending_action_id", "conversations", ["pending_action_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "conversations" in tables:
        columns = {column["name"] for column in inspector.get_columns("conversations")}
        if "pending_action_id" in columns:
            op.drop_index("ix_conversations_pending_action_id", table_name="conversations")
            op.drop_column("conversations", "pending_action_id")
    if "pending_actions" in tables:
        op.drop_table("pending_actions")
