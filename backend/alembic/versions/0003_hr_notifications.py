"""In-app HR workflow notifications."""

from alembic import op
import sqlalchemy as sa

revision = "0003_hr_notifications"
down_revision = "0002_pending_actions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "hr_notifications" not in tables:
        op.create_table(
            "hr_notifications",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("recipient_employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
            sa.Column("recipient_role", sa.String(length=30), nullable=False),
            sa.Column("event_type", sa.String(length=60), nullable=False),
            sa.Column("resource_type", sa.String(length=40), nullable=False),
            sa.Column("resource_id", sa.String(length=60), nullable=False),
            sa.Column("title", sa.String(length=180), nullable=False),
            sa.Column("body", sa.Text(), nullable=False),
            sa.Column("channel", sa.String(length=40), nullable=False),
            sa.Column("delivery_status", sa.String(length=30), nullable=False),
            sa.Column("payload_json", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint(
                "event_type",
                "resource_type",
                "resource_id",
                "recipient_employee_id",
                name="uq_hr_notifications_event_recipient",
            ),
        )
        op.create_index("ix_hr_notifications_recipient_employee_id", "hr_notifications", ["recipient_employee_id"])
        op.create_index("ix_hr_notifications_event_type", "hr_notifications", ["event_type"])
        op.create_index("ix_hr_notifications_created_at", "hr_notifications", ["created_at"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "hr_notifications" in set(inspector.get_table_names()):
        op.drop_table("hr_notifications")
