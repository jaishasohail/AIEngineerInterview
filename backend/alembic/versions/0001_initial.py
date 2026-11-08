"""Initial Helios schema, including the pgvector column used for policy chunks."""

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    from sqlalchemy import inspect

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    from app import models  # noqa: F401
    from app.database import Base

    Base.metadata.create_all(bind)
    if bind.dialect.name == "postgresql":
        inspector = inspect(bind)
        indexes = {index["name"] for index in inspector.get_indexes("policy_chunks")}
        if "ix_policy_chunks_embedding_hnsw" not in indexes:
            op.execute(
                "CREATE INDEX ix_policy_chunks_embedding_hnsw ON policy_chunks USING hnsw (embedding vector_cosine_ops)"
            )


def downgrade() -> None:
    from app import models  # noqa: F401
    from app.database import Base

    Base.metadata.drop_all(op.get_bind())
