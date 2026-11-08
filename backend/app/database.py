from collections.abc import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

_engine = None
SessionLocal: sessionmaker[Session] | None = None


class Base(DeclarativeBase):
    pass


def configure_database(url: str) -> None:
    global _engine, SessionLocal
    if _engine is not None:
        _engine.dispose()
    kwargs: dict = {"future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10
        kwargs["pool_timeout"] = 30
    _engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(_engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    SessionLocal = sessionmaker(bind=_engine, autoflush=True, expire_on_commit=False, future=True)


def get_engine():
    if _engine is None:
        raise RuntimeError("Database is not configured")
    return _engine


def create_tables() -> None:
    from app import models  # noqa: F401

    Base.metadata.create_all(get_engine())


def ensure_sqlite_schema() -> None:
    """Create tables. Recreate a local SQLite file that predates the vector index columns."""
    from sqlalchemy import inspect

    engine = get_engine()
    if engine.dialect.name != "sqlite":
        return
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "policy_documents" in tables:
        columns = {column["name"] for column in inspector.get_columns("policy_documents")}
        if "content_hash" not in columns or "holidays" not in tables:
            Base.metadata.drop_all(engine)
    create_tables()


def enable_pgvector() -> None:
    engine = get_engine()
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def upgrade_postgres() -> None:
    from alembic import command
    from alembic.config import Config

    from app.config import BACKEND_ROOT

    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    command.upgrade(config, "head")


def wait_for_database(url: str, attempts: int = 30) -> None:
    if url.startswith("sqlite"):
        return
    import time

    last_error: Exception | None = None
    probe = create_engine(url, pool_pre_ping=True, future=True)
    try:
        for _ in range(attempts):
            try:
                with probe.connect() as connection:
                    connection.execute(text("SELECT 1"))
                return
            except Exception as exc:
                last_error = exc
                time.sleep(1)
    finally:
        probe.dispose()
    raise RuntimeError(f"PostgreSQL did not become ready: {last_error}")


def session_scope() -> Generator[Session, None, None]:
    if SessionLocal is None:
        raise RuntimeError("Database is not configured")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
