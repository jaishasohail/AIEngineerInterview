import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.access import AuthorizationError
from app.ai.embeddings import EmbeddingUnavailable
from app.ai.runtime import checkpoint_store_backend, get_graph, reset_graph
from app.api.routes import router
from app.config import get_settings
from app.database import (
    configure_database,
    enable_pgvector,
    ensure_sqlite_schema,
    session_scope,
    upgrade_postgres,
    wait_for_database,
)
from app.ai.ingest import ingest_policies
from app.seed import ensure_seeded
from app.services.indexing import refresh_stale_policies, sync_policy_lifecycle, verify_index

logger = logging.getLogger("helios")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    wait_for_database(settings.database_url)
    configure_database(settings.database_url)
    if settings.database_url.startswith("sqlite"):
        ensure_sqlite_schema()
    else:
        enable_pgvector()
        upgrade_postgres()
    reset_graph()
    get_graph()
    logger.info("LangGraph checkpoint store: %s", checkpoint_store_backend())
    logger.disabled = False
    if settings.auto_seed:
        db = next(session_scope())
        try:
            ensure_seeded(db)
            try:
                ingest_policies(db)
                sync_policy_lifecycle(db)
                refresh_stale_policies(db)
                health = verify_index(db)
                if health.get("healthy"):
                    logger.info(
                        "Policy index healthy: %s documents, %s chunks",
                        health["documents"]["total"],
                        health["chunks"]["total"],
                    )
                else:
                    logger.warning("Policy index issues detected: %s", health.get("issues"))
            except EmbeddingUnavailable as exc:
                logger.error("Dense embeddings are unavailable. Retrieval will report degraded keyword mode. %s", exc)
        finally:
            db.close()
    ready = (
        f"Helios ready. LLM enabled={settings.llm_enabled} "
        f"embedding={settings.embedding_provider}/{settings.embedding_model}"
    )
    logger.info(ready)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "helios", "llm_enabled": settings.llm_enabled}

    @app.exception_handler(AuthorizationError)
    async def authorization_error(_request: Request, exc: AuthorizationError) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": exc.message})

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


app = create_app()
