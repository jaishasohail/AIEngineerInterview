from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent

# Local development only. Production must set DATABASE_URL and JWT_SECRET in the environment.
# Host development uses the published Compose port. The Compose backend uses postgres:5432.
DEV_DATABASE_URL = "postgresql+psycopg://helios:helios@localhost:55432/helios"
DEV_JWT_SECRET = "dev-only-change-me-helios-hr-copilot"
KNOWN_DEVELOPMENT_SECRETS = frozenset(
    {
        DEV_JWT_SECRET,
        "test-secret-helios-hr-copilot-32b",
        "secret",
        "changeme",
        "password",
        "change-me",
        "replace-with-a-random-string-at-least-32-characters",
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Helios HR Intelligence Copilot"
    environment: str = "development"
    database_url: str = DEV_DATABASE_URL
    checkpoint_path: str = str(BACKEND_ROOT / "data" / "checkpoints.sqlite")
    # auto (default): checkpoints on CHECKPOINT_PATH even when DATABASE_URL is PostgreSQL.
    # postgres: co-locate LangGraph state in PostgreSQL (not recommended for long-running dev/test).
    # sqlite: always use CHECKPOINT_PATH.
    checkpoint_store: str = "auto"
    jwt_secret: str = DEV_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_ttl_hours: int = 12
    org_clock: str = "2026-11-07"
    auto_seed: bool = True
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_timeout_seconds: float = 40
    openrouter_max_retries: int = 3
    embedding_provider: str = "local"
    embedding_model: str = "minishlab/potion-retrieval-32M"
    embedding_dimensions: int = 512
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_timeout_seconds: float = 60
    embedding_batch_size: int = 32
    rerank_enabled: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openrouter_api_key.strip())

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


def get_settings() -> Settings:
    settings = Settings()
    environment = (settings.environment or "development").strip().lower()
    settings.environment = environment
    if environment == "production":
        if settings.database_url.strip() == DEV_DATABASE_URL:
            raise RuntimeError("DATABASE_URL must be set explicitly when ENVIRONMENT is production.")
        secret = settings.jwt_secret.strip()
        if len(secret) < 32 or secret in KNOWN_DEVELOPMENT_SECRETS:
            raise RuntimeError("JWT_SECRET is missing or uses a known development value. Refusing to start.")
    elif not settings.jwt_secret.strip():
        settings.jwt_secret = DEV_JWT_SECRET
    return settings
