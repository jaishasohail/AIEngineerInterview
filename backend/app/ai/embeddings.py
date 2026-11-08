"""Dense embedding providers.

The chat completion API is never used to embed text. Local and external embedding
settings are independent of OPENROUTER_MODEL. A failure raises EmbeddingUnavailable.
Retrieval must not invent vectors or substitute feature hashing.
"""

from __future__ import annotations

import logging
import math

import httpx

from app.config import BACKEND_ROOT, get_settings

logger = logging.getLogger("helios.embeddings")


class EmbeddingUnavailable(RuntimeError):
    pass


class Embedder:
    provider: str
    model_id: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalModel2Vec(Embedder):
    """Distilled static dense embeddings. This is not feature hashing."""

    provider = "local"

    def __init__(self, model_id: str, dimensions: int) -> None:
        self.model_id = model_id
        self.dimensions = dimensions
        try:
            from model2vec import StaticModel
        except ImportError as exc:
            raise EmbeddingUnavailable("The local embedding package model2vec is not installed.") from exc
        try:
            self._model = StaticModel.from_pretrained(model_id)
        except Exception as exc:
            raise EmbeddingUnavailable(f"Local embedding model {model_id} could not be loaded: {exc}") from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            raw = self._model.encode(texts, show_progress_bar=False)
        except TypeError:
            raw = self._model.encode(texts)
        except Exception as exc:
            raise EmbeddingUnavailable(f"Local embedding failed: {exc}") from exc
        vectors = []
        for row in raw:
            vector = [float(value) for value in row]
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        if len(vectors) != len(texts):
            raise EmbeddingUnavailable("The embedding provider returned an unexpected number of vectors.")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise EmbeddingUnavailable(
                    f"Model {self.model_id} returned dimension {len(vector)}. EMBEDDING_DIMENSIONS is {self.dimensions}."
                )
        return vectors


class OpenAICompatibleEmbeddings(Embedder):
    provider = "openai_compatible"

    def __init__(self, model_id: str, dimensions: int, api_key: str, base_url: str, timeout: float) -> None:
        if not api_key.strip():
            raise EmbeddingUnavailable("EMBEDDING_API_KEY is empty, so the external embedding provider cannot be called.")
        if not base_url.strip():
            raise EmbeddingUnavailable(
                "EMBEDDING_BASE_URL is empty. The chat completion endpoint is not used for embeddings."
            )
        self.model_id = model_id
        self.dimensions = dimensions
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = httpx.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self.model_id, "input": texts},
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise EmbeddingUnavailable(f"Embedding API request failed: {exc}") from exc
        if response.status_code >= 400:
            raise EmbeddingUnavailable(f"Embedding API returned HTTP {response.status_code}.")
        try:
            payload = response.json()
            rows = sorted(payload.get("data") or [], key=lambda item: item.get("index", 0))
            vectors = [[float(value) for value in row["embedding"]] for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingUnavailable("Embedding API returned a response that could not be read.") from exc
        if len(vectors) != len(texts):
            raise EmbeddingUnavailable("Embedding API returned an unexpected number of vectors.")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise EmbeddingUnavailable(
                    f"Embedding API returned dimension {len(vector)}. EMBEDDING_DIMENSIONS is {self.dimensions}."
                )
        return vectors


_embedder: Embedder | None = None
_signature: tuple | None = None
_reranker = None


def get_embedder() -> Embedder:
    global _embedder, _signature
    settings = get_settings()
    signature = (
        settings.embedding_provider.strip().lower(),
        settings.embedding_model,
        settings.embedding_dimensions,
        settings.embedding_base_url,
    )
    if _embedder is not None and _signature == signature:
        return _embedder
    provider = signature[0]
    if provider == "local":
        embedder: Embedder = LocalModel2Vec(settings.embedding_model, settings.embedding_dimensions)
    elif provider == "openai_compatible":
        embedder = OpenAICompatibleEmbeddings(
            settings.embedding_model,
            settings.embedding_dimensions,
            settings.embedding_api_key,
            settings.embedding_base_url,
            settings.embedding_timeout_seconds,
        )
    else:
        raise EmbeddingUnavailable(
            f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'. Use local or openai_compatible."
        )
    _embedder = embedder
    _signature = signature
    logger.info("Embedding provider ready: %s %s (%s dimensions)", embedder.provider, embedder.model_id, embedder.dimensions)
    return embedder


def reset_embedder() -> None:
    global _embedder, _signature, _reranker
    _embedder = None
    _signature = None
    _reranker = None


def rerank_passages(query: str, passages: list[str]) -> tuple[list[float] | None, str]:
    """Optional cross-encoder rerank. Disabled stays disabled. A failure is reported, not hidden."""
    global _reranker
    settings = get_settings()
    if not settings.rerank_enabled:
        return None, "disabled"
    if not passages:
        return [], "applied"
    try:
        if _reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            cache = BACKEND_ROOT / "data" / "models"
            cache.mkdir(parents=True, exist_ok=True)
            _reranker = TextCrossEncoder(model_name=settings.rerank_model, cache_dir=str(cache))
        scores = [float(score) for score in _reranker.rerank(query, passages)]
    except Exception as exc:
        return None, f"failed: {exc}"
    if len(scores) != len(passages):
        return None, "failed: reranker returned a different number of scores"
    return scores, "applied"
