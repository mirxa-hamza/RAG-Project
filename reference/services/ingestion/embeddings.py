"""Embedding generation via fastembed, running locally on CPU.

An ``Embedder`` protocol decouples callers from the concrete provider. The
fastembed implementation runs a quantized ONNX model in-process: no API key, no
network at query time, and no rate limits — which is why there is no retry
wrapper here, unlike a hosted provider.

The model itself is downloaded once (~67 MB for the default) into ``cache_dir``
and reused. That download happens on the FIRST embed call, so the first
ingestion after a fresh install is slow; every call after it is local.

The underlying fastembed model is injectable so tests need neither the package
nor the model weights.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Provider-agnostic embedding interface."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _to_list(vector: Any) -> list[float]:
    """Normalize a numpy array (or any sequence) to plain floats.

    Pinecone rejects numpy types, and keeping the conversion here means callers
    never learn that the local model returns arrays.
    """
    tolist = getattr(vector, "tolist", None)
    if tolist is not None:
        return [float(x) for x in tolist()]
    return [float(x) for x in vector]


class FastEmbedEmbeddings:
    """Local fastembed-backed embedder.

    Uses ``passage_embed`` for documents and ``query_embed`` for questions.
    Retrieval models such as BGE are trained with an instruction prefix on the
    query side only; those helpers apply the right prefix for the chosen model,
    so an asymmetric search works without hardcoding prompt strings here.
    """

    def __init__(
        self,
        *,
        model: str,
        cache_dir: str | None = None,
        batch_size: int = 64,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._client = client  # injected in tests; lazily created otherwise

    def _get_client(self) -> Any:
        if self._client is None:
            from fastembed import TextEmbedding  # lazy: heavy import, downloads weights

            logger.info("loading_embedding_model", model=self._model)
            self._client = TextEmbedding(model_name=self._model, cache_dir=self._cache_dir)
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = [
            _to_list(v)
            for v in self._get_client().passage_embed(texts, batch_size=self._batch_size)
        ]
        logger.info("embedded_documents", count=len(vectors), model=self._model)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        for vector in self._get_client().query_embed(text):
            return _to_list(vector)
        raise RuntimeError(f"Embedding model {self._model!r} returned no vector for the query")
