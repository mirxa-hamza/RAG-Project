"""Vector (semantic) search tool over the ingested corpus."""

from __future__ import annotations

from app.agent.state import Evidence
from app.core.logging import get_logger
from app.services.ingestion.embeddings import Embedder
from app.services.vector_store import VectorStore

logger = get_logger(__name__)


class VectorSearchTool:
    name = "vector_search"
    description = (
        "Semantic search over the user's ingested document corpus. Use for "
        "questions answerable from uploaded documents."
    )

    def __init__(self, *, embedder: Embedder, vector_store: VectorStore, top_k: int) -> None:
        self._embedder = embedder
        self._vectors = vector_store
        self._top_k = top_k

    def run(self, query: str, *, document_id: str | None = None) -> list[Evidence]:
        vector = self._embedder.embed_query(query)
        filter_ = {"document_id": document_id} if document_id else None
        results = self._vectors.query(vector, top_k=self._top_k, filter=filter_)
        evidence = [
            Evidence(
                text=r.text,
                origin="vector",
                score=r.score,
                source_id=r.chunk_id,
                title=r.filename,
                document_id=r.document_id,
                page=r.page,
            )
            for r in results
        ]
        logger.info("vector_search", query_len=len(query), hits=len(evidence))
        return evidence
