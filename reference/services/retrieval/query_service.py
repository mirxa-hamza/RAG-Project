"""Baseline (non-agentic) retrieval-augmented query service."""

from __future__ import annotations

from collections.abc import Iterator

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.ingestion.embeddings import Embedder
from app.services.retrieval.answerer import Answerer, Citation, GeneratedAnswer, map_citations
from app.services.retrieval.context import AssembledContext, assemble_context
from app.services.vector_store import VectorStore

logger = get_logger(__name__)


class QueryService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        answerer: Answerer,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._vectors = vector_store
        self._answerer = answerer
        self._settings = settings

    def retrieve(
        self,
        *,
        question: str,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> AssembledContext:
        """Embed the question, query the store, and assemble bounded context."""
        vector = self._embedder.embed_query(question)
        filter_ = {"document_id": document_id} if document_id else None
        results = self._vectors.query(
            vector,
            top_k=top_k or self._settings.retrieval_top_k,
            filter=filter_,
        )
        return assemble_context(results, token_budget=self._settings.context_token_budget)

    def answer_query(
        self,
        *,
        question: str,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> tuple[GeneratedAnswer, AssembledContext]:
        context = self.retrieve(question=question, document_id=document_id, top_k=top_k)
        answer = self._answerer.answer(question=question, context=context)
        logger.info(
            "query_answered",
            used=len(context.passages),
            citations=len(answer.citations),
        )
        return answer, context

    def stream_answer(self, *, question: str, context: AssembledContext) -> Iterator[str]:
        return self._answerer.stream(question=question, context=context)

    def citations_for(self, text: str, context: AssembledContext) -> list[Citation]:
        return map_citations(text, context)
