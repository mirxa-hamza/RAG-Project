"""Ingestion orchestration: bytes -> parsed -> chunked -> embedded -> stored."""

from __future__ import annotations

import hashlib

from app.core.config import Settings
from app.core.errors import AppError, NotFoundError
from app.core.logging import get_logger
from app.models.document import Document, DocumentStatus
from app.services.document_store import DocumentStore
from app.services.ingestion.chunker import chunk_document
from app.services.ingestion.embeddings import Embedder
from app.services.ingestion.parsers import parse
from app.services.vector_store import VectorStore

logger = get_logger(__name__)


class IngestionService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        document_store: DocumentStore,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._vectors = vector_store
        self._docs = document_store
        self._settings = settings

    def ingest(self, *, filename: str, content_type: str, data: bytes) -> Document:
        """Run the full ingestion pipeline and return the resulting Document.

        Idempotent: re-uploading identical bytes returns the existing ready
        document without re-embedding.
        """
        content_hash = hashlib.sha256(data).hexdigest()
        existing = self._docs.find_ready_by_hash(content_hash)
        if existing is not None:
            logger.info("ingest_skipped_duplicate", document_id=existing.id)
            return existing

        doc = Document.from_upload(filename=filename, content_type=content_type, data=data)
        self._docs.add(doc)
        doc.mark(DocumentStatus.PROCESSING)

        try:
            parsed = parse(filename=filename, content_type=content_type, data=data)
            chunks = chunk_document(
                parsed,
                document_id=doc.id,
                filename=filename,
                max_tokens=self._settings.chunk_tokens,
                overlap_tokens=self._settings.chunk_overlap_tokens,
            )
            vectors = self._embedder.embed_documents([c.text for c in chunks])
            self._vectors.upsert(chunks, vectors)
            self._docs.set_chunk_ids(doc.id, [c.id for c in chunks])
            doc.chunk_count = len(chunks)
            doc.mark(DocumentStatus.READY)
            logger.info("ingest_complete", document_id=doc.id, chunks=len(chunks))
        except AppError as exc:
            doc.mark(DocumentStatus.FAILED, error=exc.message)
            raise
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            doc.mark(DocumentStatus.FAILED, error=str(exc))
            logger.error("ingest_failed", document_id=doc.id, error=str(exc))
            raise

        return doc

    def delete_document(self, document_id: str) -> None:
        doc = self._docs.get(document_id)
        if doc is None:
            raise NotFoundError(f"Document {document_id} not found")
        chunk_ids = self._docs.get_chunk_ids(document_id)
        self._vectors.delete(chunk_ids)
        self._docs.delete(document_id)
        logger.info("document_deleted", document_id=document_id, chunks=len(chunk_ids))
