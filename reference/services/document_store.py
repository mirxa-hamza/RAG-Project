"""In-memory document metadata store.

Tracks Document records and each document's chunk ids (needed to delete
vectors by id on Pinecone serverless). Process-local and non-durable — a
real database replaces this in Phase 6. Access is guarded by a lock so it is
safe under the threadpool FastAPI uses for sync endpoints.
"""

from __future__ import annotations

import threading

from app.models.document import Document, DocumentStatus


class DocumentStore:
    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
        self._chunk_ids: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def add(self, doc: Document) -> None:
        with self._lock:
            self._docs[doc.id] = doc

    def get(self, document_id: str) -> Document | None:
        return self._docs.get(document_id)

    def list(self) -> list[Document]:
        return sorted(self._docs.values(), key=lambda d: d.created_at, reverse=True)

    def find_ready_by_hash(self, content_hash: str) -> Document | None:
        return next(
            (
                d
                for d in self._docs.values()
                if d.content_hash == content_hash and d.status is DocumentStatus.READY
            ),
            None,
        )

    def set_chunk_ids(self, document_id: str, ids: list[str]) -> None:
        with self._lock:
            self._chunk_ids[document_id] = ids

    def get_chunk_ids(self, document_id: str) -> list[str]:
        return self._chunk_ids.get(document_id, [])

    def delete(self, document_id: str) -> None:
        with self._lock:
            self._docs.pop(document_id, None)
            self._chunk_ids.pop(document_id, None)
