"""API DTOs for the documents endpoints.

Kept separate from ``app.models.document`` so the public API contract is
decoupled from the internal domain model.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.models.document import Document, DocumentStatus


class DocumentOut(BaseModel):
    """Public view of a document."""

    id: str
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    chunk_count: int
    error: str | None = None
    created_at: datetime

    @classmethod
    def from_model(cls, doc: Document) -> DocumentOut:
        return cls(
            id=doc.id,
            filename=doc.filename,
            content_type=doc.content_type,
            size_bytes=doc.size_bytes,
            status=doc.status,
            chunk_count=doc.chunk_count,
            error=doc.error,
            created_at=doc.created_at,
        )


class DocumentUploadResponse(BaseModel):
    """Returned immediately after an upload is accepted."""

    document_id: str
    status: DocumentStatus


class DocumentListResponse(BaseModel):
    items: list[DocumentOut]
    total: int
