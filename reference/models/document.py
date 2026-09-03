"""Domain models for documents and chunks.

These are the internal source of truth. The HTTP layer never returns these
directly — it maps them to the DTOs in ``app.schemas.documents`` so the API
contract can evolve independently of internal representation.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    """Generate a self-describing, sortable-ish id like ``doc_1a2b3c...``."""
    return f"{prefix}_{uuid.uuid4().hex}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ChunkMetadata(BaseModel):
    """Provenance for a chunk — travels into the vector store as metadata."""

    document_id: str
    filename: str
    page: int | None = None
    start_char: int | None = None
    end_char: int | None = None


class Chunk(BaseModel):
    """A single searchable unit of text derived from a document."""

    id: str = Field(default_factory=lambda: new_id("chk"))
    document_id: str
    text: str
    ordinal: int  # position of the chunk within its document (0-based)
    token_count: int = 0
    metadata: ChunkMetadata


class Document(BaseModel):
    """An uploaded source document and its processing lifecycle."""

    id: str = Field(default_factory=lambda: new_id("doc"))
    filename: str
    content_type: str
    size_bytes: int
    content_hash: str
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @classmethod
    def from_upload(cls, *, filename: str, content_type: str, data: bytes) -> Document:
        """Build a pending Document from raw upload bytes.

        The content hash makes ingestion idempotent: re-uploading identical
        bytes yields the same hash, so we can skip re-embedding.
        """
        return cls(
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            content_hash=hashlib.sha256(data).hexdigest(),
        )

    def mark(self, status: DocumentStatus, *, error: str | None = None) -> None:
        self.status = status
        self.error = error
        self.updated_at = _utcnow()
