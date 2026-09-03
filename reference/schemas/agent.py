"""API DTOs for the agentic query endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.query import CitationOut, SourceOut
from app.schemas.verification import VerificationOut


class AgentQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: str | None = None
    max_iterations: int | None = Field(default=None, ge=1, le=5)
    verify: bool = False


class AgentQueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    sources: list[SourceOut]
    tools_run: list[str]
    iterations: int
    verification: VerificationOut | None = None
