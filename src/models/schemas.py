"""
Request and response shapes for the API.

Kept separate from the route handlers so the wire contract is readable in one place, and
so validation rules (bounds, blank-question rejection) live with the data rather than
being scattered through endpoint bodies.
"""
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from src.core.config import MAX_TOP_K, TOP_K


class Turn(BaseModel):
    """One previous exchange, sent by the client (the backend is stateless)."""
    question: str = ""
    answer: str = ""


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # Bounded: an unbounded top_k lets a client build an enormous prompt.
    top_k: int = Field(TOP_K, ge=1, le=MAX_TOP_K)
    # Optional: restrict retrieval to one ingested document.
    source: Optional[str] = None
    # Recent conversation, oldest first. Only the last HISTORY_TURNS are used.
    history: List[Turn] = Field(default_factory=list, max_length=50)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question can't be empty.")
        return stripped


class Source(BaseModel):
    source: Optional[str]
    pages: str
    # Neighbour chunks are fetched by index rather than by search, so they carry neither
    # a similarity nor a re-rank score - both are genuinely optional.
    similarity: Optional[float] = None
    rerank_score: Optional[float] = None
    neighbor: bool = False
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    # Set only when a follow-up was rewritten into a standalone question for retrieval.
    search_query: Optional[str] = None
