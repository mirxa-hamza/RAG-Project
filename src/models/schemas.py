"""
Request and response shapes for the API.

Kept separate from the route handlers so the wire contract is readable in one place, and
so validation rules (bounds, blank-question rejection) live with the data rather than
being scattered through endpoint bodies.
"""
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from src.core.config import MAX_TOP_K, TOP_K


class Credentials(BaseModel):
    """Signup and login share a shape."""
    # 3 characters minimum keeps usernames typeable and unambiguous; the pattern keeps
    # them safe to use as a path segment and inside a Mongo query.
    username: str = Field(..., min_length=3, max_length=40, pattern=r"^[A-Za-z0-9._-]+$")
    # 8 is a floor, not a policy. The 200 ceiling matters more than it looks: Argon2 will
    # happily hash a 10MB "password", which is a free denial-of-service.
    password: str = Field(..., min_length=8, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    username: str


class UserPublic(BaseModel):
    id: str
    username: str


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
