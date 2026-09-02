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


class PasswordChange(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=200)
    new_password: str = Field(..., min_length=8, max_length=200)


class ConfirmPassword(BaseModel):
    """Deleting an account is irreversible, so the password is required again."""
    password: str = Field(..., min_length=1, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    username: str


class UserPublic(BaseModel):
    id: str
    username: str


class Turn(BaseModel):
    """
    One previous exchange, sent by the client (the backend is stateless).

    The lengths are capped because this is client-supplied text that goes STRAIGHT into the
    LLM request. Unbounded, one request could carry 4,000,000 characters of "history" -
    measured, not hypothetical - which is somebody else's money and the model's context
    window. `HISTORY_TURNS` limits how many turns are used; it does not limit their size.
    """
    question: str = Field("", max_length=4000)
    answer: str = Field("", max_length=12000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # Bounded: an unbounded top_k lets a client build an enormous prompt.
    top_k: int = Field(TOP_K, ge=1, le=MAX_TOP_K)
    # Optional: restrict retrieval to one ingested document. Kept for compatibility with
    # anything already calling the API; `sources` is what the UI sends now.
    source: Optional[str] = None
    # Optional: restrict retrieval to a chosen subset of documents. An empty list and None
    # both mean "search everything" - "search nothing" is not a state the UI can produce,
    # and treating it as one would silently answer "not in these documents" forever.
    # Bounded because each entry becomes a term in a store-side filter.
    sources: Optional[List[str]] = Field(None, max_length=100)

    def wanted_sources(self) -> Optional[List[str]]:
        """The document filter to retrieve with, from either field. None = everything."""
        chosen = [s for s in (self.sources or []) if s]
        if not chosen and self.source:
            chosen = [self.source]
        return chosen or None
    # Recent conversation, oldest first. Only the last HISTORY_TURNS are used, and the
    # assembled history is clamped again by characters in llm._history_messages().
    history: List[Turn] = Field(default_factory=list, max_length=20)

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
