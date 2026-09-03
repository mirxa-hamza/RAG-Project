"""Agent working-memory state for the LangGraph orchestrator.

``evidence`` uses an additive reducer so each tool node appends its findings
rather than overwriting; every other field is last-write-wins.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

from app.services.retrieval.answerer import Citation
from app.services.retrieval.context import AssembledContext

EvidenceOrigin = Literal["vector", "web"]


class Evidence(BaseModel):
    """A single retrieved item from any tool, normalized for merging."""

    text: str
    origin: EvidenceOrigin
    score: float
    # Provenance (vector hits carry these; web hits map url->filename, title->page-less)
    source_id: str  # chunk_id or url
    title: str  # filename or web page title
    document_id: str | None = None
    page: int | None = None
    url: str | None = None


class AgentState(TypedDict, total=False):
    question: str
    document_id: str | None

    plan: list[str]  # tool names selected by the planner
    tools_run: Annotated[list[str], add]
    evidence: Annotated[list[Evidence], add]  # appended by each tool node

    context: AssembledContext | None
    answer: str
    citations: list[Citation]

    iteration: int
    max_iterations: int
    done: bool
