"""Deterministic citation validation.

Structural checks that complement the LLM grounding judge: every citation
must reference a real source, and substantive claims should carry a citation.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.services.retrieval.answerer import INSUFFICIENT_EVIDENCE
from app.services.retrieval.context import AssembledContext

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MIN_FACTUAL_CHARS = 15  # ignore very short fragments ("Yes.", "OK.")


class CitationReport(BaseModel):
    valid: bool
    invalid_indices: list[int] = Field(default_factory=list)
    uncited_sentences: list[str] = Field(default_factory=list)
    coverage: float = Field(ge=0.0, le=1.0, default=1.0)


def _sentence_is_factual(sentence: str) -> bool:
    stripped = sentence.strip()
    return len(stripped) >= _MIN_FACTUAL_CHARS


def validate_citations(answer: str, context: AssembledContext) -> CitationReport:
    stripped = answer.strip()
    if not stripped or stripped == INSUFFICIENT_EVIDENCE:
        return CitationReport(valid=True, coverage=1.0)

    valid_indices = {p.index for p in context.passages}

    invalid: list[int] = []
    factual_total = 0
    cited_total = 0
    uncited: list[str] = []

    for sentence in _SENTENCE_SPLIT.split(stripped):
        cited = [int(m.group(1)) for m in _CITATION_RE.finditer(sentence)]
        for idx in cited:
            if idx not in valid_indices and idx not in invalid:
                invalid.append(idx)

        if _sentence_is_factual(sentence):
            factual_total += 1
            if cited:
                cited_total += 1
            else:
                uncited.append(sentence.strip())

    coverage = 1.0 if factual_total == 0 else cited_total / factual_total
    return CitationReport(
        valid=not invalid,
        invalid_indices=invalid,
        uncited_sentences=uncited,
        coverage=coverage,
    )
