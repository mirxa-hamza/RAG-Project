"""Flat API DTO for verification results."""

from __future__ import annotations

from pydantic import BaseModel

from app.services.verification.service import VerificationResult


class VerificationOut(BaseModel):
    trustworthy: bool
    verdict: str
    confidence: float
    grounded: bool
    grounding_score: float
    unsupported_claims: list[str]
    citations_valid: bool
    citation_coverage: float
    invalid_citation_indices: list[int]
    uncited_sentences: list[str]
    reasoning: str

    @classmethod
    def from_result(cls, r: VerificationResult) -> VerificationOut:
        return cls(
            trustworthy=r.trustworthy,
            verdict=r.verdict,
            confidence=r.confidence,
            grounded=r.grounding.grounded,
            grounding_score=r.grounding.score,
            unsupported_claims=r.grounding.unsupported_claims,
            citations_valid=r.citations.valid,
            citation_coverage=r.citations.coverage,
            invalid_citation_indices=r.citations.invalid_indices,
            uncited_sentences=r.citations.uncited_sentences,
            reasoning=r.grounding.reasoning,
        )
