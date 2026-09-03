"""Verification service: combine grounding + citation checks into a verdict."""

from __future__ import annotations

from pydantic import BaseModel

from app.core.logging import get_logger
from app.services.retrieval.context import AssembledContext
from app.services.verification.citations import CitationReport, validate_citations
from app.services.verification.grounding import GroundingReport, GroundingVerifier

logger = get_logger(__name__)

_HIGH = 0.75
_MEDIUM = 0.5


class VerificationResult(BaseModel):
    grounding: GroundingReport
    citations: CitationReport
    confidence: float
    verdict: str  # "high" | "medium" | "low"
    trustworthy: bool


def _confidence(grounding: GroundingReport, citations: CitationReport) -> float:
    # Grounding is the gate (multiplicative): an ungrounded answer cannot score
    # high just because it is well-cited. Coverage only modulates within
    # [0.7, 1.0]; invalid citations halve the result.
    validity_penalty = 1.0 if citations.valid else 0.5
    coverage_factor = 0.7 + 0.3 * citations.coverage
    return round(grounding.score * coverage_factor * validity_penalty, 3)


def _verdict(confidence: float) -> str:
    if confidence >= _HIGH:
        return "high"
    if confidence >= _MEDIUM:
        return "medium"
    return "low"


class VerificationService:
    def __init__(self, *, grounding_verifier: GroundingVerifier) -> None:
        self._grounding = grounding_verifier

    def verify(
        self, *, question: str, answer: str, context: AssembledContext
    ) -> VerificationResult:
        grounding = self._grounding.verify(question=question, answer=answer, context=context)
        citations = validate_citations(answer, context)
        confidence = _confidence(grounding, citations)
        result = VerificationResult(
            grounding=grounding,
            citations=citations,
            confidence=confidence,
            verdict=_verdict(confidence),
            trustworthy=grounding.grounded and citations.valid and confidence >= _MEDIUM,
        )
        logger.info(
            "verification_complete",
            confidence=confidence,
            verdict=result.verdict,
            trustworthy=result.trustworthy,
        )
        return result
