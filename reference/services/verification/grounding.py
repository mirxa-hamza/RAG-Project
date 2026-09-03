"""Grounding verification: an independent LLM-as-judge hallucination check.

Compares a generated answer against the sources it was supposed to use and
flags claims the sources do not support. This is a second, adversarially
framed model call — separate from answer generation — so it acts as a check
rather than a self-assessment.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.services.llm import LLMClient
from app.services.retrieval.answerer import INSUFFICIENT_EVIDENCE
from app.services.retrieval.context import AssembledContext

logger = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

GROUNDING_SYSTEM = (
    "You are a strict fact-checker. You are given a QUESTION, an ANSWER, and the "
    "numbered SOURCES the answer was supposed to be based on. Your job is to decide "
    "whether every factual claim in the ANSWER is supported by the SOURCES.\n"
    "Do not use outside knowledge. A claim is 'supported' only if the SOURCES state "
    "or directly imply it.\n"
    'Respond ONLY with JSON: {"grounded": bool, "score": number between 0 and 1, '
    '"unsupported_claims": ["..."], "reasoning": "..."}. '
    "score = fraction of the answer's claims that are supported."
)


class GroundingReport(BaseModel):
    grounded: bool
    score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning: str = ""


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_report(raw: str) -> GroundingReport | None:
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        score = _clamp(float(data.get("score", 0.5)))
        claims = [str(c) for c in data.get("unsupported_claims", []) if isinstance(c, str)]
        grounded = bool(data.get("grounded", not claims))
        return GroundingReport(
            grounded=grounded,
            score=score,
            unsupported_claims=claims,
            reasoning=str(data.get("reasoning", "")),
        )
    except (TypeError, ValueError):
        return None


def build_grounding_prompt(question: str, answer: str, context: AssembledContext) -> str:
    return (
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        f"SOURCES:\n{context.to_prompt_block()}\n\n"
        "Return the JSON verdict."
    )


class GroundingVerifier:
    def __init__(self, *, llm: LLMClient, max_tokens: int = 600) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def verify(
        self, *, question: str, answer: str, context: AssembledContext
    ) -> GroundingReport:
        stripped = answer.strip()
        # Correctly declining to answer is fully grounded behavior.
        if not stripped or stripped == INSUFFICIENT_EVIDENCE or context.is_empty:
            return GroundingReport(
                grounded=True, score=1.0, reasoning="Nothing to verify (declined or no context)."
            )

        prompt = build_grounding_prompt(question, stripped, context)
        raw = self._llm.generate(
            system=GROUNDING_SYSTEM, prompt=prompt, max_tokens=self._max_tokens
        )
        report = _parse_report(raw)
        if report is None:
            logger.warning("grounding_parse_failed")
            return GroundingReport(
                grounded=True, score=0.5, reasoning="Verifier response could not be parsed."
            )
        logger.info("grounding_verified", grounded=report.grounded, score=report.score)
        return report
