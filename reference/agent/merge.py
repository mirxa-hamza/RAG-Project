"""Merge heterogeneous evidence (vector + web) into a bounded, citable context."""

from __future__ import annotations

from app.agent.state import Evidence
from app.services.ingestion import tokenizer
from app.services.retrieval.context import AssembledContext, ContextPassage


def merge_evidence(
    evidence: list[Evidence],
    *,
    token_budget: int,
    max_passages: int | None = None,
) -> AssembledContext:
    """Dedupe by source, rank by score, and pack under a token budget.

    Produces the same AssembledContext the answerer consumes, so vector and
    web evidence are cited through one uniform path.
    """
    best_by_source: dict[str, Evidence] = {}
    for e in evidence:
        current = best_by_source.get(e.source_id)
        if current is None or e.score > current.score:
            best_by_source[e.source_id] = e

    ranked = sorted(best_by_source.values(), key=lambda e: e.score, reverse=True)

    passages: list[ContextPassage] = []
    seen_text: set[str] = set()
    used_tokens = 0
    index = 1
    for e in ranked:
        if max_passages is not None and len(passages) >= max_passages:
            break
        normalized = e.text.strip()
        if not normalized or normalized in seen_text:
            continue
        cost = tokenizer.count_tokens(normalized)
        if passages and used_tokens + cost > token_budget:
            break
        seen_text.add(normalized)
        used_tokens += cost
        passages.append(
            ContextPassage(
                index=index,
                chunk_id=e.source_id,
                document_id=e.document_id,
                filename=e.title,
                page=e.page,
                text=normalized,
                score=e.score,
                origin=e.origin,
                url=e.url,
            )
        )
        index += 1

    return AssembledContext(passages=passages)
