"""Context assembly: rank, dedupe, and token-budget-pack retrieved passages.

Turns raw vector-store hits into a bounded, citable context block. Each kept
passage gets a 1-based citation index that the model is asked to cite and that
verification (Phase 4) validates against.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.services.ingestion import tokenizer
from app.services.vector_store import SearchResult


class ContextPassage(BaseModel):
    index: int  # 1-based citation number
    chunk_id: str
    document_id: str | None
    filename: str
    page: int | None
    text: str
    score: float
    origin: str = "vector"  # "vector" | "web"
    url: str | None = None

    def citation_label(self) -> str:
        if self.page is not None:
            return f"{self.filename}, p.{self.page}"
        return self.filename


class AssembledContext(BaseModel):
    passages: list[ContextPassage]

    @property
    def is_empty(self) -> bool:
        return not self.passages

    def to_prompt_block(self) -> str:
        """Render numbered sources for injection into the answer prompt."""
        blocks = [
            f"[{p.index}] ({p.citation_label()})\n{p.text}" for p in self.passages
        ]
        return "\n\n".join(blocks)


def assemble_context(
    results: list[SearchResult],
    *,
    token_budget: int,
    max_passages: int | None = None,
) -> AssembledContext:
    """Dedupe, rank by score, and pack passages under a token budget."""
    # Dedupe by chunk id (keep highest score) then by exact text.
    best_by_chunk: dict[str, SearchResult] = {}
    for r in results:
        current = best_by_chunk.get(r.chunk_id)
        if current is None or r.score > current.score:
            best_by_chunk[r.chunk_id] = r

    ranked = sorted(best_by_chunk.values(), key=lambda r: r.score, reverse=True)

    passages: list[ContextPassage] = []
    seen_text: set[str] = set()
    used_tokens = 0
    index = 1
    for r in ranked:
        if max_passages is not None and len(passages) >= max_passages:
            break
        normalized = r.text.strip()
        if normalized in seen_text:
            continue
        cost = tokenizer.count_tokens(normalized)
        if passages and used_tokens + cost > token_budget:
            # Budget exhausted; stop (passages are score-ordered).
            break
        seen_text.add(normalized)
        used_tokens += cost
        passages.append(
            ContextPassage(
                index=index,
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                filename=r.filename,
                page=r.page,
                text=normalized,
                score=r.score,
            )
        )
        index += 1

    return AssembledContext(passages=passages)
