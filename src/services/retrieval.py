"""
The retrieval pipeline: everything between a question and the chunks handed to the LLM.

    question
      -> vector search  (semantic, good at paraphrase)  ─┐
      -> BM25 search    (lexical, good at exact terms)  ─┴─> Reciprocal Rank Fusion
      -> cross-encoder re-rank (precision)
      -> relevance floor (drop weak matches; empty result = "not in these documents")
      -> neighbour expansion (pull chunk_index +/- 1 for context)
      -> chunks

Each stage can be turned off from config, which is what makes the eval harness able to
A/B them (`eval/run_eval.py --no-rerank --no-bm25`).
"""
from typing import Dict, List, Optional

from src.ml import reranker
from src.services import bm25, vectorstore
from src.core.config import (
    HYBRID_ENABLED,
    MIN_SIMILARITY,
    NEIGHBOR_EXPANSION,
    RERANK_ENABLED,
    RETRIEVAL_CANDIDATES,
    RRF_K,
)
from src.core.logging import get_logger, timed

log = get_logger(__name__)


def _key(chunk: Dict) -> tuple:
    """Identity of a chunk across the two ranked lists."""
    return (chunk.get("source"), chunk.get("chunk_index"))


def fuse(ranked_lists: List[List[Dict]], k: int = RRF_K) -> List[Dict]:
    """
    Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank).

    RRF combines rankings without needing their scores to be comparable - which matters
    here because cosine distance and BM25 scores are on entirely different scales. A chunk
    that both methods rank highly beats one that only a single method loves.
    """
    scores: Dict[tuple, float] = {}
    seen: Dict[tuple, Dict] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            key = _key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            # Keep the copy that carries a similarity score, if there is one.
            if key not in seen or ("similarity" in chunk and "similarity" not in seen[key]):
                seen[key] = chunk

    fused = []
    for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        chunk = dict(seen[key])
        chunk["fusion_score"] = round(score, 6)
        fused.append(chunk)
    return fused


def _expand_neighbors(chunks: List[Dict], radius: int,
                      user_id: Optional[str] = None) -> List[Dict]:
    """
    Adds each hit's neighbouring chunks, keeping documents in reading order.

    All the neighbours are fetched in one batched lookup per source rather than one query
    per hit - with top_k=8 that was eight separate round-trips to Chroma for a single
    question.
    """
    if radius < 1 or not chunks:
        return chunks

    seen = {_key(c) for c in chunks}

    wanted: Dict[str, set] = {}
    for chunk in chunks:
        index = chunk.get("chunk_index")
        if index is None or not chunk.get("source"):
            continue
        bucket = wanted.setdefault(chunk["source"], set())
        for offset in range(-radius, radius + 1):
            if offset and index + offset >= 0:
                bucket.add(index + offset)

    try:
        found = vectorstore.get_neighbors_bulk(wanted, user_id=user_id) if wanted else {}
    except Exception:
        # Neighbour expansion is extra context around a hit that already cleared the
        # relevance floor, not the hit itself - losing it to a transient store error (a
        # Pinecone read timeout, say) should degrade the answer, not fail the question. Same
        # trade as a keyword-index build failure: worse context beats no answer.
        log.warning("Neighbour expansion failed; answering from the hits alone.", exc_info=True)
        found = {}

    out: List[Dict] = []
    for chunk in chunks:
        group = [chunk]
        index = chunk.get("chunk_index")
        if index is not None:
            for offset in range(-radius, radius + 1):
                if not offset:
                    continue
                neighbor = found.get((chunk.get("source"), index + offset))
                if neighbor is None or _key(neighbor) in seen:
                    continue
                seen.add(_key(neighbor))
                neighbor = dict(neighbor, neighbor_of=index)
                group.append(neighbor)
        # Present each hit with its neighbours in document order, so the model reads
        # continuous prose rather than a shuffled fragment.
        group.sort(key=lambda c: (c.get("chunk_index") is None, c.get("chunk_index", 0)))
        out.extend(group)
    return out


def retrieve(
    question: str,
    top_k: int = 4,
    source=None,
    *,
    user_id: Optional[str] = None,
    use_rerank: Optional[bool] = None,
    use_hybrid: Optional[bool] = None,
    expand: Optional[int] = None,
) -> List[Dict]:
    """
    Returns the chunks to answer `question` from, best first, or [] when nothing clears
    the relevance floor (the caller should then say so instead of calling the LLM).

    `source` narrows the search to one document name or a list of them; None searches the
    whole library.

    `user_id` scopes retrieval to one person's documents and MUST be passed on every
    request path. It is threaded into all three stages that can reach stored text - the
    vector query, the BM25 ranking, and neighbour expansion - and asserted again on the
    way out. Only offline callers with no user (the CLI, eval/run_eval.py) may omit it.

    The remaining keyword arguments exist so the eval harness can measure each stage's
    contribution.
    """
    use_rerank = RERANK_ENABLED if use_rerank is None else use_rerank
    use_hybrid = HYBRID_ENABLED if use_hybrid is None else use_hybrid
    expand = NEIGHBOR_EXPANSION if expand is None else expand

    # Asking for more results than candidates were fetched would silently return fewer
    # than top_k - the candidate pool has to be at least as large as what we keep.
    candidate_count = max(RETRIEVAL_CANDIDATES, top_k)

    with timed(log, "retrieval"):
        dense = vectorstore.query_chunks(question, top_k=candidate_count, source=source,
                                         user_id=user_id)
        if not dense and not use_hybrid:
            return []

        lists = [dense]
        if use_hybrid:
            lexical = [chunk for chunk, _score in
                       bm25.search(question, limit=candidate_count, source=source,
                                   user_id=user_id)]
            if lexical:
                lists.append(lexical)
            log.debug("candidates: %d dense, %d lexical", len(dense), len(lexical))

        candidates = fuse(lists) if len(lists) > 1 else dense
        if not candidates:
            return []

        if use_rerank:
            ranked = reranker.rerank(question, candidates)
            if ranked is not None:
                # The floor comes from the re-ranker, not from config directly: a local
                # cross-encoder scores in unbounded logits and Cohere Rerank scores 0..1,
                # so one hard-coded constant would be wrong for one of them.
                floor = reranker.score_floor()
                kept = [dict(chunk, rerank_score=round(score, 3))
                        for chunk, score in ranked if score >= floor]
                if not kept:
                    log.info(
                        "No candidate cleared the %s re-ranker's floor of %s (best was "
                        "%.2f) - treating the question as unanswerable from these documents.",
                        reranker.provider_name(), floor, ranked[0][1],
                    )
                    return []
                candidates = kept
        else:
            # Without the cross-encoder, fall back to a cosine floor on the dense hits.
            floored = [c for c in candidates if c.get("similarity", 1.0) >= MIN_SIMILARITY]
            if not floored:
                log.info("Nothing cleared MIN_SIMILARITY=%s - unanswerable.", MIN_SIMILARITY)
                return []
            candidates = floored

        hits = candidates[:top_k]
        expanded = _expand_neighbors(hits, expand, user_id=user_id)

        # Defence in depth. Every stage above already filters by owner; this catches the
        # case where a future stage forgets to. It is cheap (a list comprehension over at
        # most a few dozen dicts) and it turns a silent data leak into a loud log line.
        if user_id:
            clean = [c for c in expanded if c.get("user_id") == user_id]
            if len(clean) != len(expanded):
                log.error(
                    "Retrieval returned %d chunk(s) not owned by %s - dropped. This is a "
                    "bug in an isolation filter, not a normal condition.",
                    len(expanded) - len(clean), user_id,
                )
            return clean
        return expanded
