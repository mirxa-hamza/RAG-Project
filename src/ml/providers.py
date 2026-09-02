"""
HTTP embedding and re-ranking providers, used when RAG_MODE=cloud.

Why these exist at all: the local path (sentence-transformers) needs torch, a ~130MB model
file and a few seconds to load it. That is fine on a laptop and impossible on a serverless
host - the bundle does not fit, and even if it did, every cold start would pay the load
again before answering a single question. So in cloud mode the same two stages become HTTP
calls to a provider that keeps the model warm.

Everything here is deliberately plain `httpx` rather than each vendor's SDK: two POST
bodies are not worth two more dependencies to keep pinned, and stubbing one transport in
the tests is easier than stubbing two SDKs.

The interface each embedding provider exposes matches what src/ml/embeddings.py needs:

    embed_passages(texts) -> list[list[float]]
    embed_query(text)     -> list[float]
    max_input_tokens()    -> int
    count_tokens(text)    -> int

Free-tier reality, since that is the reason this file is written the way it is: request
limits are per minute and calls are counted per REQUEST, not per input. So passages are
sent in the largest batch the vendor accepts, and a 429 is retried with a pause rather than
being allowed to fail an ingest halfway through a book.
"""
import math
import time
from typing import Dict, List, Optional, Tuple

from src.core.config import (
    API_EMBED_TOKEN_LIMIT,
    CHARS_PER_TOKEN,
    COHERE_API_KEY,
    COHERE_EMBED_BATCH,
    COHERE_EMBED_MODEL,
    COHERE_RERANK_MODEL,
    JINA_API_KEY,
    JINA_EMBED_BATCH,
    JINA_EMBED_MODEL,
    PINECONE_API_KEY,
    PINECONE_API_VERSION,
    PINECONE_EMBED_BATCH,
    PINECONE_EMBED_MODEL,
    PINECONE_RERANK_MODEL,
    PROVIDER_MAX_RETRIES,
    PROVIDER_TIMEOUT_SECONDS,
)
from src.core.logging import get_logger, timed

log = get_logger(__name__)

COHERE_BASE = "https://api.cohere.com"
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"


class ProviderError(RuntimeError):
    """A provider call failed in a way retrying will not fix (bad key, bad model name)."""


def estimate_tokens(text: str) -> int:
    """
    Token count without a tokenizer.

    Used only to decide whether a chunk needs splitting before it is embedded. It rounds
    UP: over-estimating splits a chunk slightly earlier than necessary, which costs one
    extra vector, while under-estimating lets the provider truncate the tail silently -
    the exact failure this whole path exists to avoid.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / max(1.0, CHARS_PER_TOKEN)))


def _post(url: str, payload: dict, api_key: str, *, what: str,
          headers: Optional[Dict[str, str]] = None) -> dict:
    """
    One POST with retries on the failures that are worth retrying.

    429 (free-tier rate limit) and 5xx get backed off and tried again; 401/400 do not,
    because a wrong key or a wrong model name will still be wrong in four seconds.

    `headers` overrides the default bearer-token auth - Pinecone wants the key in its own
    `Api-Key` header and a pinned API version alongside it.
    """
    import httpx  # local import: keeps it off the import path of the local-only setup

    if not api_key:
        raise ProviderError(f"{what}: no API key set - see .env")

    headers = headers or {"Authorization": f"Bearer {api_key}",
                          "Content-Type": "application/json"}
    last_exc: Optional[Exception] = None

    for attempt in range(1, PROVIDER_MAX_RETRIES + 1):
        try:
            response = httpx.post(url, json=payload, headers=headers,
                                  timeout=PROVIDER_TIMEOUT_SECONDS)
        except httpx.HTTPError as exc:  # connection reset, timeout, DNS
            last_exc = exc
        else:
            if response.status_code < 300:
                return response.json()
            body = response.text[:300]
            if response.status_code in (400, 401, 403, 404, 422):
                raise ProviderError(f"{what}: HTTP {response.status_code} - {body}")
            last_exc = ProviderError(f"{what}: HTTP {response.status_code} - {body}")

        if attempt < PROVIDER_MAX_RETRIES:
            # Free tiers limit per MINUTE, so the pause has to be seconds, not milliseconds.
            pause = min(30.0, 2.0 ** attempt)
            log.warning("%s failed (%s); retrying in %.0fs (attempt %d/%d).",
                        what, last_exc, pause, attempt + 1, PROVIDER_MAX_RETRIES)
            time.sleep(pause)

    raise ProviderError(f"{what} failed after {PROVIDER_MAX_RETRIES} attempts: {last_exc}")


class _ApiEmbeddings:
    """Shared behaviour for the HTTP embedding providers."""

    name = "api"
    batch_size = 32

    def _embed(self, texts: List[str], *, is_query: bool) -> List[List[float]]:
        raise NotImplementedError

    # -- interface used by src/ml/embeddings.py ----------------------------------------
    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            with timed(log, f"{self.name} embed {len(batch)} passages"):
                out.extend(self._embed(batch, is_query=False))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], is_query=True)[0]

    def max_input_tokens(self) -> int:
        return API_EMBED_TOKEN_LIMIT

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def warm_up(self) -> None:
        """Nothing to load - the model lives on the provider's machines."""

    def is_ready(self) -> bool:
        return True


class CohereEmbeddings(_ApiEmbeddings):
    """
    Cohere /v2/embed.

    input_type is not optional and not cosmetic: v3 models embed a passage and a question
    into deliberately different spaces ("search_document" vs "search_query"), and getting
    it backwards degrades every retrieval without erroring. It replaces the bge query
    prefix, which is why embeddings.py skips EMBEDDING_QUERY_PREFIX for this provider.
    """

    name = "cohere"

    def __init__(self) -> None:
        self.batch_size = COHERE_EMBED_BATCH

    def _embed(self, texts: List[str], *, is_query: bool) -> List[List[float]]:
        payload = {
            "model": COHERE_EMBED_MODEL,
            "texts": texts,
            "input_type": "search_query" if is_query else "search_document",
            "embedding_types": ["float"],
        }
        data = _post(f"{COHERE_BASE}/v2/embed", payload, COHERE_API_KEY, what="Cohere embed")
        vectors = (data.get("embeddings") or {}).get("float")
        if not vectors or len(vectors) != len(texts):
            raise ProviderError(f"Cohere embed returned {len(vectors or [])} vectors "
                                f"for {len(texts)} inputs")
        return vectors


class JinaEmbeddings(_ApiEmbeddings):
    """
    Jina /v1/embeddings - the alternative free tier.

    Same asymmetry as Cohere, spelled `task`: retrieval.passage / retrieval.query.
    """

    name = "jina"

    def __init__(self) -> None:
        self.batch_size = JINA_EMBED_BATCH

    def _embed(self, texts: List[str], *, is_query: bool) -> List[List[float]]:
        payload = {
            "model": JINA_EMBED_MODEL,
            "task": "retrieval.query" if is_query else "retrieval.passage",
            "input": texts,
        }
        data = _post(JINA_EMBED_URL, payload, JINA_API_KEY, what="Jina embed")
        rows = data.get("data") or []
        if len(rows) != len(texts):
            raise ProviderError(f"Jina embed returned {len(rows)} vectors "
                                f"for {len(texts)} inputs")
        # The API does not promise the rows come back in order; it does give each an index.
        rows = sorted(rows, key=lambda r: r.get("index", 0))
        return [r["embedding"] for r in rows]


PINECONE_BASE = "https://api.pinecone.io"


def _pinecone_headers() -> Dict[str, str]:
    """
    Pinecone's own auth header, plus a pinned API version.

    The version matters: an omitted X-Pinecone-Api-Version defaults to the OLDEST supported
    version, so the app would silently move to a different API the day that one is retired.
    """
    return {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Pinecone-Api-Version": PINECONE_API_VERSION,
    }


class PineconeEmbeddings(_ApiEmbeddings):
    """
    Pinecone /embed - the production embedding path.

    Same passage/query asymmetry as the others, spelled `input_type`: "passage" for stored
    chunks, "query" for questions. truncate=END is explicit rather than default so an
    oversized chunk is cut instead of failing the whole batch - split_to_token_limit()
    should have prevented that already, and this is the backstop.
    """

    name = "pinecone"

    def __init__(self) -> None:
        self.batch_size = PINECONE_EMBED_BATCH

    def _embed(self, texts: List[str], *, is_query: bool) -> List[List[float]]:
        payload = {
            "model": PINECONE_EMBED_MODEL,
            "inputs": [{"text": t} for t in texts],
            "parameters": {
                "input_type": "query" if is_query else "passage",
                "truncate": "END",
            },
        }
        data = _post(f"{PINECONE_BASE}/embed", payload, PINECONE_API_KEY,
                     what="Pinecone embed", headers=_pinecone_headers())
        rows = data.get("data") or []
        if len(rows) != len(texts):
            raise ProviderError(f"Pinecone embed returned {len(rows)} vectors "
                                f"for {len(texts)} inputs")
        return [r["values"] for r in rows]


class PineconeReranker:
    """
    Pinecone /rerank.

    Scores are 0..1 relevance like Cohere's, not cross-encoder logits, so the caller
    compares against MIN_RERANK_SCORE_API. The free tier includes 500 requests a month of
    bge-reranker-v2-m3 - one per question asked, so it is the tightest quota in the system.
    """

    name = "pinecone"

    def available(self) -> bool:
        return bool(PINECONE_API_KEY)

    def rerank(self, question: str, chunks: List[Dict]) -> Optional[List[Tuple[Dict, float]]]:
        if not chunks:
            return None
        if not PINECONE_API_KEY:
            log.warning("RERANKER_PROVIDER=pinecone but PINECONE_API_KEY is unset - "
                        "falling back to fused ranking.")
            return None

        payload = {
            "model": PINECONE_RERANK_MODEL,
            "query": question,
            # Documents are addressed by position on the way back (`index`), so the id here
            # is only for readability in a request log.
            "documents": [{"id": str(i), "text": c["text"]} for i, c in enumerate(chunks)],
            "top_n": len(chunks),
            "return_documents": False,
            "rank_fields": ["text"],
            "parameters": {"truncate": "END"},
        }
        try:
            with timed(log, f"pinecone re-rank {len(chunks)} candidates"):
                data = _post(f"{PINECONE_BASE}/rerank", payload, PINECONE_API_KEY,
                             what="Pinecone rerank", headers=_pinecone_headers())
        except ProviderError as exc:
            # Fail OPEN: a spent monthly quota should cost ranking quality, not the answer.
            log.warning("Pinecone rerank unavailable (%s) - using fused ranking.", exc)
            return None

        ranked: List[Tuple[Dict, float]] = []
        for row in data.get("data") or []:
            index = row.get("index")
            if index is None or not (0 <= index < len(chunks)):
                continue
            ranked.append((chunks[index], float(row.get("score", 0.0))))
        if not ranked:
            return None
        ranked.sort(key=lambda p: p[1], reverse=True)
        return ranked


class CohereReranker:
    """
    Cohere /v2/rerank.

    Returns 0..1 relevance, not a cross-encoder logit, so the caller has to compare against
    MIN_RERANK_SCORE_API rather than MIN_RERANK_SCORE. Mixing the two floors up is the one
    mistake here that fails quietly: -6.0 against a 0..1 score keeps every candidate and
    turns the "not in these documents" answer off entirely.
    """

    name = "cohere"

    def available(self) -> bool:
        return bool(COHERE_API_KEY)

    def rerank(self, question: str, chunks: List[Dict]) -> Optional[List[Tuple[Dict, float]]]:
        if not chunks:
            return None
        if not COHERE_API_KEY:
            log.warning("RERANKER_PROVIDER=cohere but COHERE_API_KEY is unset - "
                        "falling back to fused ranking.")
            return None

        payload = {
            "model": COHERE_RERANK_MODEL,
            "query": question,
            "documents": [c["text"] for c in chunks],
            "top_n": len(chunks),
        }
        try:
            with timed(log, f"cohere re-rank {len(chunks)} candidates"):
                data = _post(f"{COHERE_BASE}/v2/rerank", payload, COHERE_API_KEY,
                             what="Cohere rerank")
        except ProviderError as exc:
            # Fail OPEN, exactly like the local re-ranker: a rate-limited free tier should
            # give slightly worse ranking, not a failed question.
            log.warning("Cohere rerank unavailable (%s) - using fused ranking.", exc)
            return None

        ranked: List[Tuple[Dict, float]] = []
        for row in data.get("results") or []:
            index = row.get("index")
            if index is None or not (0 <= index < len(chunks)):
                continue
            ranked.append((chunks[index], float(row.get("relevance_score", 0.0))))
        if not ranked:
            return None
        ranked.sort(key=lambda p: p[1], reverse=True)
        return ranked
