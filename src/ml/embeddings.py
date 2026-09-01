"""
Step 2 of the pipeline: turn text into vectors.

Loading a sentence-transformers model takes a few seconds, so it's loaded once
(module-level singleton) rather than per request - but LAZILY, on first use.

Why lazy matters: uvicorn imports the app before it binds the socket, so anything done at
import time happens while the port is still closed and the browser answers
ERR_CONNECTION_REFUSED. Importing torch and loading the model at import cost ~18s of that.
Now the port opens in a couple of seconds and the model loads in a warm-up thread behind
the loading screen (see warm_up(), called from the lifespan in src/main.py).

Two details that matter for retrieval quality:

* **Truncation.** Every embedding model has a hard token window and silently truncates
  anything longer - no error, no warning, the tail of the chunk simply never influences
  whether that chunk gets retrieved. `warn_if_truncated()` makes that visible instead.
* **Query prefix.** bge/e5-family models are trained with an instruction prefix on the
  query side only. Embedding queries bare throws away most of the model's advantage over
  a plain MiniLM. Configured via EMBEDDING_QUERY_PREFIX (set it to "" for MiniLM).
"""
import threading
from typing import List

from src.core.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, EMBEDDING_QUERY_PREFIX
from src.core.logging import get_logger, timed

log = get_logger(__name__)

_model = None
_lock = threading.Lock()


def _get_model():
    """
    The model singleton. Double-checked locking, because the ingestion thread, the warm-up
    thread and a request handler can all reach for it at once and the loader is not
    something to run twice.
    """
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            # Imported here, not at module level: `import sentence_transformers` pulls in
            # torch, which is most of the delay before uvicorn can accept connections.
            from sentence_transformers import SentenceTransformer

            log.info(
                "Loading embedding model '%s' (first run downloads it, then it's cached)...",
                EMBEDDING_MODEL,
            )
            with timed(log, "load embedding model"):
                _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def warm_up() -> None:
    """Load the model ahead of the first request. Safe to call from a background thread."""
    try:
        _get_model()
    except Exception:
        # A failed warm-up must not kill the server; the next real call retries and
        # surfaces the error to whoever asked.
        log.exception("Embedding model warm-up failed; it will be retried on first use.")


def is_ready() -> bool:
    """True once the model is loaded - the app can serve pages before this is true."""
    return _model is not None


def max_input_tokens() -> int:
    """The model's hard token window. Text longer than this is silently truncated."""
    return int(getattr(_get_model(), "max_seq_length", 0) or 0)


def count_tokens(text: str) -> int:
    tokenizer = getattr(_get_model(), "tokenizer", None)
    if tokenizer is None:  # e.g. the stub model used by the offline test suite
        return len(text.split())
    return len(tokenizer(text)["input_ids"])


def warn_if_truncated(texts: List[str], sample: int = 25) -> int:
    """
    Checks a sample of chunks against the model's token window and logs a warning if any
    would be truncated. Returns how many of the sampled chunks were over the limit.

    This is the guard for the bug that motivated it: 300-word chunks fed to a 256-token
    model meant roughly the last third of every chunk was never embedded.
    """
    limit = max_input_tokens()
    if not limit or not texts:
        return 0

    step = max(1, len(texts) // sample)
    sampled = texts[::step][:sample]
    over = [n for n in (count_tokens(t) for t in sampled) if n > limit]
    if over:
        log.warning(
            "%d/%d sampled chunks exceed the embedding model's %d-token window "
            "(largest sampled: %d tokens). Their tails are being silently truncated - "
            "lower CHUNK_SIZE_WORDS or switch to a model with a longer window.",
            len(over), len(sampled), limit, max(over),
        )
    return len(over)


def embed_passages(texts: List[str]) -> List[List[float]]:
    """Embed document chunks (ingestion side). Batched, so a big book doesn't spike memory."""
    embeddings = _get_model().encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=len(texts) > 200,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(text: str) -> List[float]:
    """Embed a user question (retrieval side), with the model's query prefix applied."""
    embedding = _get_model().encode(
        [EMBEDDING_QUERY_PREFIX + text],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding[0].tolist()
