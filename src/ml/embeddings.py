"""
Step 2 of the pipeline: turn text into vectors.

There are two ways to do that, chosen by EMBEDDINGS_PROVIDER (which defaults from
RAG_MODE):

* **local** - sentence-transformers on this CPU. Free, private, and what development runs
  on. Loading the model takes a few seconds, so it is a module-level singleton, loaded
  LAZILY on first use. Why lazy matters: uvicorn imports the app before it binds the
  socket, so anything done at import time happens while the port is still closed and the
  browser answers ERR_CONNECTION_REFUSED. Importing torch and loading the model at import
  cost ~18s of that. Now the port opens in a couple of seconds and the model loads in a
  warm-up thread behind the loading screen (see warm_up(), called from src/main.py).
* **pinecone / cohere / jina** - an HTTP call (src/ml/providers.py). Used in cloud mode,
  where torch does not fit in the bundle and a per-cold-start model load would be paid on
  every request. Pinecone is the production default because the vectors live there too, so
  it is one vendor and one key rather than three.

The rest of the codebase calls embed_passages/embed_query and never learns which one is
in use.

Two details that matter for retrieval quality either way:

* **Truncation.** Every embedding model has a hard token window and silently truncates
  anything longer - no error, no warning, the tail of the chunk simply never influences
  whether that chunk gets retrieved. split_to_token_limit() prevents that; with an API
  provider the token count is an estimate (see providers.estimate_tokens).
* **Query/passage asymmetry.** bge-family models want an instruction prefix on the QUERY
  side only (EMBEDDING_QUERY_PREFIX). The API providers express the same idea with an
  input_type/task field instead, so the prefix is NOT applied to them - doing both would
  put a bge instruction into a Cohere query for no reason.
"""
import threading
from typing import List, Optional

from src.core.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PREFIX,
    EMBEDDINGS_PROVIDER,
)
from src.core.logging import get_logger, timed

log = get_logger(__name__)


class LocalEmbeddings:
    """sentence-transformers, in this process."""

    name = "local"

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        """
        The model singleton. Double-checked locking, because the ingestion thread, the
        warm-up thread and a request handler can all reach for it at once and the loader is
        not something to run twice.
        """
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                # Imported here, not at module level: `import sentence_transformers` pulls
                # in torch, which is most of the delay before uvicorn can accept
                # connections.
                from sentence_transformers import SentenceTransformer

                log.info(
                    "Loading embedding model '%s' (first run downloads it, then it's cached)...",
                    EMBEDDING_MODEL,
                )
                with timed(log, "load embedding model"):
                    self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def warm_up(self) -> None:
        self._get_model()

    def is_ready(self) -> bool:
        return self._model is not None

    def max_input_tokens(self) -> int:
        return int(getattr(self._get_model(), "max_seq_length", 0) or 0)

    def count_tokens(self, text: str) -> int:
        tokenizer = getattr(self._get_model(), "tokenizer", None)
        if tokenizer is None:  # e.g. the stub model used by the offline test suite
            return len(text.split())
        return len(tokenizer(text)["input_ids"])

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._get_model().encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self._get_model().encode(
            [EMBEDDING_QUERY_PREFIX + text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0].tolist()


_provider_instance = None
_provider_lock = threading.Lock()


def _provider():
    """The configured provider, built once."""
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance
    with _provider_lock:
        if _provider_instance is None:
            if EMBEDDINGS_PROVIDER == "local":
                _provider_instance = LocalEmbeddings()
            elif EMBEDDINGS_PROVIDER == "pinecone":
                from src.ml.providers import PineconeEmbeddings
                _provider_instance = PineconeEmbeddings()
            elif EMBEDDINGS_PROVIDER == "cohere":
                from src.ml.providers import CohereEmbeddings
                _provider_instance = CohereEmbeddings()
            elif EMBEDDINGS_PROVIDER == "jina":
                from src.ml.providers import JinaEmbeddings
                _provider_instance = JinaEmbeddings()
            else:
                raise ValueError(
                    f"EMBEDDINGS_PROVIDER must be local, pinecone, cohere or jina - got "
                    f"{EMBEDDINGS_PROVIDER!r}"
                )
            log.info("Embeddings provider: %s", _provider_instance.name)
    return _provider_instance


def provider_name() -> str:
    """Which backend is in use. Reported on /info; also what the tests assert on."""
    return _provider().name


def reset_provider() -> None:
    """Drop the cached provider. Only for tests that flip the configuration."""
    global _provider_instance
    with _provider_lock:
        _provider_instance = None


def warm_up() -> None:
    """Load the model ahead of the first request. Safe to call from a background thread."""
    try:
        _provider().warm_up()
    except Exception:
        # A failed warm-up must not kill the server; the next real call retries and
        # surfaces the error to whoever asked.
        log.exception("Embedding warm-up failed; it will be retried on first use.")


def is_ready() -> bool:
    """True once embeddings can be produced - the app serves pages before this is true."""
    try:
        return _provider().is_ready()
    except Exception:
        return False


def max_input_tokens() -> int:
    """The model's hard token window. Text longer than this is silently truncated."""
    return _provider().max_input_tokens()


def count_tokens(text: str) -> int:
    return _provider().count_tokens(text)


def embed_passages(texts: List[str]) -> List[List[float]]:
    """Embed document chunks (ingestion side). Batched, so a big book doesn't spike memory."""
    return _provider().embed_passages(texts)


def embed_query(text: str) -> List[float]:
    """Embed a user question (retrieval side), with whatever query marking the model wants."""
    return _provider().embed_query(text)


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


def split_to_token_limit(chunks: List[dict], limit: Optional[int] = None) -> List[dict]:
    """
    Splits any chunk that would be truncated at embedding time, in place of warning about it.

    The chunker packs by WORDS, but the model reads TOKENS, and the ratio is not a constant:
    ordinary prose runs ~1.3 tokens per word, while dense technical pages - formulae,
    hyphenated terms, tables - have been measured at 616 tokens for a 300-word chunk here,
    past bge-small's 512 window. Everything past the window is silently dropped, so the tail
    of such a chunk never influences retrieval at all.

    Splitting is done on the chunk's own words, in halves, until each piece fits. Page
    attribution is inherited: a split piece belongs to the same page range as its parent,
    which is a slight over-estimate at the boundary and the same approximation the chunker
    already makes.
    """
    limit = limit or max_input_tokens()
    if not limit or not chunks:
        return chunks

    # Leave headroom: some models add special tokens to every input.
    budget = max(64, int(limit * 0.95))
    out: List[dict] = []
    split_count = 0

    for chunk in chunks:
        text = chunk["text"]
        if count_tokens(text) <= budget:
            out.append(chunk)
            continue

        # Halve until each piece fits. Iterative rather than recursive so a pathological
        # chunk cannot blow the stack.
        pending = [text]
        while pending:
            piece = pending.pop(0)
            words = piece.split()
            if count_tokens(piece) <= budget or len(words) < 2:
                out.append(dict(chunk, text=piece))
                continue
            middle = len(words) // 2
            pending.insert(0, " ".join(words[middle:]))
            pending.insert(0, " ".join(words[:middle]))
            split_count += 1

    if split_count:
        log.info("Split %d oversized chunk(s) to fit the model's %d-token window.",
                 split_count, limit)
    return out
