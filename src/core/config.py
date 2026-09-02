"""
Central place for all configuration. Every other module imports from here instead of
calling os.getenv() directly, so there's exactly one source of truth.

Paths are resolved against the project root (not the current working directory), so the
server and the CLI scripts behave identically no matter where you launch them from.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# src/core/config.py -> src/ -> project root
SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent

# The web UI FastAPI serves at "/" (see src/main.py).
STATIC_DIR = SRC_DIR / "static"

load_dotenv(PROJECT_ROOT / ".env")


def _path_setting(env_var: str, default: Path) -> Path:
    """Reads a path from the environment. Relative values resolve against the project root."""
    raw = os.getenv(env_var)
    if not raw:
        return default
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# ---------------------------------------------------------------- Mode
# One switch that picks the whole backend set. "local" is the laptop setup: models run on
# this CPU, vectors live on disk, PDFs live in data/. "cloud" is the deployable setup for
# a serverless host (Vercel), where there is no persistent disk, no room for torch, and no
# time to load a model per request: embedding and re-ranking become HTTP calls, vectors
# live in Chroma Cloud, PDFs live in Cloudinary.
#
# Everything below can still be overridden one piece at a time, so a half-and-half setup
# (cloud vectors, local models) is a matter of setting the individual variable.
RAG_MODE = os.getenv("RAG_MODE", "local").strip().lower()
if RAG_MODE not in ("local", "cloud"):
    raise ValueError(f"RAG_MODE must be 'local' or 'cloud', got {RAG_MODE!r}")
IS_CLOUD = RAG_MODE == "cloud"


def _mode_default(env_var: str, local: str, cloud: str) -> str:
    """A setting whose default follows RAG_MODE but can be pinned explicitly."""
    return (os.getenv(env_var) or (cloud if IS_CLOUD else local)).strip().lower()


# ---------------------------------------------------------------- LLM (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "800"))

# ---------------------------------------------------------------- Embeddings
# bge-small-en-v1.5 has a 512-token window (vs all-MiniLM-L6-v2's 256), so a ~300-word
# chunk fits without being silently truncated at embedding time.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# bge models are trained with an instruction prefix on the QUERY side only; passages are
# embedded bare. Set to "" if you switch to a model that doesn't want one (e.g. MiniLM).
#
# The trailing space is re-added here on purpose: python-dotenv strips trailing whitespace
# from unquoted .env values, which silently glued the prefix to the question
# ("...passages:What is A* search?") and degraded every single query embedding.
_raw_prefix = os.getenv(
    "EMBEDDING_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages:",
).strip()
EMBEDDING_QUERY_PREFIX = f"{_raw_prefix} " if _raw_prefix else ""
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# Who actually produces the vectors: "local" (sentence-transformers on this CPU), or
# "pinecone" / "cohere" / "jina" (an HTTP call). The API providers exist because torch + a model does not fit in
# a serverless bundle and would be re-loaded on every cold start if it did.
EMBEDDINGS_PROVIDER = _mode_default("EMBEDDINGS_PROVIDER", local="local", cloud="pinecone")

# Cohere's trial key is free and needs no card (1,000 calls/month at the time of writing),
# which is why it is the cloud default. embed-english-v3.0 is 1024-dimensional.
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
# Cohere caps a single /v2/embed call at 96 inputs.
COHERE_EMBED_BATCH = int(os.getenv("COHERE_EMBED_BATCH", "96"))

# Jina is the alternative free tier: a key gives a block of free tokens, no card.
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_EMBED_MODEL = os.getenv("JINA_EMBED_MODEL", "jina-embeddings-v3")
JINA_EMBED_BATCH = int(os.getenv("JINA_EMBED_BATCH", "64"))

# The token window used for chunk splitting when the provider is an API and there is no
# local tokenizer to ask. Both Cohere v3 and Jina v3 truncate at 8192 tokens, but chunks
# that big make re-ranking useless, so the effective limit stays near the local model's.
API_EMBED_TOKEN_LIMIT = int(os.getenv("API_EMBED_TOKEN_LIMIT", "512"))
# Characters per token, used only to estimate length without a tokenizer. ~4 is the usual
# English figure; it is deliberately conservative because over-estimating only splits a
# chunk earlier, while under-estimating silently truncates it.
CHARS_PER_TOKEN = float(os.getenv("CHARS_PER_TOKEN", "3.6"))

# Seconds to wait on any embedding/re-rank HTTP call before giving up.
PROVIDER_TIMEOUT_SECONDS = float(os.getenv("PROVIDER_TIMEOUT_SECONDS", "60"))
# Retries for a 429 or a 5xx. Free tiers rate-limit per minute, so one retry with a pause
# is the difference between an ingest finishing and an ingest failing halfway.
PROVIDER_MAX_RETRIES = int(os.getenv("PROVIDER_MAX_RETRIES", "3"))

# ---------------------------------------------------------------- Storage
DATA_DIR = _path_setting("DATA_DIR", PROJECT_ROOT / "data")
# Generated index state lives under storage/, kept out of the source tree and gitignored.
CHROMA_DIR = _path_setting("CHROMA_DIR", PROJECT_ROOT / "storage" / "chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_documents")

# Which store holds the vectors: "chroma" (development - a folder on disk) or "pinecone"
# (production). Serverless hosts have no persistent disk, so a folder-backed store there is
# empty again on every cold start.
#
# The two are not interchangeable for the same data: vectors written by one are not
# readable by the other, and the embedding models differ. Switching means re-indexing.
VECTOR_STORE = _mode_default("VECTOR_STORE", local="chroma", cloud="pinecone")

# Only for VECTOR_STORE=chroma: "disk" (a folder, single process, no network) or "cloud"
# (Chroma Cloud over HTTP), for running the development stack against a hosted store.
CHROMA_BACKEND = os.getenv("CHROMA_BACKEND", "disk").strip().lower()
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY", "")
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "")

# ---------------------------------------------------------------- Pinecone (production)
# One vendor for vectors, embeddings and re-ranking. The free Starter plan needs no card:
# 2GB, 5M embedding tokens/month per model, 500 rerank requests/month.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-documents")
# Where a new index is created, if it doesn't exist yet. The Starter plan is AWS-only.
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
# llama-text-embed-v2 is 1024-dimensional. If you change the model, change this too - an
# index is created with a fixed dimension and will reject vectors of any other size.
PINECONE_EMBED_MODEL = os.getenv("PINECONE_EMBED_MODEL", "llama-text-embed-v2")
PINECONE_EMBED_DIM = int(os.getenv("PINECONE_EMBED_DIM", "1024"))
# bge-reranker-v2-m3 is the model the free tier includes 500 monthly requests of.
PINECONE_RERANK_MODEL = os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3")
# Pinecone caps an upsert at 1000 vectors / ~2MB and an embed call at 96 inputs.
PINECONE_UPSERT_BATCH = int(os.getenv("PINECONE_UPSERT_BATCH", "100"))
PINECONE_EMBED_BATCH = int(os.getenv("PINECONE_EMBED_BATCH", "96"))
# Pin the API version explicitly: an unset version header defaults to the OLDEST supported
# one, which eventually disappears and takes the app with it.
PINECONE_API_VERSION = os.getenv("PINECONE_API_VERSION", "2025-10")
# Chroma enforces a max batch size per add() call; stay well under it.
CHROMA_ADD_BATCH = int(os.getenv("CHROMA_ADD_BATCH", "1000"))
# Small JSON sidecar tracking what's been ingested, so /stats and the re-ingest check
# don't have to read every chunk's metadata out of Chroma.
MANIFEST_PATH = CHROMA_DIR / "manifest.json"

# Uploads (POST /upload). Enforced while streaming, so an over-sized file is cut off and
# deleted rather than being written to disk first and measured afterwards.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024

# Total bytes one account may store. Without it, a single user can fill the disk 100MB at
# a time - and a full disk stops MongoDB and every other user, not just the culprit.
MAX_USER_STORAGE_BYTES = int(os.getenv("MAX_USER_STORAGE_MB", "2048")) * 1024 * 1024

# ---------------------------------------------------------------- Cloudinary (production PDF storage)
# Where PDF bytes live: "local" (DATA_DIR, this process's own disk) or "cloudinary"
# (uploaded straight from the browser, fetched back by URL for ingestion). Serverless hosts
# have no persistent disk, so "local" would lose every file the moment the function that
# wrote it exits.
DOCUMENT_STORE = _mode_default("DOCUMENT_STORE", local="local", cloud="cloudinary")
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")
# Prefix inside the Cloudinary account; each user gets "<folder>/users/<user_id>/..." below
# it, the same shape DATA_DIR uses locally, so ownership stays derivable from the path.
CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "rag-documents")

# ---------------------------------------------------------------- State store
# Where the rate limiter, the answer cache, the ingestion job, and the document manifest
# keep their state. "memory" is what this app has always done - a dict in this process,
# correct only because the app is single-worker (see ratelimit.py). "mongo" moves all four
# into MongoDB, which is what a serverless host needs: a cold start gets a fresh process
# with empty memory every time, but the same Mongo documents.
#
# Deliberately a SYNCHRONOUS (pymongo) client rather than the async one used for accounts:
# this state is read from both request handlers and the local background-ingestion thread,
# which has no event loop to await into, and a blocking call against one indexed document
# is a few milliseconds either way - this app has never claimed to serve concurrent load.
STATE_STORE = _mode_default("STATE_STORE", local="memory", cloud="mongo")

# ---------------------------------------------------------------- Auth (MongoDB + JWT)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "rag_app")
USERS_COLLECTION = os.getenv("USERS_COLLECTION", "users")
# Append-only record of logins, uploads and deletions. The first thing anyone asks for
# after an incident, and impossible to reconstruct after the fact.
AUDIT_COLLECTION = os.getenv("AUDIT_COLLECTION", "audit")

# Signing key for JWTs. Generated on first run and written to .env if absent (see
# src/services/security.py) - a key that changed on every restart would silently log
# everyone out, and a hard-coded default would let anyone mint a valid token.
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "12"))

# ---------------------------------------------------------------- Chunking
# 300 words is ~400 tokens for ordinary prose, but dense technical pages (formulae,
# hyphenated terms, tables) tokenise far worse - the ingest log has shown 616 tokens for a
# 300-word chunk, i.e. past bge-small's 512 window, and the tail of such a chunk is dropped
# at embedding time. Set CHUNK_SIZE_WORDS=220 in .env and re-run `python scripts/ingest.py
# --force` if warn_if_truncated() keeps firing on your corpus.
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "300"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))

# OCR fallback for scanned/image-only PDFs. Needs `pip install pytesseract pillow` AND the
# Tesseract binary installed on the machine; when either is missing, ingestion still works
# and such PDFs are simply reported as skipped.
OCR_ENABLED = os.getenv("OCR_ENABLED", "false").lower() in ("1", "true", "yes")
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_LANG = os.getenv("OCR_LANG", "eng")

# ---------------------------------------------------------------- Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))  # hard ceiling on what a client may request

# Retrieve a wide candidate set, then narrow it with the re-ranker. A bi-encoder is fast
# but coarse; the cross-encoder is the thing that decides what actually reaches the model.
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "30"))

# Hybrid search: BM25 keyword ranking fused with vector ranking (Reciprocal Rank Fusion).
# Dense vectors are weak on exact technical terms ("A* search", "Bayes decision rule");
# BM25 is strong there, and vice versa.
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "true").lower() in ("1", "true", "yes")
RRF_K = int(os.getenv("RRF_K", "60"))  # RRF damping constant; 60 is the published default

# Cross-encoder re-ranking. Set RERANK_ENABLED=false to A/B it against the eval harness.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# "local" runs the cross-encoder here; "pinecone" and "cohere" call a hosted re-ranker.
RERANKER_PROVIDER = _mode_default("RERANKER_PROVIDER", local="local", cloud="pinecone")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")

# Cross-encoder logits are unbounded; ms-marco models put clearly-irrelevant pairs well
# below zero. Chunks scoring under this are dropped, and if nothing survives we answer
# "not in these documents" WITHOUT calling the LLM.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-6.0"))
# Cohere Rerank returns a normalised 0..1 relevance, NOT a logit - the local floor of -6.0
# would keep everything. Hence a second floor, chosen per provider by
# reranker.score_floor(). 0.02 is roughly as permissive on that scale as -6.0 is on logits.
MIN_RERANK_SCORE_API = float(os.getenv("MIN_RERANK_SCORE_API", "0.02"))
# Fallback floor used when re-ranking is off (cosine similarity, 0..1).
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.15"))

# Context window expansion: also pull the chunks immediately before/after each hit, since
# the sentence that explains an answer often sits in the neighbouring chunk.
NEIGHBOR_EXPANSION = int(os.getenv("NEIGHBOR_EXPANSION", "1"))

# Cap on the assembled CONTEXT block, independent of top_k, so a large retrieval can
# never blow past the model's context window.
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))

# Answer cache: repeated questions skip retrieval and the paid LLM call entirely. Entries
# are per user, die when that user's documents change, and expire after the TTL.
ANSWER_CACHE_SIZE = int(os.getenv("ANSWER_CACHE_SIZE", "256"))
ANSWER_CACHE_TTL_SECONDS = int(os.getenv("ANSWER_CACHE_TTL_SECONDS", "3600"))
# Set false to measure the pipeline without cache hits confusing the numbers.
ANSWER_CACHE_ENABLED = os.getenv("ANSWER_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------- Conversation
# How many previous question/answer pairs to carry into the prompt.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "4"))
# Hard ceiling on the conversation carried into the prompt, independent of HISTORY_TURNS.
# The per-field limits in schemas.py stop one enormous turn; this stops several large ones
# adding up. CONTEXT has always had such a budget (MAX_CONTEXT_CHARS); history needs one
# for exactly the same reason.
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "8000"))
# Rewrite a follow-up ("what about the second one?") into a standalone question before
# retrieving - the raw follow-up embeds to nothing useful.
REWRITE_FOLLOWUPS = os.getenv("REWRITE_FOLLOWUPS", "true").lower() in ("1", "true", "yes")

# Origins allowed to call the API from a browser. The UI is served by this same process,
# so the default is "same-origin only" - a wildcard would let any site on the internet
# drive this API with a token it obtained by other means. Comma-separated; set it only if
# you deliberately serve the frontend from somewhere else.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

# ---------------------------------------------------------------- Misc
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# "text" for a human at a terminal, "json" for anything that ships logs somewhere.
LOG_FORMAT = os.getenv("LOG_FORMAT", "text").lower()
