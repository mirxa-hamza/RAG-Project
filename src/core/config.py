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

# ---------------------------------------------------------------- Storage
DATA_DIR = _path_setting("DATA_DIR", PROJECT_ROOT / "data")
# Generated index state lives under storage/, kept out of the source tree and gitignored.
CHROMA_DIR = _path_setting("CHROMA_DIR", PROJECT_ROOT / "storage" / "chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_documents")
# Chroma enforces a max batch size per add() call; stay well under it.
CHROMA_ADD_BATCH = int(os.getenv("CHROMA_ADD_BATCH", "1000"))
# Small JSON sidecar tracking what's been ingested, so /stats and the re-ingest check
# don't have to read every chunk's metadata out of Chroma.
MANIFEST_PATH = CHROMA_DIR / "manifest.json"

# Uploads (POST /upload). Enforced while streaming, so an over-sized file is cut off and
# deleted rather than being written to disk first and measured afterwards.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024

# ---------------------------------------------------------------- Auth (MongoDB + JWT)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "rag_app")
USERS_COLLECTION = os.getenv("USERS_COLLECTION", "users")

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
# Cross-encoder logits are unbounded; ms-marco models put clearly-irrelevant pairs well
# below zero. Chunks scoring under this are dropped, and if nothing survives we answer
# "not in these documents" WITHOUT calling the LLM.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-6.0"))
# Fallback floor used when re-ranking is off (cosine similarity, 0..1).
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.15"))

# Context window expansion: also pull the chunks immediately before/after each hit, since
# the sentence that explains an answer often sits in the neighbouring chunk.
NEIGHBOR_EXPANSION = int(os.getenv("NEIGHBOR_EXPANSION", "1"))

# Cap on the assembled CONTEXT block, independent of top_k, so a large retrieval can
# never blow past the model's context window.
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))

# ---------------------------------------------------------------- Conversation
# How many previous question/answer pairs to carry into the prompt.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "4"))
# Rewrite a follow-up ("what about the second one?") into a standalone question before
# retrieving - the raw follow-up embeds to nothing useful.
REWRITE_FOLLOWUPS = os.getenv("REWRITE_FOLLOWUPS", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------- Misc
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
