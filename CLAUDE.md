# CLAUDE.md

Guidance for Claude (or any AI assistant) working in this repository.

## What this project is

A from-scratch Retrieval-Augmented Generation (RAG) app: drop PDFs in the backend's `data/`
folder, ask questions about them, get answers grounded only in that content. Built
deliberately without LangChain/LlamaIndex so every pipeline step is plain, readable
Python — this is a learning project first, a working app second.

**Documents come only from `data/` on the backend's own filesystem.** There is no upload
endpoint and never should be — a user talking to the frontend can ask questions but can
never add, replace, or remove what's in the vector store. If asked to "add a way to upload
files from the browser," confirm explicitly first — that's a deliberate, stated design
constraint, not an oversight.

Flow: PDF → `PyMuPDF` extraction (paragraphs preserved) → structure-aware chunking →
local embeddings (`sentence-transformers`) → ChromaDB (persistent, on-disk) → cosine
retrieval → Groq LLM answers strictly from retrieved chunks.

## Layout

```
backend/
  app/
    main.py            FastAPI app: /ingest, /ingest/status, /chat, /stats, /reset, /health
    config.py          every setting; paths resolve against PROJECT_ROOT, not the CWD
    ingest.py          single entry point for documents + background job state machine
    manifest.py        JSON sidecar: what's ingested, with content hashes
    pdf_utils.py       extraction (keeps paragraph breaks) + paragraph-packing chunker
    embeddings.py      sentence-transformers singleton, query prefix, truncation guard
    vectorstore.py     ChromaDB add/query/delete/reset, batched
    llm.py             grounded prompt + Groq call
    logging_setup.py   logging config + `timed()` context manager
  scripts/
    ingest.py          CLI index builder (--force, --status)
    make_test_pdf.py   fixture generator -> backend/test_fixtures/, never data/
  tests/
    test_pipeline_offline.py   37 checks, offline, no Groq key needed
  requirements.txt / requirements-dev.txt / .env.example / .env
frontend/
  index.html / style.css / script.js   talks to http://localhost:8000 (API_BASE in script.js)
data/
  real PDFs the user drops here — the live ingestion source, gitignored
refrence/
  third-party reference project kept locally for study; gitignored, not part of this codebase
```

## Conventions

- **Every module imports settings from `app.config`**, never `os.getenv()` directly. New
  setting → add it there with a sensible default, and to `.env.example`.
- **`config.py` reads env vars at import time**, and resolves relative paths against
  `PROJECT_ROOT`. Tests that need different values must set `os.environ[...]` *before*
  `app.config` is first imported anywhere in the chain.
- **No LangChain/LlamaIndex, no hidden abstractions.** Keep new pipeline code plain and
  readable — that's the point of this project.
- **`app/ingest.py` is the single entry point for adding documents.** Called from the CLI,
  from the FastAPI lifespan startup, and from `POST /ingest`. All three go through
  `ingest_data_folder()`, which fingerprints files by SHA-256 and skips unchanged ones.
- **Ingestion never blocks a request.** `start_job()` runs it on a background thread;
  `/ingest` and `/reset` return 202 and the client polls `/ingest/status`.
- **Embeddings model is a module-level singleton** — model loads are slow, don't move it
  into a request path.
- `data/` and `backend/test_fixtures/` are strictly separate: `data/` is real user
  documents; `test_fixtures/` is synthetic PDFs from `make_test_pdf.py`. Never point the
  fixture generator's default output at `data/`.

## Running it

```bash
cd backend
python -m venv venv && venv\Scripts\activate   # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then set GROQ_API_KEY
python scripts/ingest.py               # build the index (or let startup do it)
uvicorn app.main:app --reload --port 8000
```

Note the module path is `app.main:app`, run from `backend/`.

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
python tests/test_pipeline_offline.py
```

Stubs `sentence_transformers` via `sys.modules` so it runs offline, points `DATA_DIR` and
`CHROMA_DIR` at temp folders, and drives real HTTP endpoints through `TestClient`. It does
**not** prove the real model downloads or that a live Groq call succeeds — do a manual
smoke test with a real `GROQ_API_KEY` before considering a change done.

## Known gotchas (already fixed, keep them fixed)

- **Embedding truncation.** `all-MiniLM-L6-v2` reads only 256 tokens; the project's
  ~300-word chunks were being silently truncated, so the tail of every chunk never
  influenced retrieval. Now on `BAAI/bge-small-en-v1.5` (512 tokens), and
  `embeddings.warn_if_truncated()` logs a warning if chunks ever exceed the window again.
  If you switch models, check `max_seq_length` and set `EMBEDDING_QUERY_PREFIX`
  accordingly (bge/e5 want a query-side prefix; MiniLM wants none).
- **Paragraph structure must survive extraction.** `_normalize()` deliberately keeps
  `\n\n`. Collapsing all whitespace (`" ".join(text.split())`) destroys the boundaries the
  chunker packs on and measurably worsened page attribution.
- **Use `chromadb>=1.0`, not 0.5.x.** 0.5.5 hard-requires `numpy<2.0`, which has no Windows
  ARM64 wheel — pip is forced onto numpy 2.x and 0.5.5 then crashes at import
  (`np.float_` removed). 1.x also dropped the posthog telemetry dependency and its log spam.
- **Use `groq>=1.x`, not 0.11.0.** The old client passes a `proxies` kwarg that
  `httpx>=0.28` rejects: `TypeError: Client.__init__() got an unexpected keyword argument`.
- `pdf_utils.py` imports `pymupdf`, not `fitz` (the old name is deprecated), and silences
  MuPDF's per-image `cmsOpenProfileFromMem` stderr spam.
- **PyMuPDF is AGPL-3.0**, unlike every other dependency here. Fine for personal/local use;
  comply with AGPL or buy a commercial license before shipping this closed-source.

## Known limitations (see OPTIMIZATIONS.md for the reviewed backlog)

- No re-ranking and no keyword/hybrid search — dense vectors only.
- No evaluation harness; answer quality is unmeasured.
- No OCR — scanned/image-only PDFs are reported `"status": "skipped"`.
- No conversation history.
- One collection, no per-document filter at query time.
- No auth on `/ingest` or `/reset`.

## Before you commit

- Never commit `backend/.env` (real API key) or documents in `data/` — both gitignored.
- Run `tests/test_pipeline_offline.py`; all checks must pass.
