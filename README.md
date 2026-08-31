# Document Q&A — a from-scratch RAG system

PDF in → chunked & embedded locally → stored in ChromaDB → retrieved at question time →
answered by an LLM on Groq, grounded only in what's in the PDF.

No LangChain/LlamaIndex — every step is plain Python so you can see exactly what's happening
at each stage. That's deliberate, given this is a learning project.

**Documents come only from the backend's `data/` folder.** There is no upload endpoint —
the frontend can only ask questions, never add or change what's in the vector store.

## Stack

| Stage             | Tool                                  |
|-------------------|----------------------------------------|
| PDF text extraction | `PyMuPDF` (`pymupdf`)                 |
| Chunking           | custom structure-aware chunker (`app/pdf_utils.py`) |
| Embeddings         | `sentence-transformers` (`BAAI/bge-small-en-v1.5`, runs locally, free) |
| Vector store       | ChromaDB (persistent, local folder, no server to run) |
| LLM                | Groq API (`openai/gpt-oss-20b` by default) |
| Backend            | FastAPI + Uvicorn |
| Frontend           | Plain HTML/CSS/JS (no build step) |

> **License note:** `PyMuPDF` is licensed AGPL-3.0 (unlike the permissively-licensed
> libraries above). Fine for personal/learning use; if you ever open-source or sell this,
> either comply with AGPL (source stays open) or buy Artifex's commercial PyMuPDF license.

## Project layout

```
My RAG Project/
├── backend/
│   ├── app/                        the application package
│   │   ├── main.py                 FastAPI app: /ingest, /ingest/status, /chat, /stats, /reset, /health
│   │   ├── config.py               every setting, loaded once from .env
│   │   ├── ingest.py               the ONLY path documents enter the system + background job
│   │   ├── manifest.py             small JSON record of what's ingested (hashes, page/chunk counts)
│   │   ├── pdf_utils.py            PDF → pages → structure-aware overlapping chunks
│   │   ├── embeddings.py           wraps sentence-transformers (+ truncation guard)
│   │   ├── vectorstore.py          ChromaDB: add / query / delete / reset
│   │   ├── llm.py                  builds the grounded prompt, calls Groq
│   │   └── logging_setup.py        logging config + per-stage timing helper
│   ├── scripts/
│   │   ├── ingest.py               CLI: build the index without starting the API
│   │   └── make_test_pdf.py        generates the fictional fixture PDF
│   ├── tests/
│   │   └── test_pipeline_offline.py   automated end-to-end test (see below)
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── .env.example                copy to .env and fill in your Groq key
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/                           your real PDFs go here — the only way to add documents
├── README.md
├── CLAUDE.md                       architecture notes for AI-assisted work on this repo
├── PLAN.md                         roadmap / changelog
└── OPTIMIZATIONS.md                code review + remaining optimization backlog
```

## How a question actually gets answered (the RAG loop)

**At ingestion time (CLI, server startup, or `POST /ingest`):**
1. The backend scans `data/` and fingerprints each PDF (SHA-256). New or changed files are
   ingested; unchanged ones are skipped.
2. `PyMuPDF` extracts text page by page, preserving paragraph breaks.
3. Paragraphs are packed into overlapping chunks of ~300 words (falling back to sentence
   and then word splits only when a paragraph is too big). Each chunk records the page
   range it covers.
4. Each chunk is embedded (turned into a vector capturing its meaning) by a local model —
   no API call, no cost.
5. Chunk text + vector + page range get stored in ChromaDB, in batches.

**At question time (every chat message):**
1. Your question is embedded with the *same* model, with the model's query prefix applied.
2. ChromaDB returns the chunks whose vectors are closest to the question's (cosine
   similarity) — the "retrieval" in Retrieval-Augmented Generation.
3. Those chunks go into a prompt as CONTEXT (under a character budget), with a system
   prompt instructing the model to answer only from that context.
4. Groq's LLM generates the answer, returned with the documents and pages it drew from.

## Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate           # Windows;  macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
copy .env.example .env          # Windows;  macOS/Linux: cp .env.example .env
```

Open `.env` and set `GROQ_API_KEY` (free key: https://console.groq.com/keys).
The default model is `openai/gpt-oss-20b` — Groq deprecates models periodically, so if you
get a "model not found" error, check https://console.groq.com/docs/models and update
`GROQ_MODEL`.

## Adding documents

Put PDFs directly in the `data/` folder — that's the only way documents get in. Then:

```bash
cd backend
python scripts/ingest.py          # ingest new / changed PDFs
python scripts/ingest.py --status # show what's in the store
python scripts/ingest.py --force  # wipe and rebuild from scratch
```

Files are fingerprinted by content hash, so re-running is always safe: unchanged files are
skipped, and an edited PDF is re-ingested (its old chunks are deleted first, not
duplicated). You can also trigger the same scan at runtime with `POST /ingest`, or the
"Sync from data folder" button in the frontend — both run in the background.

The first run downloads the embedding model (~130MB) and then caches it. Embedding a large
book is real CPU work and can take several minutes.

## Running it

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

The API starts immediately and kicks off an ingestion pass in the background — it does not
block on embedding. `GET /ingest/status` reports progress.

Then open `frontend/index.html` directly in your browser (double-click it, or use VS Code's
Live Server). It is a local file, **not** served by the backend — `http://localhost:8000`
is the API only. Change `API_BASE` at the top of `frontend/script.js` if you serve the
backend elsewhere.

## API

| Method | Path              | Purpose |
|--------|-------------------|---------|
| GET    | `/health`         | liveness check |
| POST   | `/ingest`         | start a background scan of `data/` (202) |
| GET    | `/ingest/status`  | progress of the current/last ingestion job |
| POST   | `/chat`           | `{"question": "...", "top_k": 4}` → answer + sources |
| GET    | `/stats`          | chunk count, per-document pages/chunks, ingestion state |
| POST   | `/reset`          | wipe the store and rebuild from `data/` (202) |

Interactive docs at `http://localhost:8000/docs`.

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
python tests/test_pipeline_offline.py
```

37 checks covering the chunker (page ranges, oversized paragraphs, overlap edge cases),
startup ingestion, `/ingest` idempotency, change detection (an edited PDF is re-ingested,
not duplicated), input validation, retrieval, and `/reset`. It generates its own fixture
PDFs into an isolated temp folder — never `data/` — swaps in a deterministic fake embedding
model so it runs offline, and doesn't call Groq. It does **not** prove the real model
downloads or that a live Groq call succeeds; do a manual smoke test with a real key.

## Manual smoke test (with a real GROQ_API_KEY)

```bash
# terminal 1
uvicorn app.main:app --reload --port 8000

# terminal 2 — ask about whatever is in data/
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" ^
     -d "{\"question\": \"What is this document about?\"}"
```

Worth testing deliberately:
- Ask something *not* in the documents → it should say so, not invent an answer.
- Check the `sources` line cites the right document and a plausible page range.
- Try a scanned/image-only PDF → `/ingest` should report `"status": "skipped"` (no OCR yet).

## Known limitations

- **No OCR** — scanned/image PDFs extract no text and are skipped.
- **No conversation history** — each question is answered independently.
- **Single collection, no per-document filter** — a question retrieves across every
  ingested document.
- **Page ranges are chunk-level**, not sentence-level — good for "roughly where to look".
- **No auth on `/ingest` or `/reset`** — fine locally, not for public exposure.
- **Windows ARM64**: numpy has no official MSVC/OpenBLAS build there, so pip installs a
  community MINGW-W64 build that prints an "experimental" warning. Harmless on its own.

See `OPTIMIZATIONS.md` for the reviewed backlog (re-ranking, hybrid BM25 search, an
evaluation harness) and `PLAN.md` for the roadmap.
