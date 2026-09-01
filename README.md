# Document Q&A — a from-scratch RAG system

PDF in → chunked & embedded locally → stored in ChromaDB → retrieved at question time →
answered by an LLM on Groq, grounded only in what's in the PDF.

No LangChain/LlamaIndex — every step is plain Python so you can see exactly what's happening
at each stage. That's deliberate, given this is a learning project.

**Everything is indexed from the backend's `data/` folder.** Drop PDFs in there yourself,
or upload them from the web UI — `POST /upload` writes the file into that same folder and
then the ordinary ingestion job indexes it.

**It is multi-tenant.** Accounts live in a local MongoDB; documents belong to the account
that uploaded them, and no filter, search or answer ever crosses that line. Tokens are
revocable (`token_version`), requests are rate limited, and each account has a storage
quota. The API is still plain HTTP, so put TLS in front of it before it leaves localhost.

**Where your data goes.** PDFs are stored on the server and indexed locally - extraction,
embedding and search never leave the machine. When you ask a question, the passages that
match it (up to ~24,000 characters) are sent to Groq to write the answer. Nothing else is.

## Stack

| Stage             | Tool                                  |
|-------------------|----------------------------------------|
| PDF text extraction | `PyMuPDF` (`pymupdf`)                 |
| Chunking           | custom structure-aware chunker (`app/pdf_utils.py`) |
| Embeddings         | `sentence-transformers` (`BAAI/bge-small-en-v1.5`, runs locally, free) |
| Vector store       | ChromaDB (persistent, local folder, no server to run) |
| Keyword search     | `rank-bm25` fused with vector search via Reciprocal Rank Fusion |
| Re-ranking         | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, free) |
| LLM                | Groq API (`openai/gpt-oss-20b` by default) |
| Backend            | FastAPI + Uvicorn |
| Frontend           | Plain HTML/CSS/JS (no build step) |

> **License note:** `PyMuPDF` is licensed AGPL-3.0 (unlike the permissively-licensed
> libraries above). Fine for personal/learning use; if you ever open-source or sell this,
> either comply with AGPL (source stays open) or buy Artifex's commercial PyMuPDF license.

## Project layout

```
My RAG Project/
├── src/                          the application package
│   ├── main.py                   app assembly: router, CORS, lifespan, static mount
│   ├── api/                      HTTP layer — thin handlers
│   │   ├── chat.py               /chat, /chat/stream (SSE)
│   │   ├── documents.py          /ingest, /ingest/status, /stats, /reset, /upload, DELETE /documents
│   │   └── system.py             /health, /info
│   ├── core/
│   │   ├── config.py             every setting, loaded once from .env
│   │   └── logging.py            logging config + per-stage timing helper
│   ├── models/
│   │   └── schemas.py            pydantic request/response shapes
│   ├── ml/
│   │   ├── embeddings.py         sentence-transformers (+ truncation guard)
│   │   ├── reranker.py           lazy cross-encoder singleton
│   │   └── llm.py                grounded prompt, Groq call (sync + streaming)
│   ├── services/
│   │   ├── ingestion.py          the ONLY path documents enter the system
│   │   ├── manifest.py           what's ingested, with content hashes
│   │   ├── pdf.py                extraction + structure-aware chunking
│   │   ├── vectorstore.py        ChromaDB add / query / neighbours / delete / reset
│   │   ├── bm25.py               lazy in-memory BM25 keyword index
│   │   └── retrieval.py          fusion → re-rank → floor → neighbour expansion
│   └── static/                   the web UI, served at "/" by FastAPI
│       └── index.html / style.css / script.js
├── scripts/
│   ├── ingest.py                 CLI: build the index without starting the API
│   └── make_test_pdf.py          generates the fictional fixture PDF
├── eval/
│   ├── golden_questions.json     golden set for measuring answer quality
│   └── run_eval.py               hit-rate@k, MRR, refusal rate, optional LLM-as-judge
├── tests/
│   └── test_pipeline_offline.py  86 offline checks
├── data/                         your real PDFs go here — the only way to add documents
├── storage/chroma_db/            generated index state (gitignored)
├── requirements.txt / requirements-dev.txt / .env.example
└── README.md · CLAUDE.md · PLAN.md · OPTIMIZATIONS.md
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
1. If the conversation has history, the question is first rewritten into a standalone one
   ("what about the second one?" retrieves nothing useful as written).
2. Two searches run over the same corpus: **vector** (semantic, good at paraphrase) and
   **BM25** (lexical, good at exact terms like "A* search"). Their ranked lists are merged
   with Reciprocal Rank Fusion.
3. A **cross-encoder re-ranks** the ~30 fused candidates by reading each (question, chunk)
   pair together, and the best `top_k` survive.
4. A **relevance floor** applies: if nothing scores well enough, the app answers "not in
   these documents" *without calling the LLM at all*.
5. Each surviving chunk is returned with its **neighbouring chunks**, so the model reads
   continuous prose rather than a fragment.
6. Those chunks go into a prompt as CONTEXT (under a character budget), with a system
   prompt instructing the model to answer only from that context.
7. Groq generates the answer — streamed token by token — with the documents and pages it
   drew from.

## Setup

Run everything from the **project root**:

```bash
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
python scripts/ingest.py          # ingest new / changed PDFs, prune deleted ones
python scripts/ingest.py --status # show what's in the store
python scripts/ingest.py --force  # wipe and rebuild from scratch
```

Subfolders work too — `data/textbooks/norvig.pdf` is ingested and identified by that
relative path.

Files are fingerprinted by content hash, so re-running is always safe: unchanged files are
skipped, and an edited PDF is re-ingested (its old chunks are deleted first, not
duplicated). You can also trigger the same scan at runtime with `POST /ingest`, or the
"Sync documents" button in the frontend — both run in the background.

The first run downloads the embedding model (~130MB) and then caches it. Embedding a large
book is real CPU work and can take several minutes.

## Running it

```bash
uvicorn src.main:app --reload --port 8000
```

Then open **http://localhost:8000** — that's it. The web UI is served by the same FastAPI
process from `src/static/`, so there is no separate file to open and no CORS hop.
`http://localhost:8000/docs` gives you the interactive API reference.

The API starts immediately and kicks off an ingestion pass in the background — it does not
block on embedding. `GET /ingest/status` reports progress.

## API

| Method | Path              | Purpose |
|--------|-------------------|---------|
| GET    | `/`               | the web UI (served from `src/static/`) |
| GET    | `/health`         | liveness check |
| GET    | `/info`           | which embedding / re-rank / LLM models this instance is running |
| POST   | `/ingest`         | start a background scan of `data/` (202) |
| GET    | `/ingest/status`  | progress of the current/last ingestion job |
| POST   | `/chat`           | `{"question": "...", "top_k": 4, "source": null, "history": []}` → answer + sources |
| POST   | `/chat/stream`    | same body, streamed as SSE (`sources`, then `token`s, then `done`) |
| GET    | `/stats`          | chunk count, per-document pages/chunks, ingestion state |
| POST   | `/reset`          | wipe the store and rebuild from `data/` (202) |
| GET    | `/ready`          | readiness: 503 until the model is loaded and MongoDB answers |
| POST   | `/api/signup`     | create an account, returns a JWT (201) |
| POST   | `/api/me/password` | change password; invalidates every other session |
| POST   | `/api/me/signout-everywhere` | invalidate all tokens for this account |
| DELETE | `/api/me`         | delete the account and everything it owns |
| POST   | `/api/login`      | exchange credentials for a JWT |
| GET    | `/api/me`         | the signed-in user |
| GET    | `/api/documents`  | the caller's documents |
| POST   | `/upload`         | upload one or more PDFs (multipart `files`), then index them (202) |
| DELETE | `/documents/{name}` | remove a document: its vectors, its manifest entry, and the PDF |

Interactive docs at `http://localhost:8000/docs`.

## Testing

```bash
pip install -r requirements-dev.txt
python tests/test_pipeline_offline.py
```

86 checks covering the chunker (page ranges, oversized paragraphs, overlap edge cases),
startup ingestion, `/ingest` idempotency, change detection (an edited PDF is re-ingested,
not duplicated), BM25 tokenisation and scoping, RRF fusion, cross-encoder re-ranking, the
relevance floor, neighbour expansion, per-document scoping, conversation history, SSE
streaming, input validation, and `/reset`. It generates its own fixture PDFs into an
isolated temp folder — never `data/` — stubs the embedding and re-ranking models so it runs
offline, and doesn't call Groq. It does **not** prove the real models download or that a
live Groq call succeeds; do a manual smoke test with a real key.

## Measuring answer quality

The test suite proves the plumbing works. The eval harness measures whether the answers are
any *good* — and, more usefully, whether a change made them better:

```bash
python eval/run_eval.py                  # hit-rate@k, MRR, refusal rate (no API key needed)
python eval/run_eval.py --judge          # + LLM-as-judge correctness/groundedness/relevance
python eval/run_eval.py --no-rerank      # A/B: what is the cross-encoder actually worth?
python eval/run_eval.py --no-hybrid      # A/B: what is BM25 worth?
```

`eval/golden_questions.json` currently holds questions about the *fixture* PDF so the
harness runs out of the box. **Replace them with 20–30 questions about your real documents**
— that is what turns tuning (chunk size, `top_k`, the relevance floor) from guesswork into
measurement.

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

- **OCR is off by default** — set `OCR_ENABLED=true` and install `pytesseract`, `pillow`
  and the Tesseract binary; otherwise scanned/image PDFs are skipped.
- **Conversation history is client-side** — the backend is stateless, so history is lost on
  a browser reload.
- **Page ranges are chunk-level**, not sentence-level — good for "roughly where to look".
- **The golden question set still targets the fixture PDF**, so the eval numbers describe a
  fictional 3-page document until you replace them.
- **No auth on `/ingest` or `/reset`** — fine locally, not for public exposure.
- **Windows ARM64**: numpy has no official MSVC/OpenBLAS build there, so pip installs a
  community MINGW-W64 build that prints an "experimental" warning. Harmless on its own.

`OPTIMIZATIONS.md` is the rationale record for why the pipeline looks the way it does;
`PLAN.md` is the changelog and the remaining backlog.
