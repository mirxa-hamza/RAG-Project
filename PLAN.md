# PLAN.md

Roadmap and changelog for the Document Q&A RAG project. Status as of 2026-08-31.

## Status

Full retrieval pipeline in place: hybrid (vector + BM25) search, Reciprocal Rank Fusion,
cross-encoder re-ranking, a relevance floor, neighbour expansion, per-document scoping,
streaming answers, conversation history with follow-up rewriting, optional OCR, and an
evaluation harness. 86 automated checks pass offline. Two real textbooks (AI: A Modern
Approach; Pattern Classification) live in `data/`.

**Action required after this change:** none beyond a restart — the audit pass below is
backend-only and needs no re-ingest. (The previous pass added `rank-bm25`; if you haven't
run `pip install -r requirements.txt` since then, do that first.)

```bash
pip install -r requirements.txt
python scripts/ingest.py --status      # confirm the index is intact (no re-ingest needed)
uvicorn src.main:app --reload --port 8000   # then open http://localhost:8000
```

The embedding model and chunking are unchanged this pass, so **the existing index is still
valid** — no `--force` re-ingest required. The cross-encoder (~80MB) downloads on the first
question asked.

## Changelog — Tier 1 + Tier 2 (previous pass)

**Retrieval correctness**

- [x] **Fixed silent embedding truncation.** `all-MiniLM-L6-v2` reads 256 tokens; ~300-word
      chunks were ~400+ tokens, so roughly the last third of every chunk never influenced
      retrieval. Switched to `BAAI/bge-small-en-v1.5` (512-token window) with the
      query-side instruction prefix these models expect, plus a
      `warn_if_truncated()` guard that logs if chunks ever exceed the window again.
- [x] **Structure-aware chunking.** Extraction now preserves paragraph breaks (the old
      `" ".join(text.split())` destroyed them before the chunker ran), and chunks are
      packed from whole paragraphs — falling back to sentence splits, then hard word
      splits, only when a paragraph is oversized. Replaces blind every-N-words slicing.
- [x] **Page ranges instead of a single guessed page.** Chunks now record `page_start` and
      `page_end`, and citations render "page 7" or "pages 7-8". (Visible in the test run:
      the funding question now cites page 2, where the figure actually is — it cited
      page 1 before.)
- [x] **Clamped similarity.** Chroma's cosine distance runs 0-2, so `1 - distance` could
      report a negative similarity.
- [x] **Content-hash change detection.** Files are fingerprinted by SHA-256; an edited PDF
      is re-ingested and its old chunks deleted first, instead of being invisible forever
      (filename-only matching) or duplicated.

**Architecture / performance**

- [x] **Ingestion no longer blocks the API.** It runs on a background thread; `/ingest` and
      `/reset` return 202 immediately and the frontend polls `GET /ingest/status` for
      progress. Server startup is instant.
- [x] **CLI index builder:** `python scripts/ingest.py [--force|--status]`, so the index
      can be built offline without running the API at all.
- [x] **Batched embedding and storage** (`EMBEDDING_BATCH_SIZE`, `CHROMA_ADD_BATCH`) —
      removes the memory spike from encoding thousands of chunks in one call and stays
      under Chroma's per-add batch ceiling.
- [x] **Manifest sidecar** replaces `collection.get(include=["metadatas"])`, which pulled
      every chunk's metadata on every `/stats` call, every page load, and every ingest.
      `/stats` now also reports per-document page and chunk counts.
- [x] **Structured logging with per-stage timings** (`logging_setup.timed`) instead of
      scattered `print()` calls.

**Hardening (small items from Tier 4)**

- [x] `top_k` is bounded (`1..MAX_TOP_K`) and blank questions are rejected at validation,
      so a client can't build an enormous prompt.
- [x] The assembled CONTEXT is capped by `MAX_CONTEXT_CHARS`, independent of `top_k`.
- [x] The Groq call is wrapped — a rate limit or deprecated model id returns a readable
      message instead of an unhandled 500.

**Project structure**

- [x] Backend reorganized into `app/` (package), `scripts/` (CLI entry points), `tests/`.
      Run target is now `uvicorn app.main:app`.
- [x] Dropped `python-multipart` — it was only needed by the removed upload endpoint.
- [x] `refrence/` gitignored (kept on disk for study, not part of this codebase).

## Changelog — Tier 3 + Tier 4 (this pass)

**Retrieval quality (Tier 3)**

- [x] **Cross-encoder re-ranking** (`app/reranker.py`). Retrieve `RETRIEVAL_CANDIDATES`
      (30) cheaply, then score each (question, chunk) pair with
      `cross-encoder/ms-marco-MiniLM-L-6-v2` and keep the best `top_k`. Loads lazily on
      first question and degrades to fusion-only ranking if the model can't be loaded.
- [x] **Hybrid search with RRF** (`app/bm25.py`). BM25 keyword ranking fused with vector
      ranking by Reciprocal Rank Fusion. The tokenizer deliberately preserves `A*`,
      `k-means`, `f1` — a plain `\w+` split destroys exactly the terms BM25 exists to
      catch. The index is built lazily in memory and invalidated on any store change.
- [x] **Per-document scoping.** `POST /chat` accepts `source`; the frontend has an
      "Ask about" dropdown fed from `/stats`.
- [x] **Retrieve-more-then-filter + relevance floor.** Nothing above `MIN_RERANK_SCORE`
      (or `MIN_SIMILARITY` when re-ranking is off) means the system answers "not in these
      documents" **without calling the LLM** — faster, cheaper, and it removes a whole
      class of confident answers built on irrelevant context.
- [x] **Neighbour expansion.** Each hit is returned with its `chunk_index ± 1` neighbours,
      in document order, so the model reads continuous prose rather than a fragment.

**Evaluation and polish (Tier 4)**

- [x] **Evaluation harness** (`eval/run_eval.py` + `eval/golden_questions.json`).
      Retrieval hit-rate@k, MRR, and refusal rate need no API key; `--judge` adds
      LLM-as-judge correctness/groundedness/relevance via Groq, modelled on the four axes
      in the reference project's LangSmith notebook. `--no-rerank` / `--no-hybrid` /
      `--no-expand` / `--top-k` make every stage measurable. First run on the fixture set:
      hit-rate 100%, MRR 1.000, refusal 100% — and with `--no-rerank`, MRR drops to 0.812
      and refusal to 0%, which is the harness doing its job.
- [x] **Streaming answers.** `POST /chat/stream` emits SSE: `sources` first, then `token`
      events, then `done`. The frontend renders tokens as they arrive with a caret.
- [x] **Conversation history + follow-up rewriting.** The frontend keeps recent turns and
      sends them with each question; the backend rewrites follow-ups into standalone
      questions *before* retrieval (a raw "what about the second one?" embeds to noise),
      and shows the rewritten query in the sources line.
- [x] **OCR fallback.** `OCR_ENABLED=true` rasterises text-less pages with PyMuPDF and runs
      Tesseract. Both the Python packages and the binary are optional — missing pieces log
      a warning and the document is skipped exactly as before.
- [x] **Ordering fix found while testing:** "nothing relevant was retrieved" is now
      returned *before* the missing-API-key check, since that answer needs no LLM call.

## Changelog — project restructure + self-served UI (this pass)

- [x] **Reorganised into a purpose-based `src/` package at the project root**, matching the
      layout used in the owner's other FastAPI projects: `api/` (thin HTTP handlers, split
      by endpoint group), `core/` (config + logging), `models/` (pydantic schemas), `ml/`
      (embeddings, re-ranker, LLM), `services/` (pdf, chunking, stores, retrieval,
      ingestion, manifest), `static/` (the UI). The `backend/` wrapper folder is gone —
      there was only ever one backend, so it added a level of nesting for nothing.
- [x] **`main.py` is now app assembly only** — router wiring, CORS, lifespan, static mount.
      Endpoints moved to `src/api/{chat,documents,system}.py`, schemas to
      `src/models/schemas.py`.
- [x] **Added `GET /info`** reporting the embedding / re-rank / LLM models in use — the
      first thing to check when answers look different from what you expected.
- [x] **The frontend folder is gone; FastAPI serves the UI itself.** `src/static/` is
      mounted at `/` (registered *after* the API router, so API paths always win), and
      `script.js` now uses a relative `API_BASE`. Open http://localhost:8000 — no separate
      HTML file, no CORS hop.
- [x] **Generated state moved to `storage/chroma_db/`**, out of the source tree, and
      `tests/fixtures/` replaces `backend/test_fixtures/`.
- [x] All 86 checks still pass, and the UI/API route precedence is covered by a live check
      (`/` serves HTML, `/health` and `/chat` still resolve as API routes).

## Changelog — edge-case audit (previous pass)

Six defects found by probing the code with runnable repros rather than reading it, plus
four robustness gaps. Every one has a regression test.

**Critical**

- [x] **One malformed PDF aborted the entire ingest job.** Verified: with three files and a
      corrupt one in the middle, the third was never indexed and the job reported failure.
      `ingest_data_folder()` now catches per file and records
      `{"status": "failed", "error": ...}`, continuing with the rest of the corpus.
- [x] **`build_context()` could return an empty string.** If the first chunk exceeded
      `MAX_CONTEXT_CHARS` the loop broke immediately, so the model was told "answer only
      from CONTEXT" and handed nothing — a direct hallucination path. It now truncates that
      chunk (keeping its citation label) instead of dropping it.
- [x] **The embedding query prefix was malformed by `.env`.** python-dotenv strips trailing
      whitespace from unquoted values, so every query embedded as
      `...passages:What is A* search?` with no space. `config.py` now normalises the prefix.
      This silently degraded *every* query — the same class of bug as the truncation issue.

**Major**

- [x] **Deleted PDFs were never pruned.** A file removed from `data/` stayed searchable and
      citable indefinitely. `prune_deleted()` now reconciles the store against disk on every
      ingest and reports `{"status": "removed"}`.
- [x] **Manifest writes were not atomic.** A torn read returned `{}`, which convinced the
      ingester nothing was stored and triggered a full re-embed of both textbooks. Writes
      now go through a temp file + `os.replace` under a lock.
- [x] **Nested PDFs were invisible.** `data/textbooks/norvig.pdf` was silently ignored;
      the scan is now recursive and documents are keyed by POSIX-style relative path.

**Moderate**

- [x] **N+1 queries for neighbour expansion** — measured 5 separate Chroma `get()` calls for
      one question. Now one batched query per source (`get_neighbors_bulk`).
- [x] **`top_k` above `RETRIEVAL_CANDIDATES`** silently returned fewer hits than requested;
      the candidate pool now widens to match.
- [x] **Byte-identical duplicates** under different filenames were indexed twice and then
      competed with themselves for every retrieval slot. Now detected and skipped.
- [x] **The re-ranker latched off permanently** after a single failed load — one transient
      network blip disabled the biggest quality stage for the life of the process. It now
      retries after 5 minutes.

## Next steps — the remaining backlog

Full detail and rationale in `OPTIMIZATIONS.md`. In priority order:

0. **Re-run `python scripts/ingest.py`** once after updating, so the deleted-document
   pruning and recursive scan reconcile the store with what's actually on disk.
1. **Replace the golden question set with real ones.** `eval/golden_questions.json` still
   targets the fictional fixture PDF. Write 20-30 questions against the two textbooks with
   the page you'd expect cited. Until then the eval numbers describe a 3-page fake
   document, not the corpus you actually query. This is now the highest-value task in the
   project — everything below should be measured against it.
2. **Tune with the harness, now that it exists.** Sweep `CHUNK_SIZE_WORDS`,
   `RETRIEVAL_CANDIDATES`, `TOP_K` and `MIN_RERANK_SCORE` against real questions; the
   defaults were chosen by reasoning, not measurement.
3. **Calibrate `MIN_RERANK_SCORE` on real data.** -6.0 is a sensible starting point for
   ms-marco logits, but the right floor depends on the corpus. Too high and good answers
   get refused; too low and the "not in these documents" path never fires.
4. **Persist conversations** if you want history to survive a browser reload — currently
   the backend is stateless by design and history lives in the page.
5. **Auth in front of `/ingest` and `/reset`** before this is ever exposed beyond
   localhost.
6. **Agentic routing (OPTIMIZATIONS 4.6)** — gate retrieval behind a cheap decision for
   greetings and general-knowledge questions. Mostly redundant now that the relevance
   floor short-circuits irrelevant questions without an LLM call; keep it as a learning
   exercise rather than a priority.
7. **If you ever deploy this beyond your own machine:** hosted vector DB instead of local
   Chroma persistence, tighten `allow_origins=["*"]`, and note that the in-memory BM25
   index is per-process (each worker builds its own).

## Explicitly out of scope

- A build step for the frontend (deliberately plain HTML/CSS/JS).
- LangChain/LlamaIndex — seeing every step is the point of this project.
- Any way for the frontend/a user to add, replace, or delete documents. Deliberate design
  constraint, not a missing feature.
