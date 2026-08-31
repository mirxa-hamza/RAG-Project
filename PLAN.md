# PLAN.md

Roadmap and changelog for the Document Q&A RAG project. Status as of 2026-08-31.

## Status

Working end-to-end, backend-only ingestion, restructured into a proper `app/` package.
37 automated checks pass offline. Two real textbooks (AI: A Modern Approach; Pattern
Classification) live in `data/`.

**Action required after this change:** the embedding model changed, so the existing index
is stale and must be rebuilt:

```bash
cd backend
pip install -r requirements.txt
python scripts/ingest.py --force
```

## Changelog — this pass (Tier 1 + Tier 2 of OPTIMIZATIONS.md)

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

## Next steps — the remaining backlog

Full detail and rationale in `OPTIMIZATIONS.md`. In priority order:

1. **Evaluation harness (OPTIMIZATIONS 4.1).** 20-30 golden questions against the two
   textbooks plus a retrieval hit-rate@k metric. Do this *before* the retrieval-quality
   work below — otherwise there's no way to tell whether a change helped.
2. **Cross-encoder re-ranker (3.1).** Retrieve ~30, re-rank, keep 4. The single biggest
   remaining quality jump; local, free, ~100-300ms.
3. **Hybrid BM25 + vector search with RRF fusion (3.2).** Fixes exact-term questions
   ("A* search", "Bayes decision rule") that dense retrieval alone handles poorly.
4. **Per-document scoping (3.3).** A `source` filter on `/chat` plus a dropdown, so a
   question can target one textbook.
5. **Retrieve-more-then-filter + distance floor (3.4).** Answer "not in these documents"
   without calling the LLM when nothing scores well.
6. **Neighbor chunk expansion (3.5).** Pull `chunk_index ± 1` around each hit.
7. **Conversation history (4.5)** — and rewrite follow-up questions to standalone form
   *before* retrieving, or the retrieval step gets a question that embeds to nothing.
8. **Streaming responses (4.4).** Groq is fast; the UI currently hides that.
9. **OCR fallback (4.7)** via `page.get_pixmap()` + `pytesseract` for scanned PDFs.

## Explicitly out of scope

- A build step for the frontend (deliberately plain HTML/CSS/JS).
- LangChain/LlamaIndex — seeing every step is the point of this project.
- Any way for the frontend/a user to add, replace, or delete documents. Deliberate design
  constraint, not a missing feature.
