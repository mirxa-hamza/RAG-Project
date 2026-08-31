# OPTIMIZATIONS.md

A code review of this project's RAG pipeline, written after reading both this codebase and
the reference project in `refrence/` (LangChain + FAISS + Groq, with notebooks on RAG
evaluation, agentic RAG with LangGraph, PageIndex vectorless RAG, and Typesense hybrid
search).

Ranked by impact. Tier 1 items are affecting answer quality *right now* on the two
textbooks in `data/`. Tier 4 items are polish.

> **Status (2026-08-31): everything in this document is implemented.** Tier 1 and Tier 2
> first, then Tier 3 (re-ranking, hybrid BM25 + RRF, per-document scoping, relevance floor,
> neighbour expansion) and Tier 4 (evaluation harness, streaming, conversation history with
> follow-up rewriting, OCR fallback, `top_k` bounds, Groq error handling). See PLAN.md for
> the changelog and CLAUDE.md for how the pieces fit.
>
> This document is kept as the **rationale record** — why each change was made and what it
> was worth — not as an open backlog. The remaining work is listed in PLAN.md, and the top
> item is replacing `eval/golden_questions.json` with questions about the real textbooks:
> until then every number the harness reports describes a 3-page fictional PDF.

**A note on the reference project:** it leans on LangChain for loaders, splitting, and the
vector store. This project deliberately avoids LangChain (see CLAUDE.md) — that constraint
is kept below. Every recommendation is written as plain Python you can implement directly,
even where the reference solves the same problem with a LangChain component.

---

## Tier 1 — Correctness issues silently degrading retrieval

### 1.1 Chunks are longer than the embedding model can read (highest impact)

`all-MiniLM-L6-v2` has `max_seq_length = 256` word-piece tokens. Anything past that is
**silently truncated** — no error, no warning.

`CHUNK_SIZE_WORDS = 300` produces chunks of roughly 380–450 tokens for technical prose
(textbook text tokenizes badly: notation, hyphenation, rare terms). So roughly the **last
third of every chunk is never embedded**. The text is still stored and still shown in
`sources`, so it looks fine — but it contributes nothing to whether that chunk gets
retrieved.

Verify on your machine:

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2")
print(m.max_seq_length)  # 256
sample = " ".join(open_some_chunk_text.split()[:300])
print(len(m.tokenizer(sample)["input_ids"]))  # compare against 256
```

Three ways out, best first:

1. **Switch embedding model to one with a 512-token window** and better retrieval quality:
   `BAAI/bge-small-en-v1.5` (384-dim, same speed class as MiniLM, 512 tokens) or
   `intfloat/e5-small-v2`. Drop-in via `EMBEDDING_MODEL` in `.env` — but note bge/e5 models
   expect prefixes (`"query: "` / `"passage: "` for e5; bge wants an instruction prefix on
   queries only). If you switch, add that prefixing in `embeddings.py`, otherwise you lose
   most of the gain.
2. **Or shrink chunks** to `CHUNK_SIZE_WORDS = 180`, `CHUNK_OVERLAP_WORDS = 40` so they fit
   inside 256 tokens.
3. Either way, **assert it** — log a warning during ingestion when a chunk tokenizes past
   the model's `max_seq_length`, so this can never silently regress again.

Re-ingest (`POST /reset`) after changing any of this — old vectors were built with the
truncated text.

### 1.2 The chunker ignores document structure

`chunk_document()` flattens everything into one word stream and slices every N words. A
chunk routinely starts mid-sentence and ends mid-sentence, and a section heading gets
severed from the paragraph it introduces. For a textbook that's expensive: the heading
("12.4 Bayesian Network Inference") is often the most retrievable text on the page, and it
ends up in a different chunk than its content.

The reference uses `RecursiveCharacterTextSplitter` with `["\n\n", "\n", " ", ""]`. The same
idea in ~25 lines of plain Python: try to split on paragraph breaks; if a piece is still too
long, split on sentence ends; only fall back to hard word slicing as a last resort. Keep the
overlap behavior you already have.

This requires preserving newlines — currently `extract_pages()` does
`" ".join(text.split())`, which destroys every paragraph boundary before the chunker ever
sees it. Collapse runs of spaces, but keep `\n\n`.

### 1.3 Page attribution is wrong for cross-page chunks

A chunk is labeled with the page its *first word* came from. A 300-word chunk spans 1–2
pages routinely, so a citation of "(page 412)" can point a page early. Cheap fix: store
`page_start` and `page_end` in metadata (you already have per-word page numbers in
`words_with_page`) and cite a range when they differ.

### 1.4 `similarity` can go negative

`round(1 - c["distance"], 3)` assumes cosine distance ≤ 1. Chroma's cosine distance runs
0–2, so a genuinely dissimilar chunk reports a negative similarity in the UI. Clamp to
`max(0.0, 1 - distance)`, or display the raw distance and label it as such.

### 1.5 Re-ingest detection is filename-only

`ingest_data_folder()` skips a file if its name is already in the store. Edit a PDF, keep
the name, and the change is invisible forever. Store `mtime` and a content hash
(`hashlib.sha256` over the file bytes, or just size+mtime for speed) in the chunk metadata,
and re-ingest when it differs. Requires deleting the old chunks for that source first —
`_collection.delete(where={"source": filename})`.

---

## Tier 2 — Architecture and performance

### 2.1 Ingestion blocks the API from starting

The FastAPI `startup` event runs the full ingest before Uvicorn accepts connections. You
already hit this: `ERR_CONNECTION_REFUSED` for several minutes while two textbooks embedded.
Worse, `POST /reset` re-ingests *inside the HTTP request* — for this corpus that request
will exceed any normal client timeout.

Two fixes, do both:

1. **Split ingestion out into a CLI step**: `python ingest.py` (add an `if __name__ ==
   "__main__"` block calling `ingest_data_folder()`). The reference does exactly this —
   build the index once, offline, then the app just loads it. The server then starts in
   seconds.
2. **Make the HTTP paths non-blocking**: `POST /ingest` and `POST /reset` should kick off a
   background job (`fastapi.BackgroundTasks`, or a thread with a module-level status dict)
   and return `202` immediately, with a `GET /ingest/status` endpoint reporting
   `{state, current_file, files_done, chunks_done}`. The frontend's "Sync from data folder"
   button then polls that instead of hanging.

### 2.2 The whole corpus is embedded in one `encode()` call

`embed_texts()` passes every chunk of a document at once — 1781 chunks for the Russell &
Norvig book, held as one list of Python floats after `.tolist()`. That's a large, avoidable
memory spike, and it delays the first sign of progress.

Pass `batch_size=64` to `encode()`, and add to Chroma in batches too. Chroma enforces a max
batch size (`client.get_max_batch_size()`, historically 5461) — a bigger book than these two
would hit it and raise. Batching `add_chunks()` at, say, 1000 chunks per call removes both
problems and lets you print real progress between batches.

### 2.3 `list_sources()` reads every chunk's metadata

```python
all_meta = _collection.get(include=["metadatas"])["metadatas"]
```

No limit, no filter — this pulls metadata for **every chunk in the store** (thousands of
dicts) and is called by `/stats`, by the frontend on every page load, and by
`ingest_data_folder()` on every startup and every `/ingest`. It works, but it's O(corpus)
for a question that has a handful of answers.

Keep a tiny manifest instead: a `sources.json` next to the Chroma directory (or a separate
one-row-per-document Chroma collection) recording `{filename, sha256, mtime, pages, chunks,
ingested_at}`. `/stats` and the dedupe check then read a small file, and you get the
re-ingest metadata from 1.5 for free.

### 2.4 No structured timing or logging

Every stage prints ad-hoc `[ingest]` lines with `print()`. Move to `logging` with a level,
and time each stage (extract / chunk / embed / store on ingest; embed / retrieve / LLM on
query). Without per-stage timings you can't tell whether a slow answer is retrieval or Groq,
which is the first question you'll ask when tuning.

---

## Tier 3 — Retrieval quality upgrades (where the real wins are now)

These are the difference between "it finds something" and "it finds the right thing" on
dense technical books. In rough order of value-per-effort:

### 3.1 Add a cross-encoder re-ranker

The single biggest quality jump available. Retrieve top-30 by vector similarity, then score
each `(question, chunk)` pair with a cross-encoder and keep the best 4:

```python
from sentence_transformers import CrossEncoder
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # ~80MB, CPU, local, free
scores = _reranker.predict([(question, c["text"]) for c in candidates])
```

Bi-encoder embeddings compress a chunk into one vector before ever seeing the question; a
cross-encoder reads both together. On textbook content the precision difference is large,
and it costs ~100–300ms on CPU for 30 candidates. No API, no cost.

### 3.2 Add keyword (BM25) search and fuse it with the vector search

Dense embeddings are weak exactly where textbooks are strong: exact technical terms,
notation, named algorithms. Ask "what is A* search" and a pure-vector store will happily
return general search-algorithm prose. `rank_bm25` (pure Python, tiny) over the same chunks,
fused with the vector hits by Reciprocal Rank Fusion, fixes most of these:

```python
score = sum(1.0 / (60 + rank) for rank in ranks_of_this_chunk_in_each_list)
```

The reference project explored this same idea via Typesense's hybrid search — you can get it
without adding a search server.

### 3.3 Scope questions to one document

Two textbooks now share one collection, so a Pattern Classification question can pull
Russell & Norvig chunks. Chroma supports this directly:

```python
_collection.query(..., where={"source": filename})
```

Expose it as an optional `source` field on `/chat` and a dropdown in the frontend (you
already fetch the source list for the sidebar).

### 3.4 Retrieve more, then filter

`TOP_K = 4` was tuned against a 3-page fictional PDF. With ~4000 chunks across two books,
retrieve 20–30 candidates and let the re-ranker (3.1) cut back to 4–6. Also add a distance
floor: if nothing scores above a threshold, answer "not in these documents" *without*
calling the LLM — faster, cheaper, and it removes a whole class of confident-sounding
answers built on irrelevant context.

### 3.5 Give the model the chunk's neighbors

Retrieval finds the chunk containing the answer; the sentence that explains it is often in
the next chunk. Store a sequential `chunk_index` per source, and when a chunk is retrieved,
also fetch `chunk_index ± 1` before building the prompt ("context window expansion"). Cheap
to implement, noticeably better answers on definitions and derivations.

---

## Tier 4 — Evaluation, safety, and polish

### 4.1 There is no evaluation harness — this is the biggest process gap

`test_pipeline_offline.py` proves the *plumbing* works. Nothing measures whether answers are
*good*. Every tuning decision above (chunk size, model, reranker on/off, top_k) is currently
guesswork.

The reference project's `1-rag_evaluation.ipynb` is the right model: LLM-as-judge scoring on
four axes — **correctness** (vs. a reference answer), **groundedness** (answer supported by
retrieved chunks), **relevance** (answer addresses the question), and **retrieval relevance**
(retrieved chunks address the question). It uses LangSmith; you don't need to. A local
version:

- Write 20–30 golden questions against your two textbooks with expected answers and the page
  you'd expect cited. A JSON file is enough.
- Score with Groq itself as the judge (a second call with a strict rubric prompt).
- Track **retrieval hit-rate @k** separately — did the correct page appear in the retrieved
  set at all? That one number, measured before and after each change above, tells you
  whether a change helped. It needs no LLM judge and no subjective scoring.

Build this *before* Tier 3, not after — otherwise you can't tell whether the reranker helped.

### 4.2 `top_k` is unbounded and untrusted

`ChatRequest.top_k: int = TOP_K` accepts anything a client sends. `top_k=5000` builds an
enormous prompt and either blows the model's context limit or burns a lot of tokens. Use
Pydantic bounds: `top_k: int = Field(TOP_K, ge=1, le=20)`. Also cap the *assembled context*
by characters/tokens before sending it, independent of `top_k`.

### 4.3 The Groq call has no error handling

`generate_answer()` calls `chat.completions.create()` bare. A rate limit, a deprecated model
id, or a network blip becomes an unhandled exception and a 500 with a stack trace. Wrap it,
log the real error, and return a readable message — you already do exactly this for the
missing-API-key case, so the pattern is established.

### 4.4 No streaming

Groq's throughput is its main selling point and you're hiding it — the user stares at
"Thinking..." until the whole answer is done. `stream=True` plus an SSE endpoint and an
`EventSource` in `script.js` makes the app feel several times faster without being faster.

### 4.5 Conversation history

Already in PLAN.md, restating for completeness: pass the last N Q&A pairs into the prompt.
Important subtlety — use the *conversation-aware rewritten question* for retrieval, not the
raw follow-up. "What about the second one?" embeds to nothing useful; rewrite it to a
standalone question with a cheap LLM call first, then retrieve.

### 4.6 Optional: agentic routing

The reference's `agenticrag/` notebook gates retrieval behind a decision node — for
greetings or general-knowledge questions, skip retrieval entirely. Worth ~1 line of value
here (a cheap classifier call before retrieving) and mostly interesting as a learning
exercise. Lower priority than everything above; a distance floor (3.4) covers most of the
practical benefit.

### 4.7 OCR fallback

Already in PLAN.md. Now that PyMuPDF is in place, it can rasterize a page
(`page.get_pixmap()`) and hand it to `pytesseract` when `get_text()` returns empty — the
plumbing is one function.

---

## Suggested order of work

1. Fix the truncation (1.1) and re-ingest — biggest quality win, smallest change.
2. Build the eval set and hit-rate metric (4.1) — so everything after this is measurable.
3. Structure-aware chunking (1.2) + page ranges (1.3).
4. Split ingestion into a CLI + background jobs (2.1), batching (2.2).
5. Re-ranker (3.1), then hybrid BM25 (3.2) — measuring each against 4.1.
6. Per-document scoping (3.3), neighbor expansion (3.5), streaming (4.4).

Items 4.2 and 4.3 are ten-minute hardening fixes; do them whenever.
