# CLOUD_MIGRATION_PLAN.md

## Implementation status (2026-09-02)

Steps 1-4 below are implemented. Summary of what actually landed, and where it differs
from the plan as originally written:

- **Step 1 (Cloudinary uploads) — done.** `src/services/cloudinary_store.py` (new) talks to
  Cloudinary's REST API directly via `httpx` with hand-rolled SHA-1 signing, rather than
  adding the `cloudinary` SDK as a dependency — kept in line with this project's "no hidden
  abstractions" rule. `POST /upload/sign` and `POST /upload/complete` were added to
  `src/api/documents.py` exactly as planned, including the magic-byte/size/quota checks the
  plan called out as "must not be skipped" (`uploads.validate_uploaded_bytes()`, shared with
  local mode's `save_pdf()`). The ownership check the plan worried about — a caller trying to
  point `/upload/complete` at someone else's Cloudinary file — is
  `cloudinary_store.public_id_belongs_to()`, and it has dedicated test coverage, including
  the "starts with the folder name as a string but isn't actually inside it" edge case.
  File discovery in `src/services/ingestion.py` is backed by the new
  `src/services/cloud_documents.py` Mongo registry, keyed by the same `users/<id>/name.pdf`
  path identity local mode uses, so ownership/dedupe/pruning needed no divergent logic.
- **Step 2 (state that survives a cold start) — done, with one deliberate deviation.** The
  plan suggested the rate limiter and answer cache "become async." They did not: the new
  Mongo-backed code in `ratelimit.py`, `answer_cache.py`, and `manifest.py` uses a
  **synchronous** `pymongo.MongoClient` (`database.sync_collection()`, new), not `motor`.
  Reason: call sites span both the local background thread (`ingestion.py`'s `_run`, which
  has no event loop to await into) and async request handlers, and a full async refactor of
  the ingestion job wasn't worth doing alongside everything else in this migration. The
  tradeoff is accepted latency on those Mongo round-trips; it does not change correctness,
  and this app is documented single-worker/non-concurrent already (see Deployment
  invariants in `CLAUDE.md`).
- **Step 3 (resumable ingestion) — done.** `POST /ingest/continue` does one file's worth of
  work per call, backed by a per-user `ingestion_jobs` Mongo document
  (`ingestion.continue_job()`). `main.py`'s `lifespan` now skips the startup background scan
  and the embedding warm-up thread entirely when `IS_CLOUD` is true. `src/static/script.js`'s
  polling loop was updated in the same change (`pollUntilDone()` now calls
  `/ingest/continue` instead of `/ingest/status` when `document_store === "cloudinary"`).
- **Step 4 (Vercel scaffolding) — done.** `requirements-cloud.txt`, `api/index.py`,
  `api/requirements.txt` (a copy, since `@vercel/python` looks for `requirements.txt` next
  to the entrypoint, not at the project root), and `vercel.json` (`maxDuration: 60`, routing
  everything to `api/index.py`) all exist now.
- **Result-ordering bug found and fixed during implementation, not in the original plan**:
  the first cut of the shared dedupe/plan helper split "already decided" (skip/duplicate/
  already-stored) results from "pending" (needs-ingest) results into two separate lists,
  appended one after the other. That reorders `ingest_data_folder()`'s results relative to
  the original filename-scan order whenever a pending file sorts before a decided one (ASCII:
  space < period, so `"book (2).pdf"` sorts before `"book.pdf"`), which broke the existing
  "byte-identical content is skipped as a duplicate" test. Fixed by having
  `ingestion._plan_ingest()` return one ordered list with inline `{"pending": True, ...}`
  markers instead, so callers reproduce the original single-pass interleaving. Both the
  local-mode and cloud-mode job builders now go through this one function.
- **What is NOT verified**: none of this was run against live Cloudinary, Pinecone, Chroma
  Cloud, or MongoDB Atlas credentials — there weren't any available in this environment. Every
  new piece of Mongo-backed state (rate limiter, answer cache, manifest, cloud document
  registry) and the Cloudinary ownership check were tested against `tests/test_pipeline_offline.py`'s
  existing fake-Mongo pattern (`FakeSyncCollection`, extending `database.set_sync_collection()`
  the same way `database.set_users_collection()` already worked). All 320 checks in that
  suite pass. Before a real Vercel deployment: do one manual smoke test end-to-end with real
  keys (sign up, upload a PDF, watch it ingest via `/ingest/continue`, ask a question) — the
  plan's own "what you can test right now" section below is still a good way to validate the
  Pinecone/Chroma Cloud/Atlas leg of this independently of Cloudinary.

---

Concrete plan for taking this app from "runs great on my machine" to "runs on Vercel",
using the stack you asked for: **Cloudinary** for PDFs, **Chroma Cloud** (not Pinecone) for
vector storage, **Pinecone** for embeddings and re-ranking only. Written after reading the
actual code as of 2026-09-02, not from `DEPLOY.md`, which is stale in places — see the
correction note below before following it.

## Correction to DEPLOY.md first

`DEPLOY.md`'s "Still to do" list (items 3-6) undersells how far this has already come, and
its Part A env var list assumes an all-Pinecone stack, which is not what you're building.
Concretely, as of this codebase:

- **Already built, contrary to DEPLOY.md:** the embeddings/reranker provider switch
  (`EMBEDDINGS_PROVIDER`, `RERANKER_PROVIDER` in `src/core/config.py`), the Pinecone vector
  store (`src/services/vector_pinecone.py`), **and a Chroma Cloud backend**
  (`src/services/vector_chroma.py`, gated by `CHROMA_BACKEND=cloud` +
  `CHROMA_API_KEY`/`CHROMA_TENANT`/`CHROMA_DATABASE`). That last one is exactly what you
  asked for — remote Chroma, not Pinecone, for vectors — and it needs **zero new code**,
  only env vars. Also fully built: MongoDB-backed multi-tenant auth (`src/services/database.py`,
  `security.py`, `ownership.py`), which DEPLOY.md doesn't mention at all.
- **Not built, and DEPLOY.md is right about these:** Cloudinary. There is no code for it —
  grep for "cloudinary" in `src/` and the only hit is a comment in `config.py`. No env vars,
  no client, no upload flow. `src/services/uploads.py` still only ever writes to
  `DATA_DIR` on local disk (`os.fdopen` + `os.replace`).
- **Not built:** every piece of in-process state DEPLOY.md flagged is still in-process
  today — `src/services/ingestion.py`'s `_state` dict (job progress) and its
  `threading.Thread`-based background job (`main.py`'s `lifespan` calls
  `ingestion.start_job()` at startup, unconditionally), `src/core/ratelimit.py`'s `_events`
  dict, `src/services/answer_cache.py`'s in-memory LRU. None of it survives a Vercel cold
  start or is shared across concurrent function instances.
- **Better than DEPLOY.md implied:** BM25 (`src/services/bm25.py`) already builds
  **per-user**, not corpus-wide, indices — 0.09s-2.95s to rebuild depending on corpus size,
  per the module's own docstring. That's a cold-start latency cost, not a correctness or
  hang risk, so it's downgraded from "must fix" to "acceptable, revisit if it's slow in
  practice."
- **Doesn't exist yet:** `requirements-cloud.txt`, `api/index.py`, `vercel.json`. Nothing in
  the repo root targets Vercel's entrypoint convention at all.

## Target architecture (your stack)

| Piece | Setting |
|---|---|
| PDFs | Cloudinary (new — see Step 1) |
| Vectors | Chroma Cloud — `VECTOR_STORE=chroma`, `CHROMA_BACKEND=cloud` (already built) |
| Embeddings | Pinecone — `EMBEDDINGS_PROVIDER=pinecone` (already built) |
| Re-ranking | Pinecone — `RERANKER_PROVIDER=pinecone` (already built) |
| Accounts | MongoDB Atlas (already built) |
| Job state / rate limits / answer cache | MongoDB (new — see Step 2) |

Note Pinecone is still in play here for two things (embeddings, rerank) even though vectors
live in Chroma Cloud instead — that's a supported combination already, not a hack. You will
still need a `PINECONE_API_KEY`.

## Step 1 — Cloudinary uploads (the one Vercel forces on you)

Why it's not optional: Vercel rejects any request body over 4.5MB, and `MAX_UPLOAD_MB=100`.
The browser has to upload straight to Cloudinary and only tell the app the resulting URL —
the server must never see the raw bytes come in over `POST /upload` the way it does today.

Concrete changes:

1. **New `src/services/cloudinary_store.py`.** Wraps the Cloudinary SDK: a function that
   mints a signed upload payload (folder scoped to `users/<user_id>/`, resource_type=raw so
   PDFs aren't run through image processing) for the browser to POST directly to
   Cloudinary, and a function that reads bytes back by public_id for ingestion to consume.
2. **New endpoint, `POST /upload/sign`,** in `src/api/documents.py` — authenticated, returns
   the signature/timestamp/folder the frontend needs. Add `CLOUDINARY_UPLOAD` to
   `src/core/ratelimit.py` so signing itself is throttled (a stolen signature is a stolen
   upload slot, so keep the signature short-lived — a few minutes).
3. **New endpoint, `POST /upload/complete`,** — the frontend calls this with the Cloudinary
   `public_id`/URL after its direct upload succeeds. This is where you keep the existing
   guarantees from `src/services/uploads.py` (`%PDF-` magic byte check, per-account storage
   cap) — just checked against the fetched bytes/Cloudinary's reported size instead of a
   local stream. **This is the step that must not be skipped**: nothing in `uploads.py`
   today validates content type by magic bytes when the file arrives by URL instead of by
   stream, and skipping that check re-opens the exact hole `uploads.py`'s docstring says it
   closed (arbitrary file arrives disguised as a PDF).
4. **`src/services/ingestion.py`'s file discovery** currently walks `DATA_DIR` on disk
   (`Path.rglob` or similar — the "`data/` is scanned recursively" convention in
   `CLAUDE.md`). In cloud mode this has no folder to walk. Replace the local-mode scan with
   a Cloudinary-mode listing (a Mongo collection of `{user_id, public_id, url, uploaded_at,
   sha256}` records written by `/upload/complete`, since Cloudinary's own list API is not
   the source of truth for "what does this account own"). `ingest_one()` then fetches bytes
   by URL instead of `open()`ing a local path.
5. **`src/services/uploads.py`'s `delete_document`** needs a Cloudinary-mode branch that
   calls the SDK's destroy instead of `os.remove`.

This step is the largest change in the whole migration and the one with the most security
surface (anyone who can call `/upload/complete` with an arbitrary Cloudinary URL is trying
to make your ingestion pipeline fetch and embed someone else's file) — budget real time for
it, and make sure the isolation tests (`tests/test_pipeline_offline.py`) grow a case for
"an authenticated user calls /upload/complete with a public_id/folder that isn't theirs."

## Step 2 — state that survives a cold start

Three separate in-process stores need one shared home. MongoDB is already a hard dependency
(accounts), so it's the natural place — no new service to provision.

1. **Rate limiter (`src/core/ratelimit.py`).** Replace the `_events` dict with a Mongo
   collection (`rate_limit_events`), one document per event, indexed on `(bucket, key,
   timestamp)`, with a TTL index so old events expire themselves instead of the manual
   `_sweep()`. `check()`'s signature can stay the same — this is an internal swap, not an
   API change, though it becomes `async` since every Mongo call in this codebase goes
   through the `database._guard()` pattern.
2. **Answer cache (`src/services/answer_cache.py`).** Same shape: a Mongo collection keyed
   by `(user_id, question_hash, generation)`, with the existing TTL logic driving a TTL
   index instead of an in-memory sweep. The generation-counter invalidation logic (an
   entry from a stale generation is ignored) carries over unchanged.
3. **Ingestion job state (`src/services/ingestion.py`'s `_state`).** This one is the
   trickiest because it's not just "move a dict to Mongo" — it's also *who resumes it*,
   which Step 3 covers. For now: `_state` becomes a document per running job
   (`ingestion_jobs` collection), and `job_status(user_id)` reads from Mongo instead of the
   module-level dict.

Do this step before Step 3 (resumable ingestion) — you need somewhere durable to record
"how far did we get" before you can build a request that resumes from it.

## Step 3 — ingestion without a background thread

`main.py`'s `lifespan` calling `ingestion.start_job()` on startup, and
`ingestion.py`'s `threading.Thread(target=_run, ...)`, both assume the process stays alive
between requests. A Vercel function is killed the moment the response is sent — there is no
"meanwhile, in the background."

The redesign: ingestion becomes **one slice of work per request**, matching what
`DEPLOY.md` originally sketched:

1. `POST /ingest` (or `/upload/complete`, for the single-file case) writes a job document
   in Mongo (from Step 2) listing the files still to process, and does **one file's worth**
   of work — extract, chunk, embed, upsert — before returning.
2. The response includes whether more work remains. The frontend's existing polling loop
   (`GET /ingest/status`, already built for the progress bar) becomes what *drives* the next
   slice too, not just what reports on one — i.e. the frontend calls `/ingest/continue`
   (new) in a loop until the job document says `done`.
3. `main.py`'s `lifespan` startup call to `ingestion.start_job()` needs to become
   conditional on `RAG_MODE` — in cloud mode there is no long-lived process to kick a
   background scan off in, so startup-time ingestion goes away entirely in favor of the
   request-driven model above. Same for the `embeddings.warm_up()` thread: in cloud mode
   there's no local embedding model to warm up at all (Pinecone does that work), so this
   whole block should be skipped when `IS_CLOUD`.
4. **Interruption safety already exists and should carry over as-is**: `ingest_one()`
   calling `delete_source()` for `new` files too (not just `changed`), so a slice that gets
   cut off mid-file doesn't leave orphaned chunks. Keep that invariant when you refactor
   this into a per-request slice — it's the thing that makes "the function got killed
   mid-file" a retry instead of a duplicate-chunk bug.

This is the single largest architectural change after Cloudinary. It touches the contract
the frontend polls against, so update `src/static/script.js`'s polling logic in the same
change, not after.

## Step 4 — Vercel scaffolding

None of this exists yet:

1. **`requirements-cloud.txt`** — your current `requirements.txt` minus `torch`,
   `sentence-transformers`, and anything else that only serves the local-model path
   (`rank-bm25` stays; it's still used per-user even in cloud mode). Vercel's bundle size
   limit is the reason this has to be a separate file rather than an if-branch in the
   normal one.
2. **`api/index.py`** — the ASGI entrypoint Vercel's Python runtime expects
   (`from src.main import app` re-exported, roughly). Confirm against Vercel's current
   Python runtime docs before writing this — their entrypoint convention has changed
   versions before and DEPLOY.md doesn't specify one.
3. **`vercel.json`** — routes everything to the one function, sets the Python runtime
   version, and should set a generous `maxDuration` for `/ingest/continue` given it still
   does real embedding work per slice.

## Step 5 — env vars for this exact stack

Once Steps 1-4 land, this is what actually goes in Vercel's dashboard (Production +
Preview):

```dotenv
RAG_MODE=cloud

GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b

EMBEDDINGS_PROVIDER=pinecone
RERANKER_PROVIDER=pinecone
PINECONE_API_KEY=...
PINECONE_API_VERSION=2025-10

VECTOR_STORE=chroma
CHROMA_BACKEND=cloud
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...

CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...

MONGO_URI=mongodb+srv://...
MONGO_DB=rag_app
JWT_SECRET=<64 random hex chars, generated by you, never by the app>
```

`VECTOR_STORE` and `CHROMA_BACKEND` are the two lines that don't exist in `DEPLOY.md` at
all — they're what steers vectors to Chroma Cloud instead of the Pinecone default.
`CLOUDINARY_*` needs the three new config settings added to `src/core/config.py` (they
don't exist yet — only the docstring mentions them).

## Suggested order of work

1. Step 2 (Mongo-backed rate limiter + answer cache) — smallest, least risky, no frontend
   changes, and it unblocks nothing else so it's safe to do first and merge on its own.
2. Step 1 (Cloudinary) — biggest single piece, does the most to make local testing of "cloud
   mode" meaningful (right now you can point cloud mode at Pinecone/Chroma Cloud/Atlas
   today and it'll work for everything except uploads, since those still hit local disk).
3. Step 3 (resumable ingestion) — depends on Step 2's job-state collection existing.
4. Step 4 (Vercel scaffolding) — do this last, once there's something that actually needs
   to run on Vercel to test against.

At each step, run `tests/test_pipeline_offline.py` (extend it — it already stubs Mongo and
the embedding model, so a Cloudinary-mode and Mongo-state-mode set of isolation checks fits
the existing pattern) and re-run `eval/run_eval.py` per `CLAUDE.md`'s existing rule if
anything in the retrieval path changes.

## What you can test right now, before any of this

Since embeddings/reranker/vector-store provider switching already exists, you can validate
Pinecone (embeddings+rerank) + Chroma Cloud (vectors) + Atlas today, locally, by setting
`RAG_MODE=cloud`, `VECTOR_STORE=chroma`, `CHROMA_BACKEND=cloud` and the corresponding keys,
and running `python scripts/run.py` as normal. Everything will work except uploading new
PDFs through the browser (still hits local `DATA_DIR`) — hand-copying a PDF into `data/` and
running `python scripts/ingest.py` sidesteps that until Step 1 lands.
