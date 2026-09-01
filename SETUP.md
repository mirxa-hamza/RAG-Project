# Setup guide — getting this running on your machine

Everything runs locally except the answer generation, which calls Groq. Embedding,
search and re-ranking all happen on your CPU, so there's nothing to pay for there.

Expect the first run to take a few minutes: it downloads two models (~210MB total) and
then indexes whatever PDFs you put in `data/`.

---

## 1. Prerequisites

- **Python 3.10+** (3.13 is what we develop on) — `python --version`
- **Git**
- **MongoDB running locally** on `mongodb://localhost:27017` — it stores user accounts
  (nothing else). Windows: install MongoDB Community Server and it runs as a service, so
  `net start MongoDB` in an admin terminal is usually all that is needed. With Docker:
  `docker run -d -p 27017:27017 --name mongo mongo`. Check it with `mongosh` or by
  visiting <http://localhost:8000/api/health/auth> once the app is up.
- **A free Groq API key** — make your own at <https://console.groq.com/keys>.
  Don't reuse anyone else's; keys are personal and rate-limited per account.
- ~2GB of free disk (PyTorch is the bulk of it) and a normal internet connection for
  the first run.

---

## 2. Clone and install

```bash
git clone <REPO_URL>
cd "My RAG Project"

python -m venv venv

# Windows (PowerShell):
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

If `venv\Scripts\activate` is blocked on Windows PowerShell, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal first.

---

## 3. Configure your key

```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
```

Open `.env` and set your own key:

```
GROQ_API_KEY=gsk_your_own_key_here
```

Everything else in `.env` has a sensible default — leave it alone unless you're
deliberately experimenting. **`.env` is gitignored; never commit it.**

---

## 4. Create your account

Start the app (step 5), open it, and press **Create an account**. Signup is open, so pick a
password you would not mind typing again — there is no reset flow.

The **first** account is special in one way: it inherits every document that was indexed
before accounts existed, and it becomes the owner of anything you copy into `data/` by hand
afterwards. Every later account starts empty and sees only what it uploads.

Sessions last 12 hours (`JWT_EXPIRE_HOURS`). The signing key is generated on first run and
appended to `.env`; changing or deleting it signs everyone out.

---

## 4b. Add documents

The repo deliberately ships **no PDFs** — they're large, often copyrighted, and change
independently of the code.

Two ways to add them. The easy one, once the server is running: press **Add documents** in
the sidebar and drag PDFs into the window that opens (or click to browse). Each file shows
its own progress — uploading, then reading pages, then indexing — and only joins *Your
documents* once it is fully indexed and queryable. You can close that window while it
works; the sidebar keeps reporting. Up to 100MB per file (`MAX_UPLOAD_MB` in `.env`).

The other way, and the one to use for a big batch, is to put the files into `data/`
yourself:

```
data/
├── some-book.pdf
└── textbooks/            ← subfolders work too
    └── another.pdf
```

Then build the index:

```bash
python scripts/ingest.py
```

This extracts text, chunks it, embeds every chunk locally, and stores it. A 900-page
textbook takes a few minutes on CPU — the log prints per-file progress, it isn't hung.

---

## 5. Run it

```bash
python scripts/run.py
```

This starts the server and opens the browser **once it actually answers**, so you never
land on `ERR_CONNECTION_REFUSED` during the first seconds of a cold start. Plain uvicorn
works exactly as before if you prefer it:

```bash
uvicorn src.main:app --port 8000        # add --reload only while editing code
```

Open **<http://localhost:8000>** — the web UI is served by the same process, so there's
no separate HTML file to open. API docs are at `/docs`.

The server also runs an ingestion pass in the background at startup, so step 4 is
optional; doing it up front just means the app is queryable the moment it boots.

---

## Working with documents day to day

| What you want | Command |
|---|---|
| Index new or changed PDFs | `python scripts/ingest.py` |
| See what's currently indexed | `python scripts/ingest.py --status` |
| Wipe and rebuild everything | `python scripts/ingest.py --force` |
| Same, without leaving the browser | the **Sync documents** button in the sidebar |
| Add a PDF from the browser | **Add documents** in the sidebar, then drop or browse |
| Remove a document | the bin icon on its card — deletes its passages **and** the PDF |

Files are fingerprinted by SHA-256, so re-running is always safe and cheap:

- **new file** → indexed
- **unchanged file** → skipped instantly (no re-embedding)
- **edited file** → old chunks deleted, re-indexed
- **deleted file** → removed from the index automatically
- **corrupt file** → reported as `failed`, everything else still indexes

You only need `--force` if you change the embedding model or the chunk settings in
`.env`, since the stored vectors are only valid for the settings that produced them.

---

## Why the first start is slow (and how to make later ones fast)

A cold start on a machine with two big textbooks in `data/` looks like this:

| Phase | Time | Happens again? |
|---|---|---|
| Import torch + load the embedding model | ~10–20s | every start |
| Extract and chunk the PDFs | ~20s | only for new/changed files |
| Embed ~3,200 chunks on CPU | 8–9 min | only for new/changed files |

The server is answering requests during the whole embedding phase — it runs in a
background thread, and the loading screen shows its progress. The second start of the same
corpus takes seconds, because every file is fingerprinted with SHA-256 and skipped when
unchanged.

To make it quicker anyway:

- **Index before you serve**: `python scripts/ingest.py`, then start uvicorn. Same total
  work, but the server comes up already queryable.
- **Drop `--reload` when you're not editing code.** The reloader starts a second process
  and a file watcher over the whole project, which roughly doubles the boot cost.
- **Nothing else is worth tuning.** Embedding is CPU-bound matrix maths; batch size
  changes it by a few percent, not by minutes. A CUDA GPU would cut it to well under a
  minute — nothing else will.

## Verify it works

```bash
pip install -r requirements-dev.txt
python tests/test_pipeline_offline.py
```

175 checks, fully offline — no API key and no model download needed, because it stubs the
models — and no MongoDB either, because the users collection is faked. Run this before
pushing anything; the multi-tenant isolation checks are in here.

To measure *answer quality* rather than plumbing:

```bash
python eval/run_eval.py            # retrieval hit-rate@k, MRR, refusal rate
python eval/run_eval.py --judge    # + LLM-as-judge scoring (uses your Groq key)
```

---

## Things worth knowing

- **The vector store is not in the repo.** `storage/` is gitignored. It's derived data —
  rebuilt from `data/` in minutes — and it's large, binary, and unmergeable. You build
  your own with `scripts/ingest.py`.
- **First question is slower.** The cross-encoder re-ranker (~80MB) downloads lazily on
  the first question you ask, then it's cached.
- **If a question isn't covered by the documents**, the app says so instead of guessing.
  That's the relevance floor doing its job, not a bug.
- **Answers cite pages.** If a citation looks off by a page, chunks can span a page
  boundary — that's a known approximation, documented in CLAUDE.md.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the **project root**, not from inside `src/`. |
| `ModuleNotFoundError: rank_bm25` etc. | `pip install -r requirements.txt` again — the venv is stale. |
| Answer says `GROQ_API_KEY is not set` | `.env` is missing or the key line is empty; restart the server after editing it. |
| `model not found` from Groq | Groq retires models; check <https://console.groq.com/docs/models> and update `GROQ_MODEL` in `.env`. |
| Sidebar shows no documents | No PDFs in `data/`, or ingestion hasn't run — click **Sync documents**. Remember documents are per-account: another account's uploads are invisible by design. |
| Sign-in says the server can't be reached | MongoDB is down. Start it, then check <http://localhost:8000/api/health/auth>. |
| Everyone was signed out after a restart | `JWT_SECRET` changed or was cleared in `.env`. |
| A document you uploaded vanished | You are signed in as a different account than the one that uploaded it. |
| `Numpy built with MINGW-W64` warning | Windows-on-ARM only, harmless. |

More detail: `README.md` for the architecture, `CLAUDE.md` for conventions and known
gotchas, `PLAN.md` for the roadmap.
