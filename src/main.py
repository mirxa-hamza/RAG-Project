"""
Application entry point: builds the FastAPI app, wires the router, and serves the UI.

Run from the project root:

    uvicorn src.main:app --reload --port 8000

Then open http://localhost:8000 - the web UI is served by this same process from
src/static/, so there is no separate file to open and no CORS hop in normal use.

Documents come ONLY from the backend's data folder (see src/core/config.py) - there is no
upload endpoint, so a user on the frontend can ask questions but can never add or change
what's in the vector store.

Ingestion runs as a background job, so the server answers requests immediately even while
a 900-page book is still being embedded.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api import api_router
from src.core.config import STATIC_DIR
from src.core.logging import get_logger
from src.services import ingestion

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick ingestion off in the background - the API is up immediately, and new, changed
    # or deleted PDFs are reconciled while it serves requests.
    ingestion.start_job()
    yield


app = FastAPI(
    title="Document Q&A",
    description="A from-scratch RAG API over the PDFs in the backend's data folder.",
    version="3.1",
    lifespan=lifespan,
)

# The UI is same-origin now, so CORS is only needed if you point another origin (a dev
# server, a separate front end) at this API. Tighten this before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes are registered BEFORE the static mount below, so /chat, /stats and friends
# always win over a same-named file in src/static/.
app.include_router(api_router)

if STATIC_DIR.is_dir():
    # html=True serves index.html at "/" and handles directory requests. Mounting at the
    # root has to come last, after every API route.
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    log.info("Serving the web UI from %s at /", STATIC_DIR)
else:
    log.warning("Static directory %s not found - the web UI will not be served.", STATIC_DIR)
