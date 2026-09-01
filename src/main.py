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
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
# Starlette's own HTTPException - StaticFiles raises that one, not FastAPI's subclass.
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api import api_router
from src.core.config import STATIC_DIR
from src.core.logging import get_logger, quiet_access_log
from src.ml import embeddings
from src.services import ingestion

log = get_logger(__name__)


class SPAStaticFiles(StaticFiles):
    """
    StaticFiles, but a path a browser asked for that does not exist renders index.html
    instead of {"detail":"Not Found"}.

    A stray or stale URL used to show a raw JSON 404, which reads as "the app is broken".
    Non-HTML clients still get a real 404, so a mistyped API path never silently returns
    a page instead of an error.
    """

    async def get_response(self, path: str, scope):
        # Starlette signals a missing file by raising HTTPException, not by returning a
        # 404 response, so both shapes are handled here.
        try:
            response = await super().get_response(path, scope)
            if response.status_code != 404:
                return response
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
        if "text/html" in Headers(scope=scope).get("accept", ""):
            return await super().get_response("index.html", scope)
        raise StarletteHTTPException(status_code=404)

    def file_response(self, *args, **kwargs):
        """
        Tell the browser to revalidate HTML on every load.

        Without this, Chrome happily served a cached index.html against a freshly updated
        style.css, which rendered new markup with old rules - the app looked broken until a
        hard refresh. CSS and JS are cache-busted by a ?v= query in the HTML instead, so
        they can still be cached hard.
        """
        response = super().file_response(*args, **kwargs)
        if response.media_type == "text/html":
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The UI polls /ingest/status and /stats while indexing runs; keep those out of the
    # access log so real progress stays readable.
    quiet_access_log()
    # Load the embedding model off the critical path. Doing it at import time kept the
    # port closed for ~18s, so opening the link too early gave ERR_CONNECTION_REFUSED
    # rather than the loading screen.
    threading.Thread(target=embeddings.warm_up, name="warm-up", daemon=True).start()
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
    app.mount("/", SPAStaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    log.info("Serving the web UI from %s at /", STATIC_DIR)
else:
    log.warning("Static directory %s not found - the web UI will not be served.", STATIC_DIR)
