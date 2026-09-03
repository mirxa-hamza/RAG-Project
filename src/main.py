"""
Application entry point: builds the FastAPI app, wires the router, and serves the UI.

Run from the project root:

    uvicorn src.main:app --reload --port 8000

Then open http://localhost:8000 - the web UI is served by this same process from
src/static/, so there is no separate file to open and no CORS hop in normal use.

Every route that touches documents requires a signed-in user (see src/api/deps.py), and
every document is owned by the account that uploaded it. Documents are still only ever
indexed from the data folder - /upload writes the file there first, under
data/users/<user_id>/, and the ordinary ingestion job picks it up.

Ingestion runs as a background job, so the server answers requests immediately even while
a 900-page book is still being embedded.
"""
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
# Starlette's own HTTPException - StaticFiles raises that one, not FastAPI's subclass.
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api import api_router
from src.core.config import CORS_ORIGINS, IS_CLOUD, STATIC_DIR
from src.core.logging import get_logger, new_request_id, quiet_access_log, request_id
from src.ml import embeddings
from src.services import database, ingestion, sessions

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

    # Fail loudly and early if Mongo is down: without it nobody can sign in, and a server
    # that accepts connections but rejects every login looks broken in a much more
    # confusing way than one that says what is wrong at startup.
    try:
        await database.ping()
        await database.ensure_indexes()
        await sessions.ensure_indexes()
        log.info("MongoDB is reachable; the users and chat-history indexes are in place.")
    except database.DatabaseUnavailable as exc:
        log.error("%s", exc)
        log.error("The app will start, but signing in will fail until MongoDB is running.")
    except Exception:
        log.exception("Could not prepare MongoDB indexes; sign-in may be unreliable.")
    # Load the embedding model off the critical path. Doing it at import time kept the
    # port closed for ~18s, so opening the link too early gave ERR_CONNECTION_REFUSED
    # rather than the loading screen.
    threading.Thread(target=embeddings.warm_up, name="warm-up", daemon=True).start()
    # Kick ingestion off in the background - the API is up immediately, and new, changed
    # or deleted PDFs are reconciled while it serves requests. Local-mode only: it scans
    # the whole DATA_DIR unscoped to a user. Cloud-mode ingestion is per-user and driven by
    # /upload or /ingest with an authenticated caller - start_job() with no user_id raises
    # there, and an unhandled exception here fails the whole ASGI lifespan startup, which
    # crashed every cold start on Vercel.
    if not IS_CLOUD:
        ingestion.start_job()
    yield
    database.close()


app = FastAPI(
    title="Document Q&A",
    description="A from-scratch RAG API over the PDFs in the backend's data folder.",
    version="3.1",
    lifespan=lifespan,
)

@app.middleware("http")
async def request_context(request: Request, call_next):
    """
    Tags every log line emitted while serving one request with the same id, and returns it
    in `X-Request-ID` so a user reporting "it failed" can hand you the exact id to grep.
    """
    token = request_id.set(request.headers.get("X-Request-ID") or new_request_id())
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id.get("")
        return response
    finally:
        request_id.reset(token)


@app.exception_handler(database.DatabaseUnavailable)
async def database_unavailable(request: Request, exc: database.DatabaseUnavailable):
    """
    503 with an actionable sentence, not a 500 with a stack trace.

    A database that is simply not running is an operational state, not a bug in the
    request - and the person reading the message is usually the one who can fix it.
    """
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# The UI is served by this process, so cross-origin access is not needed at all by default
# and the middleware is only added when CORS_ORIGINS names somewhere specific. The previous
# allow_origins=["*"] let any website in the world call this API.
if CORS_ORIGINS:
    log.info("CORS enabled for: %s", ", ".join(CORS_ORIGINS))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
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
