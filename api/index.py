"""
Vercel's Python runtime entrypoint. It looks for an ASGI/WSGI application object named
`app` in api/<name>.py and routes requests to it per vercel.json - this file's only job is
to re-export the real application built in src/main.py, so there is exactly one FastAPI app
definition in the whole codebase (the one that also runs locally with `uvicorn src.main:app`).

Verify this file's shape against Vercel's current Python runtime documentation before
relying on it in production - their entrypoint convention has changed across runtime
versions, and this project has not been deployed against a live Vercel account yet.
"""
from src.main import app  # noqa: F401
