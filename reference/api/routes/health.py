"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", environment=settings.environment, version="0.1.0")


@router.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """Readiness: verifies required credentials are configured.

    Deeper checks (Pinecone reachability, etc.) are added in later phases.
    """
    settings = get_settings()
    checks = {
        "gemini_key": bool(settings.gemini_api_key),
        "pinecone_key": bool(settings.pinecone_api_key),
    }
    return ReadyResponse(ready=all(checks.values()), checks=checks)
