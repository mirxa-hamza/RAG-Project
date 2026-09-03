"""Aggregate API router. New phase routers are registered here."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import agent, documents, health, query

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(query.router)
api_router.include_router(agent.router)
