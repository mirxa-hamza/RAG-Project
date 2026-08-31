"""
HTTP layer: one module per group of endpoints, aggregated into a single router.

Route handlers here stay thin - they validate, call a service, and shape the response.
The pipeline itself lives in src/services/ and src/ml/.
"""
from fastapi import APIRouter

from src.api import chat, documents, system

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)

__all__ = ["api_router"]
