"""Dependency providers.

These are the composition root and the override points for tests. Real
providers read settings and construct Gemini/fastembed/Pinecone-backed services; tests
override them via ``app.dependency_overrides``.
"""

from __future__ import annotations

from functools import lru_cache

from app.agent.nodes.planner import Planner
from app.agent.orchestrator import AgentOrchestrator
from app.agent.tools.base import Tool
from app.agent.tools.vector_search import VectorSearchTool
from app.agent.tools.web_search import TavilySearchTool
from app.core.config import Settings, get_settings
from app.services.document_store import DocumentStore
from app.services.ingestion.embeddings import Embedder, FastEmbedEmbeddings
from app.services.ingestion.service import IngestionService
from app.services.llm import GeminiLLM, LLMClient
from app.services.retrieval.answerer import Answerer
from app.services.retrieval.query_service import QueryService
from app.services.vector_store import PineconeVectorStore, VectorStore
from app.services.verification.grounding import GroundingVerifier
from app.services.verification.service import VerificationService


@lru_cache
def get_document_store() -> DocumentStore:
    return DocumentStore()


@lru_cache
def get_embedder() -> Embedder:
    s: Settings = get_settings()
    return FastEmbedEmbeddings(
        model=s.embedding_model,
        cache_dir=s.embedding_cache_dir,
        batch_size=s.embedding_batch_size,
    )


@lru_cache
def get_vector_store() -> VectorStore:
    s: Settings = get_settings()
    return PineconeVectorStore(
        api_key=s.pinecone_api_key,
        index_name=s.pinecone_index,
        dimension=s.embedding_dimension,
        cloud=s.pinecone_cloud,
        region=s.pinecone_region,
    )


def get_ingestion_service() -> IngestionService:
    return IngestionService(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        document_store=get_document_store(),
        settings=get_settings(),
    )


@lru_cache
def get_llm() -> LLMClient:
    s: Settings = get_settings()
    return GeminiLLM(
        api_key=s.gemini_api_key,
        model=s.gemini_model,
        thinking_level=s.gemini_thinking_level,
    )


def get_answerer() -> Answerer:
    return Answerer(llm=get_llm(), max_tokens=get_settings().gemini_max_tokens)


def get_query_service() -> QueryService:
    return QueryService(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        answerer=get_answerer(),
        settings=get_settings(),
    )


@lru_cache
def get_tools() -> list[Tool]:
    s: Settings = get_settings()
    return [
        VectorSearchTool(
            embedder=get_embedder(),
            vector_store=get_vector_store(),
            top_k=s.retrieval_top_k,
        ),
        TavilySearchTool(api_key=s.tavily_api_key, max_results=s.web_search_max_results),
    ]


def get_planner() -> Planner:
    return Planner(llm=get_llm())


def get_orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator(
        planner=get_planner(),
        tools=get_tools(),
        answerer=get_answerer(),
        settings=get_settings(),
    )


def get_verification_service() -> VerificationService:
    return VerificationService(grounding_verifier=GroundingVerifier(llm=get_llm()))
