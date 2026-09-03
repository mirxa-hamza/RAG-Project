"""Agentic query endpoint (orchestrated multi-tool RAG)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.agent.orchestrator import AgentOrchestrator
from app.api.deps import get_orchestrator, get_verification_service
from app.schemas.agent import AgentQueryRequest, AgentQueryResponse
from app.schemas.query import CitationOut, SourceOut
from app.schemas.verification import VerificationOut
from app.services.retrieval.context import AssembledContext
from app.services.verification.service import VerificationService

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/query", response_model=AgentQueryResponse)
async def agent_query(
    request: AgentQueryRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    verifier: VerificationService = Depends(get_verification_service),
) -> AgentQueryResponse:
    result = orchestrator.run(
        question=request.question,
        document_id=request.document_id,
        max_iterations=request.max_iterations,
    )
    verification = None
    if request.verify:
        context = AssembledContext(passages=result.sources)
        vresult = verifier.verify(
            question=request.question, answer=result.answer, context=context
        )
        verification = VerificationOut.from_result(vresult)
    return AgentQueryResponse(
        answer=result.answer,
        citations=[CitationOut.from_citation(c) for c in result.citations],
        sources=[SourceOut.from_passage(p) for p in result.sources],
        tools_run=result.tools_run,
        iterations=result.iterations,
        verification=verification,
    )
