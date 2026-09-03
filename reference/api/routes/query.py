"""Retrieval-augmented query endpoint."""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_query_service, get_verification_service
from app.schemas.query import CitationOut, QueryRequest, QueryResponse, SourceOut
from app.schemas.verification import VerificationOut
from app.services.retrieval.query_service import QueryService
from app.services.verification.service import VerificationService

router = APIRouter(tags=["query"])


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    service: QueryService = Depends(get_query_service),
    verifier: VerificationService = Depends(get_verification_service),
) -> QueryResponse:
    answer, context = service.answer_query(
        question=request.question,
        document_id=request.document_id,
        top_k=request.top_k,
    )
    verification = None
    if request.verify:
        result = verifier.verify(
            question=request.question, answer=answer.answer, context=context
        )
        verification = VerificationOut.from_result(result)
    return QueryResponse(
        answer=answer.answer,
        citations=[CitationOut.from_citation(c) for c in answer.citations],
        sources=[SourceOut.from_passage(p) for p in context.passages],
        verification=verification,
    )


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    service: QueryService = Depends(get_query_service),
) -> StreamingResponse:
    context = service.retrieve(
        question=request.question,
        document_id=request.document_id,
        top_k=request.top_k,
    )

    def event_stream() -> Iterator[str]:
        # 1) sources first, so the UI can render the citation panel immediately
        yield _sse(
            "sources",
            {"sources": [SourceOut.from_passage(p).model_dump() for p in context.passages]},
        )
        # 2) token stream
        buffer: list[str] = []
        for token in service.stream_answer(question=request.question, context=context):
            buffer.append(token)
            yield _sse("token", {"text": token})
        # 3) final answer + mapped citations
        full = "".join(buffer).strip()
        citations = service.citations_for(full, context)
        yield _sse(
            "done",
            {
                "answer": full,
                "citations": [CitationOut.from_citation(c).model_dump() for c in citations],
            },
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
