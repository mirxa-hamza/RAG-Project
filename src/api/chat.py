"""
Question answering endpoints.

Both the buffered and the streamed route go through the same `_prepare()` step, so the
retrieval behaviour can never drift between them.
"""
import json
from typing import Dict, List, Tuple

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.deps import get_current_user, user_id_of
from src.core.config import HISTORY_TURNS
from src.core.logging import get_logger, timed
from src.ml.llm import generate_answer, rewrite_question, stream_answer
from src.models.schemas import ChatRequest, ChatResponse, Source
from src.services import retrieval
from src.services.pdf import format_pages

log = get_logger(__name__)

router = APIRouter(tags=["chat"])


def _to_sources(chunks: List[dict]) -> List[Source]:
    return [
        Source(
            source=c["source"],
            pages=format_pages(c["page_start"], c["page_end"]),
            similarity=c.get("similarity"),
            rerank_score=c.get("rerank_score"),
            neighbor="neighbor_of" in c,
            snippet=(c["text"][:220] + "...") if len(c["text"]) > 220 else c["text"],
        )
        for c in chunks
    ]


def _prepare(req: ChatRequest, user_id: str) -> Tuple[List[Dict], str, List[Dict]]:
    """
    Rewrite the follow-up if there's history, then retrieve - scoped to `user_id`.

    Both routes go through here so retrieval, and with it the isolation filter, can never
    drift between the buffered and the streamed answer.
    """
    history = [turn.model_dump() for turn in req.history][-HISTORY_TURNS:]
    search_query = rewrite_question(req.question, history)
    chunks = retrieval.retrieve(search_query, top_k=req.top_k, source=req.source,
                                user_id=user_id)
    return history, search_query, chunks


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    with timed(log, "chat request"):
        history, search_query, chunks = _prepare(req, user_id_of(user))
        answer = generate_answer(req.question, chunks, history)

    return ChatResponse(
        answer=answer,
        sources=_to_sources(chunks),
        search_query=search_query if search_query != req.question else None,
    )


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, user: dict = Depends(get_current_user)):
    """
    Server-Sent Events version of /chat.

    Emits `sources` first (so the UI can show provenance immediately), then a stream of
    `token` events, then `done`. Retrieval happens before the generator starts, so a
    failure there surfaces as a normal HTTP error rather than mid-stream.
    """
    history, search_query, chunks = _prepare(req, user_id_of(user))
    sources = [s.model_dump() for s in _to_sources(chunks)]

    def events():
        yield _sse("sources", {
            "sources": sources,
            "search_query": search_query if search_query != req.question else None,
        })
        for piece in stream_answer(req.question, chunks, history):
            yield _sse("token", {"text": piece})
        yield _sse("done", {})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
