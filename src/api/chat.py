"""
Question answering endpoints.

Both the buffered and the streamed route go through the same `_prepare()` step, so the
retrieval behaviour can never drift between them.
"""
import json
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from src.api.auth import _enforce
from src.api.deps import get_current_user, user_id_of
from src.core import ratelimit
from src.core.config import HISTORY_TURNS
from src.core.logging import get_logger, timed
from src.ml.llm import generate_answer, rewrite_question, stream_answer
from src.models.schemas import ChatRequest, ChatResponse, Source
from src.core.config import ANSWER_CACHE_ENABLED
from src.services import answer_cache, retrieval
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


def _scope_key(req: ChatRequest) -> Optional[str]:
    """
    The document selection, as one stable string for the answer cache.

    Sorted: ticking A then B and B then A are the same search and must hit the same entry.
    Keying on the raw list order would quietly halve the hit rate.
    """
    wanted = req.wanted_sources()
    return None if wanted is None else "|".join(sorted(wanted))


def _prepare(req: ChatRequest, user_id: str) -> Tuple[List[Dict], str, List[Dict]]:
    """
    Rewrite the follow-up if there's history, then retrieve - scoped to `user_id`.

    Both routes go through here so retrieval, and with it the isolation filter, can never
    drift between the buffered and the streamed answer.
    """
    history = [turn.model_dump() for turn in req.history][-HISTORY_TURNS:]
    search_query = rewrite_question(req.question, history)
    chunks = retrieval.retrieve(search_query, top_k=req.top_k,
                                source=req.wanted_sources(), user_id=user_id)
    return history, search_query, chunks


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    uid = user_id_of(user)
    _enforce(ratelimit.CHAT, uid)

    # Only the first question of a conversation is cacheable: with history, the same words
    # mean different things ("and the second one?"), so the key would be a lie.
    cacheable = ANSWER_CACHE_ENABLED and not req.history
    if cacheable:
        cached = answer_cache.get(uid, req.question, _scope_key(req), req.top_k)
        if cached is not None:
            log.info("Answer cache hit.")
            return ChatResponse(**cached)

    with timed(log, "chat request"):
        history, search_query, chunks = _prepare(req, uid)
        answer = generate_answer(req.question, chunks, history)

    response = ChatResponse(
        answer=answer,
        sources=_to_sources(chunks),
        search_query=search_query if search_query != req.question else None,
    )
    # Never cache "the key is missing" or a transport failure as if it were an answer.
    if cacheable and chunks:
        answer_cache.put(uid, req.question, _scope_key(req), req.top_k, response.model_dump())
    return response


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request,
                      user: dict = Depends(get_current_user)):
    """
    Server-Sent Events version of /chat.

    Emits `sources` first (so the UI can show provenance immediately), then a stream of
    `token` events, then `done`. Retrieval happens before the generator starts, so a
    failure there surfaces as a normal HTTP error rather than mid-stream.
    """
    _enforce(ratelimit.CHAT, user_id_of(user))
    history, search_query, chunks = _prepare(req, user_id_of(user))
    sources = [s.model_dump() for s in _to_sources(chunks)]

    async def events():
        yield _sse("sources", {
            "sources": sources,
            "search_query": search_query if search_query != req.question else None,
        })
        for piece in stream_answer(req.question, chunks, history):
            # A closed tab used to keep the generation running to completion - tokens
            # nobody would ever read, billed all the same. Checked between tokens, so the
            # abandoned request stops at the next one rather than at the end.
            if await request.is_disconnected():
                log.info("Client disconnected mid-answer; stopping generation.")
                return
            yield _sse("token", {"text": piece})
        yield _sse("done", {})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
