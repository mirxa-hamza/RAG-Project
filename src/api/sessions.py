"""
Chat history endpoints.

    POST   /api/sessions                  start a conversation
    GET    /api/sessions?limit&cursor     one page of the sidebar, newest first
    GET    /api/sessions/{id}             one conversation, with its messages
    POST   /api/sessions/{id}/messages    append a message
    PATCH  /api/sessions/{id}             rename
    DELETE /api/sessions/{id}             delete

Every route resolves the caller from the bearer token and passes that id into the service;
nothing here takes a user id from the request body. A session belonging to someone else is
a 404, not a 403 - "that exists but is not yours" is itself information.

The listing deliberately does NOT return messages. Opening the app with two hundred saved
conversations should transfer two hundred titles, not two hundred transcripts.
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.auth import _enforce
from src.api.deps import get_current_user, user_id_of
from src.core import ratelimit
from src.core.config import SESSION_PAGE_SIZE
from src.core.logging import get_logger
from src.services import sessions

log = get_logger(__name__)

router = APIRouter(tags=["sessions"], prefix="/api/sessions")


class SessionCreate(BaseModel):
    # Optional: the UI usually lets the first question name the conversation instead.
    title: Optional[str] = Field(None, max_length=200)


class SessionRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class MessageIn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    # The same ceiling the chat history uses. This text is stored and later replayed into
    # an LLM request, so it cannot be unbounded.
    content: str = Field(..., max_length=12000)
    # Provenance for an assistant message, kept so a reopened conversation still shows
    # which pages the answer came from.
    sources: Optional[List[dict]] = Field(None, max_length=40)


@router.post("", status_code=201)
async def create_session(body: SessionCreate, user: dict = Depends(get_current_user)):
    uid = user_id_of(user)
    # Chat rate limit rather than a new one: creating conversations in a loop is the same
    # kind of abuse as asking questions in a loop.
    _enforce(ratelimit.CHAT, uid)
    return await sessions.create(uid, sessions.title_from(body.title or "") if body.title
                                 else "New chat")


@router.get("")
async def list_sessions(
    limit: int = Query(SESSION_PAGE_SIZE, ge=1, le=50),
    cursor: Optional[str] = Query(None, max_length=200),
    user: dict = Depends(get_current_user),
):
    """One page of history. `next_cursor` is null when there is nothing older."""
    return await sessions.list_page(user_id_of(user), limit=limit, cursor=cursor)


@router.get("/{session_id}")
async def get_session(session_id: str, user: dict = Depends(get_current_user)):
    found = await sessions.get(user_id_of(user), session_id)
    if not found:
        raise HTTPException(status_code=404, detail="That conversation is not in your history.")
    return found


@router.post("/{session_id}/messages", status_code=201)
async def add_message(session_id: str, body: MessageIn,
                      user: dict = Depends(get_current_user)):
    saved = await sessions.append_message(user_id_of(user), session_id, body.role,
                                          body.content, body.sources)
    if not saved:
        raise HTTPException(status_code=404, detail="That conversation is not in your history.")
    return saved


@router.patch("/{session_id}")
async def rename_session(session_id: str, body: SessionRename,
                         user: dict = Depends(get_current_user)):
    if not await sessions.rename(user_id_of(user), session_id, body.title):
        raise HTTPException(status_code=404, detail="That conversation is not in your history.")
    return {"id": session_id, "title": sessions.title_from(body.title)}


@router.delete("/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    if not await sessions.delete(user_id_of(user), session_id):
        raise HTTPException(status_code=404, detail="That conversation is not in your history.")
    return {"id": session_id, "deleted": True}
