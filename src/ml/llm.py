"""
Step 4 of the pipeline: given the user's question and the chunks retrieved from ChromaDB,
ask Groq's LLM to answer using ONLY that context.

Also holds the two conversation-level concerns:
  * carrying recent turns into the prompt, and
  * rewriting a follow-up ("what about the second one?") into a standalone question
    BEFORE retrieval, because the raw follow-up embeds to nothing useful.
"""
from typing import Dict, Iterator, List, Optional

from groq import Groq

from src.core.config import (
    GROQ_API_KEY,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    HISTORY_TURNS,
    MAX_CONTEXT_CHARS,
    REWRITE_FOLLOWUPS,
)
from src.core.logging import get_logger, timed
from src.services.pdf import format_pages

log = get_logger(__name__)

_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

NO_KEY_MESSAGE = (
    "GROQ_API_KEY is not set. Add it to backend/.env "
    "(get a free key at https://console.groq.com/keys) and restart the server."
)

NO_CONTEXT_MESSAGE = (
    "I couldn't find anything about that in the ingested documents, so I won't guess. "
    "If you expected this to be covered, check that the right PDF is in the data folder "
    "and has been ingested."
)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "CONTEXT provided below, which was retrieved from the ingested PDF document(s). "
    "Rules:\n"
    "1. Only use information found in the CONTEXT. Do not use outside knowledge.\n"
    "2. If the CONTEXT does not contain the answer, say clearly that the "
    "document doesn't contain that information - do not guess.\n"
    "3. When you use a fact, cite the document and page it came from, exactly as "
    "labelled in the CONTEXT, e.g. (Some Book.pdf, page 3).\n"
    "4. Earlier conversation turns are for understanding what the user means; the CONTEXT "
    "is the only source of facts.\n"
    "5. Be concise and direct."
)

REWRITE_PROMPT = (
    "Rewrite the user's latest message as a standalone search query that makes sense "
    "without the conversation history. Resolve pronouns and references to earlier turns. "
    "Keep it short and keep the original technical terms. Reply with the rewritten query "
    "only - no preamble, no quotes."
)


def is_configured() -> bool:
    return _client is not None


def build_context(chunks: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Formats retrieved chunks into a labeled block the model can cite from, stopping at a
    character budget so a large top_k can never overflow the model's context window.
    """
    parts: List[str] = []
    used = 0
    for chunk in chunks:
        label = f"[{chunk.get('source')} - {format_pages(chunk['page_start'], chunk['page_end'])}]"
        block = f"{label}\n{chunk['text']}"

        if used + len(block) > max_chars:
            # Never return an empty CONTEXT. If the very first chunk is bigger than the
            # whole budget, truncate it and send that instead: the system prompt tells the
            # model to answer only from CONTEXT, so handing it nothing at all is an
            # invitation to answer from its own weights.
            if not parts:
                room = max_chars - len(label) - 1
                if room > 0:
                    log.warning(
                        "First chunk (%d chars) exceeds the whole %d-char context budget; "
                        "truncating it rather than sending an empty CONTEXT.",
                        len(block), max_chars,
                    )
                    parts.append(f"{label}\n{chunk['text'][:room]}")
            else:
                log.info("Context budget (%d chars) reached after %d chunks.", max_chars, len(parts))
            break

        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


def _history_messages(history: Optional[List[Dict]]) -> List[Dict]:
    """Last HISTORY_TURNS question/answer pairs, oldest first, as chat messages."""
    if not history:
        return []
    messages: List[Dict] = []
    for turn in history[-HISTORY_TURNS:]:
        question = (turn.get("question") or "").strip()
        answer = (turn.get("answer") or "").strip()
        if question:
            messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})
    return messages


def rewrite_question(question: str, history: Optional[List[Dict]]) -> str:
    """
    Turns a follow-up into a standalone question for the retrieval step.

    "What about the second one?" carries no retrievable content on its own - embedding it
    returns noise. Rewriting costs one small, cheap LLM call and is skipped entirely when
    there's no history, when rewriting is disabled, or when the call fails.
    """
    if not history or not REWRITE_FOLLOWUPS or _client is None:
        return question

    try:
        with timed(log, "rewrite follow-up"):
            response = _client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    *_history_messages(history),
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=120,
            )
        rewritten = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        log.warning("Follow-up rewrite failed (%s); retrieving with the raw question.", exc)
        return question

    if not rewritten or len(rewritten) > 500:
        return question
    if rewritten.lower() != question.lower():
        log.info("Rewrote follow-up for retrieval: %r -> %r", question, rewritten)
    return rewritten


def _messages(question: str, chunks: List[Dict], history: Optional[List[Dict]]) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_history_messages(history),
        {"role": "user", "content": f"CONTEXT:\n{build_context(chunks)}\n\nQUESTION:\n{question}"},
    ]


def generate_answer(
    question: str,
    chunks: List[Dict],
    history: Optional[List[Dict]] = None,
) -> str:
    # Checked before the API key: "nothing relevant was retrieved" is an answer the
    # system can give entirely on its own, with no LLM call and no key required.
    if not chunks:
        return NO_CONTEXT_MESSAGE
    if _client is None:
        return NO_KEY_MESSAGE

    try:
        with timed(log, f"Groq call ({GROQ_MODEL})"):
            response = _client.chat.completions.create(
                model=GROQ_MODEL,
                messages=_messages(question, chunks, history),
                temperature=GROQ_TEMPERATURE,  # low: grounded answers, not creative ones
                max_tokens=GROQ_MAX_TOKENS,
            )
        return response.choices[0].message.content
    except Exception as exc:
        # Rate limits, a deprecated model id, a network blip - none of these should
        # surface as an unhandled 500 with a stack trace.
        log.exception("Groq request failed")
        return _error_message(exc)


def stream_answer(
    question: str,
    chunks: List[Dict],
    history: Optional[List[Dict]] = None,
) -> Iterator[str]:
    """
    Yields the answer in pieces as Groq produces them.

    Groq's throughput is its main selling point; waiting for the whole completion before
    showing anything hides it behind a "Thinking..." spinner.
    """
    if not chunks:
        yield NO_CONTEXT_MESSAGE
        return
    if _client is None:
        yield NO_KEY_MESSAGE
        return

    try:
        stream = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=_messages(question, chunks, history),
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
            stream=True,
        )
        for event in stream:
            piece = event.choices[0].delta.content
            if piece:
                yield piece
    except Exception as exc:
        log.exception("Groq stream failed")
        yield _error_message(exc)


def _error_message(exc: Exception) -> str:
    return (
        f"The language model request failed ({type(exc).__name__}: {exc}). "
        "The retrieved sources below are still from your documents. If this says the "
        "model was not found, update GROQ_MODEL in backend/.env - see "
        "https://console.groq.com/docs/models"
    )
