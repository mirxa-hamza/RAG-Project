"""
Step 4 of the pipeline: given the user's question and the chunks retrieved from ChromaDB,
ask Groq's LLM to answer using ONLY that context.
"""
from typing import Dict, List

from groq import Groq

from app.config import (
    GROQ_API_KEY,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    MAX_CONTEXT_CHARS,
)
from app.logging_setup import get_logger, timed
from app.pdf_utils import format_pages

log = get_logger(__name__)

_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "CONTEXT provided below, which was retrieved from the ingested PDF document(s). "
    "Rules:\n"
    "1. Only use information found in the CONTEXT. Do not use outside knowledge.\n"
    "2. If the CONTEXT does not contain the answer, say clearly that the "
    "document doesn't contain that information - do not guess.\n"
    "3. When you use a fact, cite the document and page it came from, exactly as "
    "labelled in the CONTEXT, e.g. (Some Book.pdf, page 3).\n"
    "4. Be concise and direct."
)


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
            log.info("Context budget (%d chars) reached after %d chunks.", max_chars, len(parts))
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: List[Dict]) -> str:
    if _client is None:
        return (
            "GROQ_API_KEY is not set. Add it to backend/.env "
            "(get a free key at https://console.groq.com/keys) and restart the server."
        )

    if not chunks:
        return "No documents have been ingested yet, so I have no context to answer from."

    user_message = f"CONTEXT:\n{build_context(chunks)}\n\nQUESTION:\n{question}"

    try:
        with timed(log, f"Groq call ({GROQ_MODEL})"):
            response = _client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=GROQ_TEMPERATURE,  # low: we want grounded answers, not creative ones
                max_tokens=GROQ_MAX_TOKENS,
            )
        return response.choices[0].message.content
    except Exception as exc:
        # Rate limits, a deprecated model id, a network blip - none of these should
        # surface as an unhandled 500 with a stack trace.
        log.exception("Groq request failed")
        return (
            f"The language model request failed ({type(exc).__name__}: {exc}). "
            "The retrieved sources below are still from your documents. If this says the "
            "model was not found, update GROQ_MODEL in backend/.env - see "
            "https://console.groq.com/docs/models"
        )
