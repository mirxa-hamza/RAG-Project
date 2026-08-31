"""
Step 4 of the pipeline: given the user's question and the chunks retrieved
from ChromaDB, ask Groq's LLM to answer using ONLY that context.
"""
from typing import List, Dict
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "CONTEXT provided below, which was retrieved from the ingested PDF document(s). "
    "Rules:\n"
    "1. Only use information found in the CONTEXT. Do not use outside knowledge.\n"
    "2. If the CONTEXT does not contain the answer, say clearly that the "
    "document doesn't contain that information - do not guess.\n"
    "3. When you use a fact, mention which page it came from, e.g. (page 3).\n"
    "4. Be concise and direct."
)


def build_context(chunks: List[Dict]) -> str:
    """Formats retrieved chunks into a labeled block the model can cite from."""
    parts = []
    for c in chunks:
        parts.append(f"[Page {c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: List[Dict]) -> str:
    if _client is None:
        return (
            "GROQ_API_KEY is not set. Add it to backend/.env "
            "(get a free key at https://console.groq.com/keys) and restart the server."
        )

    if not chunks:
        return "No documents have been ingested yet, so I have no context to answer from."

    context = build_context(chunks)
    user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    response = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,  # low temperature: we want grounded answers, not creative ones
        max_tokens=800,
    )
    return response.choices[0].message.content
