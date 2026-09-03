"""Token-aware recursive chunker.

Splits a ParsedDocument into overlapping, embedding-sized Chunks while
preserving page-level provenance and character offsets for citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.document import Chunk, ChunkMetadata
from app.services.ingestion import tokenizer
from app.services.ingestion.parsers import ParsedDocument

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class _Segment:
    text: str
    start: int
    end: int


def _segment_paragraphs(text: str) -> list[_Segment]:
    """Split into paragraph segments, tracking char offsets in ``text``."""
    segments: list[_Segment] = []
    cursor = 0
    for part in _PARAGRAPH_SPLIT.split(text):
        stripped = part.strip()
        if not stripped:
            cursor += len(part)
            continue
        start = text.find(stripped, cursor)
        if start == -1:  # defensive; shouldn't happen
            start = cursor
        end = start + len(stripped)
        segments.append(_Segment(text=stripped, start=start, end=end))
        cursor = end
    return segments


def _tail_by_tokens(text: str, max_tokens: int) -> str:
    """Return the trailing ``max_tokens`` tokens of ``text`` as text."""
    if max_tokens <= 0:
        return ""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[-max_tokens:])


def _hard_split(segment: _Segment, max_tokens: int) -> list[_Segment]:
    """Split a single oversized segment into token-window pieces."""
    tokens = tokenizer.encode(segment.text)
    pieces: list[_Segment] = []
    for i in range(0, len(tokens), max_tokens):
        window = tokens[i : i + max_tokens]
        piece_text = tokenizer.decode(window).strip()
        if piece_text:
            # Offsets are approximate for hard-split pieces: attribute them to
            # the parent segment's span.
            pieces.append(_Segment(text=piece_text, start=segment.start, end=segment.end))
    return pieces


def chunk_document(
    parsed: ParsedDocument,
    *,
    document_id: str,
    filename: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Chunk a parsed document into overlapping, token-bounded Chunks."""
    chunks: list[Chunk] = []
    ordinal = 0

    for page in parsed.pages:
        segments = _segment_paragraphs(page.text)
        buffer: list[_Segment] = []
        buffer_tokens = 0
        overlap_text = ""  # carried tail from the previous chunk on this page

        # Bind the page number per iteration (flush is a closure over the loop).
        def flush(page_number: int | None = page.page) -> None:
            nonlocal ordinal, overlap_text, buffer, buffer_tokens
            if not buffer:
                return
            body = "\n\n".join(s.text for s in buffer)
            text = f"{overlap_text}\n\n{body}".strip() if overlap_text else body
            chunks.append(
                Chunk(
                    document_id=document_id,
                    text=text,
                    ordinal=ordinal,
                    token_count=tokenizer.count_tokens(text),
                    metadata=ChunkMetadata(
                        document_id=document_id,
                        filename=filename,
                        page=page_number,
                        start_char=buffer[0].start,
                        end_char=buffer[-1].end,
                    ),
                )
            )
            ordinal += 1
            overlap_text = _tail_by_tokens(body, overlap_tokens)
            buffer = []
            buffer_tokens = 0

        for seg in segments:
            seg_tokens = tokenizer.count_tokens(seg.text)

            if seg_tokens > max_tokens:
                flush()
                for piece in _hard_split(seg, max_tokens):
                    buffer = [piece]
                    buffer_tokens = tokenizer.count_tokens(piece.text)
                    flush()
                continue

            if buffer and buffer_tokens + seg_tokens > max_tokens:
                flush()

            buffer.append(seg)
            buffer_tokens += seg_tokens

        flush()

    return chunks
