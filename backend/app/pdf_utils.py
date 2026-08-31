"""
Step 1 of the pipeline: turn a PDF into a list of overlapping text chunks, each tagged
with the page range it came from (so answers can cite pages).

Two things matter here and both used to be wrong:

1. Paragraph structure is preserved through extraction. Collapsing all whitespace to
   single spaces (the obvious `" ".join(text.split())`) destroys every paragraph boundary
   before the chunker can use it, which means a heading gets welded to unrelated prose and
   chunks start and end mid-sentence.
2. Chunks are packed from whole paragraphs (and, when a paragraph is too long, whole
   sentences) rather than sliced blindly every N words. Same idea as LangChain's
   RecursiveCharacterTextSplitter, written out plainly.
"""
import re
from typing import Dict, List, Tuple

import pymupdf  # PyMuPDF - the `fitz` import name still works but is deprecated

from app.logging_setup import get_logger

log = get_logger(__name__)

# Many real-world PDFs (scanned pages, exported figures) carry malformed embedded ICC
# color profiles. MuPDF's C layer logs one "format error: cmsOpenProfileFromMem failed"
# line per affected image straight to stderr - cosmetic, but it drowns a big book's
# ingestion log in thousands of near-duplicate lines.
pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)

# Sentence boundary: ., ! or ? followed by whitespace and a capital/quote/digit.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def _normalize(text: str) -> str:
    """Tidies extracted text WITHOUT destroying paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)          # collapse runs of spaces/tabs
    text = re.sub(r" *\n *", "\n", text)          # trim around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)        # 3+ blank lines -> one paragraph break
    return text.strip()


def extract_pages(file_path: str) -> List[Dict]:
    """Returns [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]"""
    doc = pymupdf.open(file_path)
    pages: List[Dict] = []
    try:
        for i, page in enumerate(doc, start=1):
            text = _normalize(page.get_text() or "")
            if text:
                pages.append({"page": i, "text": text})
    finally:
        doc.close()
    return pages


def _split_long_paragraph(paragraph: str, max_words: int) -> List[str]:
    """
    Breaks an over-long paragraph into pieces of at most max_words, preferring sentence
    boundaries and falling back to a hard word split only for a single huge sentence.
    """
    pieces: List[str] = []
    buffer: List[str] = []
    buffer_words = 0

    for sentence in _SENTENCE_END.split(paragraph):
        words = sentence.split()
        if not words:
            continue

        if len(words) > max_words:
            # One sentence longer than a whole chunk (tables, formula dumps, bad OCR).
            if buffer:
                pieces.append(" ".join(buffer))
                buffer, buffer_words = [], 0
            for start in range(0, len(words), max_words):
                pieces.append(" ".join(words[start:start + max_words]))
            continue

        if buffer_words + len(words) > max_words:
            pieces.append(" ".join(buffer))
            buffer, buffer_words = [], 0

        buffer.extend(words)
        buffer_words += len(words)

    if buffer:
        pieces.append(" ".join(buffer))
    return pieces


def _units(pages: List[Dict], max_words: int) -> List[Tuple[List[str], int]]:
    """Flattens pages into (words, page) units - one per paragraph, split if oversized."""
    units: List[Tuple[List[str], int]] = []
    for page in pages:
        for paragraph in page["text"].split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            for piece in _split_long_paragraph(paragraph, max_words):
                words = piece.split()
                if words:
                    units.append((words, page["page"]))
    return units


def chunk_document(
    pages: List[Dict],
    chunk_size_words: int = 300,
    overlap_words: int = 50,
) -> List[Dict]:
    """
    Packs whole paragraphs into chunks of up to chunk_size_words, carrying the trailing
    ~overlap_words of each chunk into the next one. Overlap stops an answer-relevant
    sentence from being cut in half between two chunks and lost to both.

    Returns: [{"text": "...", "page_start": 3, "page_end": 4}, ...]
    """
    # Overlap >= chunk size would make each chunk a near-copy of the last; cap it.
    overlap = max(0, min(overlap_words, chunk_size_words // 2))
    units = _units(pages, chunk_size_words)
    if not units:
        return []

    chunks: List[Dict] = []

    def emit(buffer: List[Tuple[List[str], int]]) -> None:
        if not buffer:
            return
        chunks.append({
            "text": " ".join(word for words, _ in buffer for word in words),
            "page_start": buffer[0][1],
            "page_end": buffer[-1][1],
        })

    buffer: List[Tuple[List[str], int]] = []
    buffer_words = 0

    for words, page in units:
        if buffer and buffer_words + len(words) > chunk_size_words:
            emit(buffer)
            # Carry over trailing units up to the overlap budget.
            carried: List[Tuple[List[str], int]] = []
            carried_words = 0
            for unit in reversed(buffer):
                if carried_words + len(unit[0]) > overlap:
                    break
                carried.insert(0, unit)
                carried_words += len(unit[0])
            buffer, buffer_words = carried, carried_words

        buffer.append((words, page))
        buffer_words += len(words)

    emit(buffer)
    return chunks


def format_pages(page_start: int, page_end: int) -> str:
    """'page 7' or 'pages 7-8' - chunks routinely straddle a page break."""
    return f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"
