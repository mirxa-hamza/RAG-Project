"""
Step 1 of the pipeline: turn a PDF into a list of overlapping text chunks,
each tagged with the page number it came from (so answers can cite a page).
"""
from typing import List, Dict
import pymupdf  # PyMuPDF - the `fitz` import name still works but is deprecated

# Many real-world PDFs (scanned pages, exported figures) carry malformed/unsupported
# embedded ICC color profiles. MuPDF's C layer logs one "format error:
# cmsOpenProfileFromMem failed" line per affected image straight to stderr - purely
# cosmetic (text extraction is unaffected), but it can drown a large book's ingestion
# log in thousands of near-duplicate lines. Silenced globally here.
pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)


def extract_pages(file_path: str) -> List[Dict]:
    """Returns [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]"""
    doc = pymupdf.open(file_path)
    pages = []
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text() or ""
            text = " ".join(text.split())  # collapse weird whitespace/newlines from PDF extraction
            if text.strip():
                pages.append({"page": i, "text": text})
    finally:
        doc.close()
    return pages


def chunk_document(
    pages: List[Dict],
    chunk_size_words: int = 300,
    overlap_words: int = 50,
) -> List[Dict]:
    """
    Flattens all pages into a single word stream (remembering which page each
    word came from), then slides a window over it to produce overlapping chunks.
    Overlap matters: it stops an answer-relevant sentence from being sliced in
    half between two chunks and lost to both.

    Returns: [{"text": "...", "page": 3}, ...]
    """
    words_with_page = []
    for p in pages:
        for w in p["text"].split():
            words_with_page.append((w, p["page"]))

    if not words_with_page:
        return []

    chunks = []
    step = max(chunk_size_words - overlap_words, 1)  # avoid infinite loop if overlap >= size
    for start in range(0, len(words_with_page), step):
        window = words_with_page[start:start + chunk_size_words]
        if not window:
            break
        text = " ".join(w for w, _ in window)
        first_page = window[0][1]
        chunks.append({"text": text, "page": first_page})
        if start + chunk_size_words >= len(words_with_page):
            break

    return chunks
