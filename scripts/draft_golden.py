"""
Draft a golden question set from the documents you actually have.

    python scripts/draft_golden.py                      # -> eval/golden_draft.json
    python scripts/draft_golden.py --per-doc 20
    python scripts/draft_golden.py --source "users/<id>/book.pdf"

Why this exists: `eval/golden_questions.json` shipped with questions about a synthetic
fixture PDF, so every eval number described a document nobody cares about. Writing 25 real
questions by hand is an afternoon's work, and the boring half of that afternoon is finding
passages worth asking about and recording which page they are on.

This does the boring half. It scans each document for passages that look *answerable* -
definitions, named quantities, stated relationships - and writes a draft entry with the
page already filled in, the passage quoted for context, and a question stub.

**It does not write good questions.** The stubs are mechanical ("What does the document say
about X?") and some passages will be poor choices. Read the draft, rewrite the questions in
your own words, delete the weak ones, then save it as `eval/golden_questions.json`. Every
entry carries `"reviewed": false` until you say otherwise, and `run_eval.py` warns when it
sees them.
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import DATA_DIR  # noqa: E402
from src.services.pdf import extract_pages  # noqa: E402

# Passages that tend to make answerable questions: a definition, a named quantity, or an
# explicit relationship. Deliberately narrow - a bad candidate wastes a reviewer's time,
# and there are always more passages than questions needed.
_PATTERNS = [
    (re.compile(r"\b([A-Z][A-Za-z0-9 '\-]{2,40})\s+is (?:defined as|the|a|an)\s", re.I), "definition"),
    (re.compile(r"\b([A-Z][A-Za-z0-9 '\-]{2,40})\s+(?:refers to|means)\s", re.I), "definition"),
    (re.compile(r"\b(?:consists? of|comprises?|is composed of)\s", re.I), "structure"),
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:percent|%|times|years|steps|states|nodes)\b", re.I), "quantity"),
    (re.compile(r"\b(?:the (?:main|primary|key) (?:goal|purpose|advantage|drawback|difference))\b", re.I), "claim"),
]

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def candidates(text: str, page: int):
    """Sentences on one page that match a pattern, with the pattern's kind."""
    for sentence in _SENTENCE.split(text):
        sentence = " ".join(sentence.split())
        # Too short to contain an answer; too long to quote in a review.
        if not (60 <= len(sentence) <= 320):
            continue
        for pattern, kind in _PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            subject = (match.group(1).strip() if match.groups() else "").strip(" ,;:")
            yield {"kind": kind, "subject": subject, "sentence": sentence, "page": page}
            break


def stub_question(candidate: dict) -> str:
    """A placeholder for a human to rewrite. Never pretends to be finished."""
    subject = candidate["subject"]
    if candidate["kind"] == "definition" and subject:
        return f"What is {subject}?"
    if candidate["kind"] == "quantity":
        return "REWRITE: ask for the figure stated in this passage"
    if subject:
        return f"REWRITE: ask about {subject}"
    return "REWRITE: ask what this passage states"


def draft_for(filename: str, per_doc: int):
    path = DATA_DIR / filename
    print(f"reading {filename} ...")
    pages = extract_pages(str(path))
    if not pages:
        print(f"  no extractable text; skipped")
        return []

    found = []
    for page in pages:
        found.extend(candidates(page["text"], page["page"]))

    # Spread across the document rather than taking the first N, which would all come from
    # the front matter and the first chapter.
    if len(found) > per_doc:
        step = len(found) / per_doc
        found = [found[int(i * step)] for i in range(per_doc)]

    print(f"  {len(pages)} pages -> {len(found)} candidate passage(s)")
    return [
        {
            "question": stub_question(c),
            "expected_source": filename,
            "expected_pages": [c["page"]],
            "passage": c["sentence"],
            "kind": c["kind"],
            "reviewed": False,
        }
        for c in found
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-doc", type=int, default=15,
                        help="candidate questions per document (default: 15)")
    parser.add_argument("--source", help="only this document (path relative to data/)")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "eval" / "golden_draft.json"))
    args = parser.parse_args()

    if args.source:
        names = [args.source]
    else:
        names = sorted(p.relative_to(DATA_DIR).as_posix()
                       for p in DATA_DIR.rglob("*.pdf") if p.is_file())

    if not names:
        print(f"No PDFs under {DATA_DIR}.")
        return 1

    entries = []
    for name in names:
        entries.extend(draft_for(name, args.per_doc))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {len(entries)} draft entries to {out}")
    print("\nNext, and this part is not optional:")
    print("  1. Open it and rewrite each question the way a person would ask it.")
    print("  2. Delete entries whose passage is not actually worth asking about.")
    print("  3. Check expected_pages is right - the passage is quoted so you can.")
    print('  4. Set "reviewed": true, then save as eval/golden_questions.json.')
    print("\nAim for 20-30 good questions, including a few the documents CANNOT answer -")
    print("the refusal rate is only meaningful if some questions deserve a refusal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
