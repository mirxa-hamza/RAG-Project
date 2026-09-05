"""
Verify that every golden question's answer is actually on the page it cites.

    python scripts/check_golden.py
    python scripts/check_golden.py --set eval/golden_questions.json

A golden set is the measuring instrument for every retrieval change in this project, and a
wrong page number in it is worse than no set at all: the question scores as a MISS for
every configuration, so a real improvement and a real regression look identical. The usual
way that happens is silent - a PDF is replaced with a different scan, a page offset shifts,
someone edits a question and forgets its page.

So each entry carries an `evidence` list: literal strings that must appear in the extracted
text of `expected_pages`. This script re-extracts the documents and checks them. It uses
`pdf.extract_pages()`, the same extraction the ingest path uses, so it is checking what the
chunker will actually see - not what a human reading the PDF in a viewer sees.

Run it after editing the set, and after replacing anything in data/.

Exit status is 1 if any entry fails, so it can gate a commit.
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


def normalise(text: str) -> str:
    """Collapse whitespace, so evidence written on one line matches text wrapped over two."""
    return re.sub(r"\s+", " ", text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that golden questions point at pages that contain their answers.")
    parser.add_argument("--set", default=str(PROJECT_ROOT / "eval" / "golden_questions.json"))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    data = json.loads(Path(args.set).read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"questions": data}
    questions = data.get("questions", [])
    if not questions:
        print(f"No questions in {args.set}")
        return 1

    # Extract each referenced document once, not once per question.
    wanted = sorted({q.get("expected_source") for q in questions if q.get("expected_source")})
    text_by_page = {}
    for name in wanted:
        path = Path(args.data_dir) / name
        if not path.exists():
            print(f"MISSING DOCUMENT: {name}")
            print(f"  expected at {path}")
            return 1
        pages = extract_pages(str(path))
        text_by_page[name] = {p["page"]: normalise(p["text"]) for p in pages}
        print(f"{name}: {len(pages)} pages with text "
              f"(physical {pages[0]['page']}-{pages[-1]['page']})")

    failures = 0
    no_evidence = 0
    print()
    for index, item in enumerate(questions, start=1):
        source = item.get("expected_source")
        pages = item.get("expected_pages") or []
        evidence = item.get("evidence") or []
        label = f"{index:>3}. {item['question'][:66]}"

        if not evidence:
            # Not a failure - an older entry may predate this field - but say so, because an
            # entry with no evidence is one this script cannot protect.
            no_evidence += 1
            print(f"  [no evidence] {label}")
            continue

        available = text_by_page.get(source, {})
        combined = " ".join(available.get(page, "") for page in pages)
        if not combined.strip():
            print(f"  [EMPTY PAGE ] {label}")
            print(f"                pages {pages} of {source} have no extracted text")
            failures += 1
            continue

        missing = [phrase for phrase in evidence if normalise(phrase) not in combined]
        if missing:
            failures += 1
            print(f"  [NOT ON PAGE] {label}")
            for phrase in missing:
                print(f"                missing from pages {pages}: {phrase!r}")
                # Say where it actually is, when it is somewhere - that turns a failure into
                # a one-line fix instead of a hunt.
                found = [p for p, text in available.items() if normalise(phrase) in text]
                if found:
                    print(f"                found instead on page(s): {found[:6]}")
        else:
            print(f"  [ok         ] {label}")

    print("\n" + "=" * 70)
    print(f"{len(questions)} questions: {len(questions) - failures - no_evidence} verified, "
          f"{failures} failed, {no_evidence} carry no evidence to check")
    unanswerable = data.get("unanswerable", [])
    print(f"{len(unanswerable)} unanswerable questions (refusal rate)")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
