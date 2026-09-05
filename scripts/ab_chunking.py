"""
A/B one chunking strategy against another, offline, without touching the real index.

    python scripts/ab_chunking.py --max-pages 150
    python scripts/ab_chunking.py --docs "data/Pattern*.pdf" --top-k 8
    python scripts/ab_chunking.py --percentile 90 --out eval/runs/semantic-p90.json

WHAT IT DOES. Extracts the documents once, then for each strategy: chunks them, embeds
every chunk, and answers the golden questions by brute-force cosine search over those
vectors - optionally re-ranked, exactly as the live pipeline would. It prints hit-rate@k
and MRR for each strategy side by side, plus what the chunking cost.

WHY BRUTE FORCE RATHER THAN THE REAL STORE.
* Exact search. Chroma and Pinecone are approximate; their recall varies run to run, and
  that variance is the same size as the effect being measured. Comparing exact against
  exact means a difference in the numbers is a difference in the chunking.
* Nothing is written anywhere. No collection to create, no manifest to corrupt, no
  re-ingest of your real index, and no way for this script to leave the two strategies'
  vectors mixed in one store - which is the mistake that makes a chunking A/B produce
  confident nonsense.

WHAT IT DELIBERATELY LEAVES OUT. BM25 fusion. It is built from the live vector store, and
wiring it in here would mean writing to that store. Dense-only is also where chunking
shows up most clearly, since BM25 matches keywords regardless of where the boundaries
fall. Take the numbers as "retrieval quality attributable to chunking", not as a
prediction of the full hybrid pipeline's absolute hit-rate.

BEFORE THE NUMBERS MEAN ANYTHING: eval/golden_questions.json must contain YOUR questions
about YOUR documents. It ships pointing at the synthetic fixture PDF, and this script says
so loudly rather than printing an impressive-looking 0%.
"""
import argparse
import glob

import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import (  # noqa: E402
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    DATA_DIR,
    EMBEDDINGS_PROVIDER,
    NEIGHBOR_EXPANSION,
    RETRIEVAL_CANDIDATES,
    SEMANTIC_BREAKPOINT_PERCENTILE,
    SEMANTIC_BUFFER_SIZE,
    SEMANTIC_MAX_CHUNK_WORDS,
    SEMANTIC_MIN_CHUNK_WORDS,
    TOP_K,
)
from src.core.logging import get_logger  # noqa: E402
from src.ml import reranker  # noqa: E402
from src.ml.embeddings import embed_passages, embed_query, split_to_token_limit  # noqa: E402
from src.services.chunking import semantic_chunks  # noqa: E402
from src.services.pdf import chunk_document, extract_pages  # noqa: E402

log = get_logger("ab_chunking")

try:
    import numpy as _np
except ImportError:  # pragma: no cover - only in a cloud bundle without numpy
    _np = None


def _covers(chunk: dict, source: str, pages: list) -> bool:
    """
    True if this chunk is from the expected document and covers an expected page.

    Kept identical to eval/run_eval.py._covers ON PURPOSE, and duplicated rather than
    imported: importing that module pulls in the LLM client and the live retrieval stack,
    which is a lot of connecting-to-things for a six-line predicate in a script whose whole
    point is that it touches nothing. If you change the hit definition, change it in both
    places - the two reporting different numbers for the same run would be worse than the
    duplication.
    """
    if source and chunk.get("source") != source:
        return False
    start, end = chunk.get("page_start"), chunk.get("page_end")
    if start is None or end is None:
        return False
    return any(start <= page <= end for page in pages)


# ---------------------------------------------------------------- documents

def resolve_documents(patterns: List[str], max_pages: Optional[int]) -> Dict[str, List[Dict]]:
    """{filename: pages} for every PDF matched, extracted once and shared by both runs."""
    paths: List[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_file():
            paths.append(candidate)
            continue
        base = pattern if Path(pattern).is_absolute() else str(PROJECT_ROOT / pattern)
        paths.extend(Path(p) for p in sorted(glob.glob(base)))

    documents: Dict[str, List[Dict]] = {}
    for path in paths:
        if path.suffix.lower() != ".pdf":
            continue
        started = time.perf_counter()
        pages = extract_pages(str(path))
        if max_pages:
            pages = pages[:max_pages]
        if not pages:
            print(f"  !! {path.name}: no extractable text (scanned PDF?) - skipped")
            continue
        documents[path.name] = pages
        print(f"  {path.name}: {len(pages)} pages ({time.perf_counter() - started:.1f}s)")
    return documents


# ---------------------------------------------------------------- one strategy's index

def build_index(documents: Dict[str, List[Dict]], strategy: str, args) -> Dict:
    """Chunk + embed every document under one strategy. Returns everything a search needs."""
    chunks: List[Dict] = []
    chunk_started = time.perf_counter()

    for filename, pages in documents.items():
        if strategy == "fixed":
            produced = chunk_document(pages, args.chunk_size, args.overlap)
        else:
            produced = semantic_chunks(
                pages,
                percentile=args.percentile,
                buffer_size=args.buffer,
                min_words=args.min_words,
                max_words=args.max_words,
            )
        produced = split_to_token_limit(produced)
        # index_in_source is what neighbour expansion walks, exactly as the vector store's
        # metadata does in the live pipeline.
        for position, chunk in enumerate(produced):
            chunk["source"] = filename
            chunk["index_in_source"] = position
        chunks.extend(produced)

    chunk_seconds = time.perf_counter() - chunk_started

    embed_started = time.perf_counter()
    vectors = embed_passages([c["text"] for c in chunks])
    embed_seconds = time.perf_counter() - embed_started

    sizes = [len(c["text"].split()) for c in chunks]
    return {
        "strategy": strategy,
        "chunks": chunks,
        "vectors": _np.asarray(vectors, dtype="float32") if _np is not None else vectors,
        "stats": {
            "chunks": len(chunks),
            "words_mean": round(statistics.mean(sizes), 1) if sizes else 0,
            "words_median": round(statistics.median(sizes), 1) if sizes else 0,
            "words_min": min(sizes) if sizes else 0,
            "words_max": max(sizes) if sizes else 0,
            "chunk_seconds": round(chunk_seconds, 1),
            "embed_seconds": round(embed_seconds, 1),
        },
    }


# ---------------------------------------------------------------- search

def _similarities(vectors, query) -> List[float]:
    """Cosine similarity of one query against every chunk vector."""
    if _np is not None:
        matrix = vectors
        q = _np.asarray(query, dtype="float32")
        norms = _np.linalg.norm(matrix, axis=1) * _np.linalg.norm(q)
        # A zero-norm row would divide by zero; it can only come from an empty chunk.
        norms[norms == 0] = 1e-12
        return (matrix @ q / norms).tolist()

    out = []
    q_norm = math.sqrt(sum(x * x for x in query)) or 1e-12
    for vector in vectors:
        v_norm = math.sqrt(sum(x * x for x in vector)) or 1e-12
        out.append(sum(a * b for a, b in zip(vector, query)) / (v_norm * q_norm))
    return out


def _expand(hits: List[Dict], index: Dict, expand: int) -> List[Dict]:
    """
    Add the chunks immediately before and after each hit, as retrieval.py does.

    Marked with neighbor_of so the rank metric can still be measured on primary hits only -
    a neighbour that happens to cover the right page is a hit, but it is not evidence that
    the ranking put the answer first.
    """
    if expand <= 0:
        return hits
    by_position = {(c["source"], c["index_in_source"]): c for c in index["chunks"]}
    seen = {(c["source"], c["index_in_source"]) for c in hits}
    out = list(hits)
    for hit in hits:
        for step in range(1, expand + 1):
            for offset in (-step, step):
                key = (hit["source"], hit["index_in_source"] + offset)
                if key in by_position and key not in seen:
                    seen.add(key)
                    out.append(dict(by_position[key], neighbor_of=hit["index_in_source"]))
    return out


def search(index: Dict, question: str, top_k: int, candidates: int,
           use_rerank: bool, expand: int) -> List[Dict]:
    scores = _similarities(index["vectors"], embed_query(question))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidates]
    hits = [dict(index["chunks"][i], similarity=float(scores[i])) for i in ranked]

    if use_rerank and reranker.is_available():
        reranked = reranker.rerank(question, hits)
        if reranked is not None:
            floor = reranker.score_floor()
            hits = [dict(chunk, rerank_score=float(score))
                    for chunk, score in reranked if score >= floor]

    return _expand(hits[:top_k], index, expand)


# ---------------------------------------------------------------- scoring

def score(index: Dict, questions: List[Dict], args) -> Dict:
    hits = 0
    reciprocal = []
    rows = []

    for item in questions:
        question = item["question"]
        source = item.get("expected_source")
        pages = item.get("expected_pages", [])

        retrieved = search(index, question, args.top_k, args.candidates,
                           not args.no_rerank, args.expand)
        primary = [c for c in retrieved if "neighbor_of" not in c]

        rank = next((i + 1 for i, c in enumerate(primary) if _covers(c, source, pages)), None)
        hit = rank is not None or any(_covers(c, source, pages) for c in retrieved)
        hits += bool(hit)
        reciprocal.append(1.0 / rank if rank else 0.0)
        rows.append({"question": question, "hit": bool(hit), "rank": rank})

    total = len(questions) or 1
    return {
        "hit_rate_at_k": round(hits / total, 3),
        "mrr": round(sum(reciprocal) / total, 3),
        "rows": rows,
    }


# ---------------------------------------------------------------- report

def _delta(new: float, old: float) -> str:
    difference = new - old
    if abs(difference) < 1e-9:
        return "   ="
    return f"{difference:+.3f}"


def report(results: List[Dict], questions: int, args) -> None:
    baseline = results[0]
    print("\n" + "=" * 74)
    print(f"{'strategy':<12}{'chunks':>8}{'words(med)':>12}{'chunk s':>10}"
          f"{'embed s':>10}{'hit@' + str(args.top_k):>9}{'MRR':>8}")
    print("-" * 74)
    for result in results:
        stats, metrics = result["stats"], result["metrics"]
        print(f"{result['strategy']:<12}{stats['chunks']:>8}{stats['words_median']:>12}"
              f"{stats['chunk_seconds']:>10}{stats['embed_seconds']:>10}"
              f"{metrics['hit_rate_at_k']:>9.3f}{metrics['mrr']:>8.3f}")
    print("-" * 74)
    for result in results[1:]:
        print(f"{result['strategy']} vs {baseline['strategy']}:  "
              f"hit-rate {_delta(result['metrics']['hit_rate_at_k'], baseline['metrics']['hit_rate_at_k'])}"
              f"   MRR {_delta(result['metrics']['mrr'], baseline['metrics']['mrr'])}")
    print("=" * 74)

    # A difference smaller than one question cannot be distinguished from noise with a set
    # this size, and saying so is the difference between a measurement and a vibe.
    resolution = 1.0 / questions if questions else 1.0
    print(f"\n{questions} questions: the smallest meaningful hit-rate difference is "
          f"{resolution:.3f} (one question).")
    if len(results) > 1:
        gap = abs(results[1]["metrics"]["hit_rate_at_k"]
                  - baseline["metrics"]["hit_rate_at_k"])
        if gap < resolution * 2:
            print("This gap is within noise for a set of this size. Either strategy is a "
                  "defensible choice on this evidence; add questions before deciding.")

    print("\nRemember: hit-rate here is dense retrieval only (no BM25), on exact search. "
          "It measures chunking, not the full pipeline.")


def _warn_if_fixture_set(questions: List[Dict], documents: Dict[str, List[Dict]]) -> None:
    """The numbers are worthless if the questions do not target the documents loaded."""
    expected = {q.get("expected_source") for q in questions if q.get("expected_source")}
    missing = sorted(name for name in expected if name not in documents)
    if missing:
        print("\n!! These questions expect documents that were not loaded:")
        for name in missing:
            print(f"!!   {name}")
        print("!! Every one of them can only score as a MISS, for both strategies, which")
        print("!! makes the comparison meaningless. Point --docs at them, or replace")
        print("!! eval/golden_questions.json with questions about the documents you loaded.\n")
    unreviewed = [q for q in questions if q.get("reviewed") is False]
    if unreviewed:
        print(f"!! {len(unreviewed)}/{len(questions)} questions are unreviewed drafts - "
              "edit them before trusting any of this.\n")


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare chunking strategies on retrieval quality, offline.")
    parser.add_argument("--set", default=str(PROJECT_ROOT / "eval" / "golden_questions.json"))
    parser.add_argument("--docs", nargs="*", default=[str(DATA_DIR / "*.pdf")],
                        help="PDF paths or globs (default: everything in data/)")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="only the first N pages of each document - a 1000-page book "
                             "takes a long time to embed sentence by sentence")
    parser.add_argument("--strategies", default="fixed,semantic",
                        help="comma-separated; the first is the baseline everything else "
                             "is compared against")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--candidates", type=int, default=RETRIEVAL_CANDIDATES)
    parser.add_argument("--expand", type=int, default=NEIGHBOR_EXPANSION)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_WORDS)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP_WORDS)
    parser.add_argument("--percentile", type=float, default=SEMANTIC_BREAKPOINT_PERCENTILE)
    parser.add_argument("--buffer", type=int, default=SEMANTIC_BUFFER_SIZE)
    parser.add_argument("--min-words", type=int, default=SEMANTIC_MIN_CHUNK_WORDS)
    parser.add_argument("--max-words", type=int, default=SEMANTIC_MAX_CHUNK_WORDS)
    parser.add_argument("--out", help="write the full result to this JSON file")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in ("fixed", "semantic")]
    if unknown:
        print(f"Unknown strategy: {', '.join(unknown)}")
        return 1

    data = json.loads(Path(args.set).read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"questions": data}
    questions = data.get("questions", [])
    if not questions:
        print(f"No questions in {args.set}")
        return 1

    if EMBEDDINGS_PROVIDER != "local":
        print(f"\n!! EMBEDDINGS_PROVIDER={EMBEDDINGS_PROVIDER}. Semantic chunking embeds "
              "every sentence, so this run will spend a lot of API quota.")
        print("!! Set EMBEDDINGS_PROVIDER=local in .env for the experiment.\n")

    print("Extracting documents...")
    documents = resolve_documents(args.docs, args.max_pages or None)
    if not documents:
        print("No readable PDFs matched --docs.")
        return 1
    _warn_if_fixture_set(questions, documents)

    print(f"\ntop_k={args.top_k}  candidates={args.candidates}  expand={args.expand}  "
          f"rerank={not args.no_rerank}")
    print(f"fixed: {args.chunk_size}w/{args.overlap}w   "
          f"semantic: p{args.percentile:g} buffer={args.buffer} "
          f"{args.min_words}-{args.max_words}w")

    results = []
    for strategy in strategies:
        print(f"\n--- {strategy} ---")
        index = build_index(documents, strategy, args)
        print(f"  {index['stats']['chunks']} chunks, "
              f"median {index['stats']['words_median']} words, "
              f"chunked in {index['stats']['chunk_seconds']}s, "
              f"embedded in {index['stats']['embed_seconds']}s")
        metrics = score(index, questions, args)
        for row in metrics["rows"]:
            print(f"  [{'HIT ' if row['hit'] else 'MISS'}] rank={row['rank'] or '-':<3} "
                  f"{row['question'][:60]}")
        results.append({"strategy": strategy, "stats": index["stats"], "metrics": metrics})
        # The vectors are the big object; drop them before the next strategy is built so a
        # two-strategy run over a large corpus does not hold both in memory at once.
        del index

    report(results, len(questions), args)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "config": vars(args),
            "documents": {name: len(pages) for name, pages in documents.items()},
            "results": results,
        }, indent=2), encoding="utf-8")
        print(f"\nFull results -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
