"""
Evaluation harness - the thing that turns tuning from guesswork into measurement.

`tests/test_pipeline_offline.py` proves the plumbing works. This proves whether the
answers are any *good*, and - more usefully - whether a change made them better.

    python eval/run_eval.py                     # retrieval metrics only, no API key needed
    python eval/run_eval.py --judge             # + LLM-as-judge scoring (uses Groq)
    python eval/run_eval.py --no-rerank         # A/B: what does the cross-encoder buy?
    python eval/run_eval.py --no-hybrid         # A/B: what does BM25 buy?
    python eval/run_eval.py --top-k 8 --out runs/topk8.json

Metrics
-------
retrieval hit-rate@k   Did a chunk covering the expected page of the expected document
                       make it into the retrieved set? No LLM, no subjectivity, no cost.
                       This is the number to watch when changing chunking, the embedding
                       model, BM25 or the re-ranker.
MRR                    1/rank of the first correct chunk - rewards ranking it first, not
                       just retrieving it somewhere.
refusal rate           On deliberately unanswerable questions, did the system correctly
                       decline instead of inventing something?
groundedness /         LLM-as-judge, only with --judge. Modelled on the four axes in the
correctness /          reference project's evaluation notebook (which uses LangSmith; this
relevance              does the same thing with a strict rubric prompt to Groq).
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml import llm
from src.services import retrieval# noqa: E402
from src.core.config import GROQ_MODEL, TOP_K  # noqa: E402
from src.core.logging import get_logger  # noqa: E402

log = get_logger("eval")

DEFAULT_SET = Path(__file__).with_name("golden_questions.json")

JUDGE_PROMPT = """You are a strict grader for a retrieval-augmented question answering system.

Grade the ANSWER on three axes, each true or false:
- "correct": the ANSWER agrees with the REFERENCE on the facts asked about. Extra detail is
  fine; contradicting or missing the key fact is not. If no REFERENCE is given, judge
  whether the ANSWER is fully supported by the CONTEXT.
- "grounded": every factual claim in the ANSWER is supported by the CONTEXT. An answer that
  is correct but not derivable from the CONTEXT is NOT grounded.
- "relevant": the ANSWER actually addresses the QUESTION.

Reply with JSON only: {"correct": bool, "grounded": bool, "relevant": bool, "why": "one short sentence"}"""


def _covers(chunk: dict, source: str, pages: list) -> bool:
    """True if this chunk is from the expected document and covers an expected page."""
    if source and chunk.get("source") != source:
        return False
    start, end = chunk.get("page_start"), chunk.get("page_end")
    if start is None or end is None:
        return False
    return any(start <= page <= end for page in pages)


def judge(question: str, answer: str, context: str, reference: str) -> dict:
    if not llm.is_configured():
        return {"error": "no GROQ_API_KEY"}
    try:
        response = llm._client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": (
                    f"QUESTION:\n{question}\n\nREFERENCE:\n{reference or '(none given)'}\n\n"
                    f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
                )},
            ],
            temperature=0.0,
            max_tokens=250,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as exc:
        log.warning("Judge call failed: %s", exc)
        return {"error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure retrieval and answer quality.")
    parser.add_argument("--set", default=str(DEFAULT_SET), help="path to the golden question JSON")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--judge", action="store_true", help="also run LLM-as-judge scoring (needs GROQ_API_KEY)")
    parser.add_argument("--no-rerank", action="store_true", help="disable the cross-encoder for this run")
    parser.add_argument("--no-hybrid", action="store_true", help="disable BM25 fusion for this run")
    parser.add_argument("--no-expand", action="store_true", help="disable neighbour expansion for this run")
    parser.add_argument("--out", help="write full results to this JSON file")
    args = parser.parse_args()

    data = json.loads(Path(args.set).read_text(encoding="utf-8"))
    questions = data.get("questions", [])
    unanswerable = data.get("unanswerable", [])
    if not questions:
        print(f"No questions in {args.set}")
        return 1

    options = dict(
        use_rerank=not args.no_rerank,
        use_hybrid=not args.no_hybrid,
        expand=0 if args.no_expand else None,
    )
    print(f"Config: top_k={args.top_k}  rerank={options['use_rerank']}  "
          f"hybrid={options['use_hybrid']}  expand={'off' if args.no_expand else 'on'}  "
          f"judge={args.judge}\n")

    rows = []
    hits = 0
    reciprocal_ranks = []
    started = time.perf_counter()

    for item in questions:
        question = item["question"]
        source = item.get("expected_source")
        pages = item.get("expected_pages", [])

        chunks = retrieval.retrieve(question, top_k=args.top_k, **options)
        # Neighbour chunks count as retrieved, but rank is measured on the primary hits.
        primary = [c for c in chunks if "neighbor_of" not in c]

        rank = next((i + 1 for i, c in enumerate(primary) if _covers(c, source, pages)), None)
        hit = rank is not None or any(_covers(c, source, pages) for c in chunks)
        hits += bool(hit)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        row = {
            "question": question,
            "expected": {"source": source, "pages": pages},
            "hit": bool(hit),
            "rank": rank,
            "retrieved": [
                {"source": c["source"], "pages": [c["page_start"], c["page_end"]],
                 "rerank_score": c.get("rerank_score"), "similarity": c.get("similarity")}
                for c in primary
            ],
        }

        if args.judge:
            answer = llm.generate_answer(question, chunks)
            row["answer"] = answer
            row["judge"] = judge(question, answer, llm.build_context(chunks),
                                 item.get("reference_answer", ""))

        rows.append(row)
        mark = "HIT " if hit else "MISS"
        print(f"[{mark}] rank={rank or '-':<3} {question}")

    # Unanswerable questions: the system should decline, not invent.
    refusals = 0
    for item in unanswerable:
        question = item["question"]
        chunks = retrieval.retrieve(question, top_k=args.top_k, **options)
        answer = llm.generate_answer(question, chunks) if args.judge else None
        declined = not chunks or (
            answer is not None and any(
                phrase in answer.lower()
                for phrase in ("couldn't find", "does not contain", "doesn't contain",
                               "not contain", "no information", "not in the")
            )
        )
        refusals += bool(declined)
        rows.append({"question": question, "unanswerable": True, "declined": bool(declined),
                     "retrieved": len(chunks), "answer": answer})
        print(f"[{'OK  ' if declined else 'BAD '}] unanswerable: {question}")

    total = len(questions)
    summary = {
        "questions": total,
        "hit_rate_at_k": round(hits / total, 3) if total else 0.0,
        "mrr": round(sum(reciprocal_ranks) / total, 3) if total else 0.0,
        "unanswerable": len(unanswerable),
        "refusal_rate": round(refusals / len(unanswerable), 3) if unanswerable else None,
        "seconds": round(time.perf_counter() - started, 1),
        "config": {"top_k": args.top_k, **{k: v for k, v in options.items()}},
    }

    if args.judge:
        judged = [r["judge"] for r in rows if isinstance(r.get("judge"), dict) and "error" not in r["judge"]]
        if judged:
            for axis in ("correct", "grounded", "relevant"):
                summary[axis] = round(sum(bool(j.get(axis)) for j in judged) / len(judged), 3)

    print("\n" + "=" * 60)
    print(f"hit-rate@{args.top_k} : {summary['hit_rate_at_k']:.0%}   "
          f"MRR: {summary['mrr']:.3f}   "
          f"refusal: {summary['refusal_rate'] if summary['refusal_rate'] is None else format(summary['refusal_rate'], '.0%')}")
    if args.judge and "correct" in summary:
        print(f"judge - correct: {summary['correct']:.0%}  "
              f"grounded: {summary['grounded']:.0%}  relevant: {summary['relevant']:.0%}")
    print(f"({summary['seconds']}s)")
    print("=" * 60)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8")
        print(f"\nFull results -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
