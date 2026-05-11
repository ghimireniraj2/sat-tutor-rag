"""
Eval pipeline — measures retrieval and end-to-end response quality.

Usage:
    python eval.py --test retrieval          # retrieval hit rate (no reranker)
    python eval.py --test retrieval --k 5   # test with top-5 retrieval
    python eval.py --test compare           # before/after reranker comparison (Step 4+)
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass

from retrieval import retrieve
from config import settings

# retrieve_reranked is added in Step 4 — import conditionally so eval
# runs correctly before the reranker is implemented
try:
    from retrieval import retrieve_reranked
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

EVAL_DIR = Path(__file__).parent.parent / "evals"


@dataclass
class EvalResult:
    id: int
    query: str
    passed: bool
    topic_hit: bool
    keyword_hits: list[str]
    expected_keywords: list[str]
    top_scores: list[float]
    top_topics: list[str]
    skipped: bool = False


def run_retrieval_eval(top_k: int = 3, use_reranker: bool = False) -> list[EvalResult]:
    """Run retrieval evaluation against the full eval set."""
    tests = json.loads((EVAL_DIR / "eval_set.json").read_text())
    results = []

    for test in tests:
        # Skip graph-dependent questions
        if test.get("graph_dependent", False):
            results.append(EvalResult(
                id=test["id"],
                query=test["query"],
                passed=False,
                topic_hit=False,
                keyword_hits=[],
                expected_keywords=test["expected_keywords"],
                top_scores=[],
                top_topics=[],
                skipped=True,
            ))
            continue

        if use_reranker and RERANKER_AVAILABLE:
            retrieved = retrieve_reranked(test["query"], top_k=top_k)
        else:
            if use_reranker and not RERANKER_AVAILABLE:
                print("  Note: reranker not yet implemented, using standard retrieval")
            retrieved = retrieve(test["query"], top_k=top_k)

        top_text = " ".join(r["text"].lower() for r in retrieved)
        top_topics = [r["metadata"].get("topic", "") for r in retrieved]

        keyword_hits = [
            kw for kw in test["expected_keywords"]
            if kw.lower() in top_text
        ]
        topic_hit = any(t in top_topics for t in test["expected_topics"])
        passed = len(keyword_hits) >= 2 and topic_hit

        results.append(EvalResult(
            id=test["id"],
            query=test["query"],
            passed=passed,
            topic_hit=topic_hit,
            keyword_hits=keyword_hits,
            expected_keywords=test["expected_keywords"],
            top_scores=[round(r["score"], 3) for r in retrieved],
            top_topics=top_topics,
        ))

    return results


def print_results(results: list[EvalResult], label: str = ""):
    """Print eval results with summary statistics."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    active = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]
    passed = [r for r in active if r.passed]

    for r in active:
        status = "✓" if r.passed else "✗"
        print(f"{status} [{r.id:02d}] {r.query[:55]}")
        if not r.passed:
            print(f"      Topics: expected {r.expected_keywords[:2]}, "
                  f"got {r.top_topics}")
            print(f"      Keywords: {r.keyword_hits} / {r.expected_keywords}")

    print(f"\nResults: {len(passed)}/{len(active)} passed "
          f"({len(passed)/len(active):.0%})")
    print(f"Skipped (graph-dependent): {len(skipped)}")

    if len(passed) / len(active) >= 0.8:
        print("✓ Hit rate target met (>=80%)")
    else:
        print("✗ Below 80% threshold")

    return len(passed) / len(active)


def run_comparison():
    """Run eval with and without reranker and compare."""
    print("Running baseline (no reranker)...")
    baseline = run_retrieval_eval(use_reranker=False)
    baseline_rate = print_results(baseline, "Baseline — No Reranker")

    print("\nRunning with reranker...")
    reranked = run_retrieval_eval(use_reranker=True)
    reranked_rate = print_results(reranked, "With Reranker")

    print(f"\n{'='*60}")
    print(f"  Comparison Summary")
    print(f"{'='*60}")
    print(f"Baseline:  {baseline_rate:.1%}")
    print(f"Reranked:  {reranked_rate:.1%}")
    delta = reranked_rate - baseline_rate
    direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    print(f"Delta:     {direction} {abs(delta):.1%}")

    # Find questions where reranker changed the outcome
    changes = []
    for b, r in zip(baseline, reranked):
        if b.passed != r.passed:
            changes.append((b, r))

    if changes:
        print(f"\nQuestions where outcome changed ({len(changes)}):")
        for b, r in changes:
            change = "PASS→FAIL" if b.passed else "FAIL→PASS"
            print(f"  [{change}] {b.query[:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["retrieval", "compare"],
                        default="retrieval")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--reranker", action="store_true")
    args = parser.parse_args()

    if args.test == "retrieval":
        results = run_retrieval_eval(top_k=args.k, use_reranker=args.reranker)
        print_results(results)
    elif args.test == "compare":
        run_comparison()