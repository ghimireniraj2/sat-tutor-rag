"""
Usage: python eval.py --test retrieval
"""
import json
import argparse
from pathlib import Path
from retrieval import retrieve

EVAL_DIR = Path(__file__).parent.parent / "evals"


def run_retrieval_eval():
    tests = json.loads((EVAL_DIR / "retrieval_test.json").read_text())
    passed = 0

    for test in tests:
        retrieved = retrieve(test["query"], top_k=3)
        top_text = " ".join(r["text"].lower() for r in retrieved)
        top_topics = [r["metadata"].get("topic") for r in retrieved]
        keyword_hits = [kw for kw in test["expected_keywords"] if kw in top_text]
        topic_hit = any(t in top_topics for t in test.get("expected_topics", [test.get("expected_topic")]))
        required_hits = 1 if len(test["expected_keywords"]) <= 3 else 2
        hit = len(keyword_hits) >= required_hits and topic_hit

        if hit:
            passed += 1

        status = "✓" if hit else "✗"
        print(f"{status} [{test['id']}] {test['query'][:60]}")
        if not hit:
            print(f"  Expected topics: {test.get('expected_topics')}, got: {top_topics}")
            print(f"  Keyword hits   : {keyword_hits} / {test['expected_keywords']}")
            print(f"  Top chunk      : {retrieved[0]['text'][:150] if retrieved else 'NONE'}")

    hit_rate = passed / len(tests)
    print(f"\nHit rate: {passed}/{len(tests)} ({hit_rate:.0%})")
    if hit_rate >= 0.8:
        print("✓ Retrieval quality target met. Ready for Phase 2.")
    else:
        print("✗ Below 80% threshold. Investigate before proceeding.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["retrieval"], default="retrieval")
    args = parser.parse_args()
    if args.test == "retrieval":
        run_retrieval_eval()