from retrieval import retrieve

results = retrieve("How do I solve a quadratic equation by factoring?")
for r in results:
    print(f"Score: {r['score']:.3f} | Topic: {r['metadata'].get('topic')}")
    print(r['text'][:200])
    print("---")