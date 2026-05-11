## Eval Results

### Phase 3 Baseline (no reranker)
Date: 2026-05-02
Hit rate: 46/50 (92%)
top_k: 3
Notes: 4 failures are eval set topic mismatches, not retrieval failures

### Phase 3 With Reranker
Date: 2026-05-02
Hit rate: 45/50 (90%)
top_k: 3 (reranked from fetch_k=10)
Model: cross-encoder/ms-marco-MiniLM-L-6-v2
Delta: -2% vs baseline
Finding: General-purpose cross-encoder slightly hurts on educational content.
Consider domain-specific reranker in future or tune fetch_k parameter.