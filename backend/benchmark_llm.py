# backend/benchmark_llm.py
"""
Compare Groq vs Ollama response quality on 5 eval questions.
Run once to understand the quality gap between models.
"""

from retrieval import retrieve
from prompts import build_explain_prompt
from llm import complete, complete_ollama

BENCHMARK_QUERIES = [
    "How do I solve a quadratic equation by factoring?",
    "What is standard deviation?",
    "How do I find the slope of a line from two points?",
    "What is the difference between mean and median?",
    "How do I solve systems of equations by substitution?",
]

for query in BENCHMARK_QUERIES:
    chunks = retrieve(query, top_k=3)
    messages = build_explain_prompt(query, chunks, history=[])

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    print("\n--- Groq (llama-3.3-70b) ---")
    groq_response = complete(messages)
    print(groq_response[:500])

    print("\n--- Ollama (llama3.2) ---")
    ollama_response = complete_ollama(messages)
    print(ollama_response[:500])