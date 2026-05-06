# backend/test_prompts.py
from groq import Groq
from config import settings
from retrieval import retrieve
from prompts import build_explain_prompt, build_practice_prompt

client = Groq(api_key=settings.groq_api_key)

def test_explain(query: str):
    chunks = retrieve(query, top_k=3)
    messages = build_explain_prompt(query, chunks, history=[])
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        max_tokens=500,
    )
    print(f"\nQuery: {query}")
    print(f"\nResponse:\n{response.choices[0].message.content}")
    print(f"\nChunks used:")
    for c in chunks:
        print(f"  [{c['score']:.3f}] {c['text'][:100]}")

def test_practice(topic: str):
    chunks = retrieve(topic, top_k=3)
    messages = build_practice_prompt(topic, chunks, history=[])
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        max_tokens=500,
    )
    print(f"\nTopic: {topic}")
    print(f"\nResponse:\n{response.choices[0].message.content}")

if __name__ == "__main__":
    test_explain("How do I solve a quadratic equation by factoring?")
    test_explain("What is the difference between mean and median?")
    test_practice("algebra")
    test_practice("statistics")