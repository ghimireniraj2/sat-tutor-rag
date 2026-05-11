"""
LLM client wrapper.
Thin layer over Groq API — keeps model config in one place and provides
a consistent interface for both streaming and non-streaming responses.
"""
import instructor
from groq import Groq
from pydantic import BaseModel
from config import settings

_client = None
_instructor_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def get_instructor_client():
    global _instructor_client
    if _instructor_client is None:
        _instructor_client = instructor.from_groq(
            get_client(),
            mode=instructor.Mode.JSON,
        )
    return _instructor_client


def complete(messages: list[dict], max_tokens: int = 1000) -> str:
    """Non-streaming completion — returns text."""
    response = get_client().chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def complete_structured(
    messages: list[dict],
    response_model: type[BaseModel],
    max_tokens: int = 1000,
    max_retries: int = 3,
) -> BaseModel:
    """
    Structured completion with Instructor.
    Automatically retries if LLM returns malformed output.
    Returns a validated Pydantic model instance.
    """
    client = get_instructor_client()
    return client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        response_model=response_model,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )

def complete_ollama(messages: list[dict], max_tokens: int = 1000) -> str:
    """
    Complete using local Ollama — for benchmarking only.
    Not used in production pipeline.
    """
    import requests
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2",
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        },
    )
    return response.json()["message"]["content"]