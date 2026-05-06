"""
LLM client wrapper.
Thin layer over Groq API — keeps model config in one place and provides
a consistent interface for both streaming and non-streaming responses.
"""

from groq import Groq
from config import settings

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def complete(messages: list[dict], max_tokens: int = 1000) -> str:
    """
    Send messages to Groq and return the response text.
    Non-streaming — returns complete response at once.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def complete_json(messages: list[dict], max_tokens: int = 1000) -> str:
    """
    Send messages to Groq with JSON mode enforced.
    Use for Practice mode where structured output is required.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content