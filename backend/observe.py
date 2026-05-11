from langfuse import Langfuse
from config import settings

_client = None


def get_langfuse() -> Langfuse | None:
    global _client
    if _client is None:
        if not settings.langfuse_secret_key:
            return None
        _client = Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_base_url,
        )
    return _client


def start_trace(name: str, input: dict):
    lf = get_langfuse()
    if lf is None:
        return None
    return lf.trace(name=name, input=input)


def log_retrieval(trace, chunks: list[dict]):
    if trace is None:
        return
    trace.span(
        name="retrieval",
        output={
            "chunks": [
                {
                    "text": c["text"][:200],
                    "vector_score": c.get("vector_score", c.get("score")),
                    "reranker_score": c.get("reranker_score"),
                    "topic": c["metadata"].get("topic"),
                }
                for c in chunks
            ]
        },
    )


def log_llm(trace, prompt: str, response: str, model: str):
    if trace is None:
        return
    trace.generation(
        name="llm",
        model=model,
        input=prompt,
        output=response,
    )


def finish_trace(trace, output: str):
    if trace is None:
        return
    trace.update(output=output)
    get_langfuse().flush()