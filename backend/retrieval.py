from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import settings

_index = None


def get_index() -> VectorStoreIndex:
    global _index
    if _index is None:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        vector_store = QdrantVectorStore(client=client, collection_name=settings.qdrant_collection)
        embed_model = HuggingFaceEmbedding(model_name=settings.embed_model)
        _index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return _index


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    index = get_index()
    nodes = index.as_retriever(similarity_top_k=top_k or settings.top_k).retrieve(query)
    return [{"text": n.get_content(), "score": n.score, "metadata": n.metadata} for n in nodes]


def retrieve_with_filter(
    query: str,
    topic: str | None = None,
    content_type: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    filters = []
    if topic:
        filters.append(FieldCondition(key="topic", match=MatchValue(value=topic)))
    if content_type:
        filters.append(FieldCondition(key="content_type", match=MatchValue(value=content_type)))

    index = get_index()
    retriever = index.as_retriever(
        similarity_top_k=top_k or settings.top_k,
        vector_store_kwargs={"qdrant_filters": Filter(must=filters) if filters else None},
    )
    nodes = retriever.retrieve(query)
    return [{"text": n.get_content(), "score": n.score, "metadata": n.metadata} for n in nodes]




from sentence_transformers import CrossEncoder

_reranker = None


def get_reranker() -> CrossEncoder:
    """Lazy load cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def retrieve_reranked(
    query: str,
    top_k: int | None = None,
    fetch_k: int | None = None,
) -> list[dict]:
    """
    Retrieve with cross-encoder reranking.

    Fetches fetch_k candidates via vector search, then reranks them
    with the cross-encoder and returns top_k results.

    The reranker sees (query, chunk) pairs and scores them directly —
    more accurate than vector similarity alone.
    """
    final_k = top_k or settings.reranker_top_k
    candidates_k = fetch_k or settings.reranker_fetch_k

    # Step 1: Fetch more candidates than we need
    candidates = retrieve(query, top_k=candidates_k)

    if not candidates:
        return []

    # Step 2: Score each (query, chunk) pair with cross-encoder
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    # Step 3: Attach reranker scores and sort
    for i, candidate in enumerate(candidates):
        candidate["reranker_score"] = float(scores[i])
        candidate["vector_score"] = candidate["score"]  # preserve original
        candidate["score"] = float(scores[i])           # replace with reranker score

    reranked = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)

    return reranked[:final_k]