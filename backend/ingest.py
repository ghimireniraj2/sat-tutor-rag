"""
Ingestion pipeline — rerunnable and idempotent.
Safe to run multiple times — upserts, does not duplicate.

Usage:
    python ingest.py                              # ingest all files in data/raw/
    python ingest.py --file openstax-algebra.pdf  # single file
    python ingest.py --reset                      # drop collection and reingest
"""

# --- Standard library imports ---
import argparse
import logging
import random
import re
import json
import uuid
from pathlib import Path

# --- LlamaIndex imports ---
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Qdrant imports ---
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

# --- Local imports ---
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Per-file topic mapping — extend as sources are added
FILE_TOPIC_MAP = {
    "openstax-algebra.pdf": "algebra",
    "openstax-statistics.pdf": "statistics",
    "openstax-prealgebra.pdf": "algebra",
}


# =============================================================================
# Template resolver — used by ingest_sat_json
# These functions must be defined before ingest_sat_json calls them
# =============================================================================

def _safe_eval(expr: str, values: dict) -> str:
    """
    Evaluate a template expression like {(a + c)} with resolved values.
    Uses restricted eval — no builtins, arithmetic only.
    Returns raw expression unchanged if eval fails rather than crashing.
    """
    for key, val in values.items():
        expr = expr.replace(key, str(val))
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return expr


def resolve_question(template: str, parameters: dict, n: int = 1) -> list[str]:
    instances = []
    for _ in range(n):
        values = {}
        skip = False
        for param, config in parameters.items():
            if not isinstance(config, dict) or "range" not in config:
                skip = True
                break
            range_val = config["range"]
            if not isinstance(range_val, list) or len(range_val) != 2:
                skip = True
                break
            low, high = range_val
            if isinstance(low, float) or isinstance(high, float):
                values[param] = round(random.uniform(low, high), 2)
            else:
                try:
                    values[param] = random.randint(int(low), int(high))
                except (TypeError, ValueError):
                    skip = True
                    break

        if skip:
            continue  # skip this question entirely

        result = template
        for param, value in values.items():
            result = result.replace(f"{{{param}}}", str(value))
        result = re.sub(
            r"\{([^}]+)\}",
            lambda m: _safe_eval(m.group(1), values),
            result
        )
        instances.append(result)
    return instances


def load_sat_questions(file_path: Path) -> list[dict]:
    """
    Parse sat-questions.json and return a flat list of resolved question dicts
    with metadata from the JSON structure preserved.

    The JSON is nested: topics → levels (difficulty) → questions (templates).
    This function walks the tree recursively, collecting topic and difficulty
    context as it descends, then resolves each template into n concrete instances.
    """
    data = json.loads(file_path.read_text(encoding="utf-8"))
    records = []

    def walk(node: dict, topic: str = "", difficulty: str = ""):
        for key, value in node.items():

            if key == "questions":
                for item in value:
                    # handle both flat list of dicts AND list of lists
                    questions = item if isinstance(item, list) else [item]
                    for q in questions:
                        if not isinstance(q, dict) or "template" not in q:
                            continue
                        for instance in resolve_question(
                            q["template"], q.get("parameters", {}), n=1
                        ):
                            records.append({
                                "text": instance,
                                "metadata": {
                                    "source": file_path.name,
                                    "content_type": "practice_question",
                                    "topic": topic or "algebra",
                                    "difficulty": difficulty,
                                    "page": 0,
                                    "tags": q.get("tags", []),
                                    "cognitive_level": q.get("cognitive_level", ""),
                                    "question_id": q.get("id", ""),
                                }
                            })

            elif key == "levels":
                # Children are difficulty labels: easy, medium, hard
                for diff_key, level_data in value.items():
                    walk(level_data, topic=topic, difficulty=diff_key)

            elif isinstance(value, dict):
                if not value or key == "dataset":
                    continue
                walk(value, topic=key, difficulty=difficulty)

    walk(data.get("topics", {}))
    return records


# =============================================================================
# Qdrant setup
# =============================================================================

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def ensure_collection(client: QdrantClient, reset: bool = False):
    """Create collection if it doesn't exist. Optionally reset."""
    exists = client.collection_exists(settings.qdrant_collection)
    if exists and reset:
        log.info(f"Resetting collection: {settings.qdrant_collection}")
        client.delete_collection(settings.qdrant_collection)
        exists = False
    if not exists:
        log.info(f"Creating collection: {settings.qdrant_collection}")
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )


# =============================================================================
# Ingestion functions
# =============================================================================

def _index_documents(docs: list, client: QdrantClient):
    """
    Chunk, embed, and upsert documents directly to Qdrant.
    Bypasses LlamaIndex vector store wiring — uses Qdrant client directly.
    """
    from embed import embed_batch

    # Step 1 — Chunk documents using LlamaIndex splitter
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(docs, show_progress=True)
    log.info(f"Created {len(nodes)} chunks")

    if not nodes:
        log.warning("No chunks created — skipping upsert")
        return

    # Step 2 — Embed and upsert in batches
    BATCH_SIZE = 64
    total_upserted = 0

    for i in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[i:i + BATCH_SIZE]
        texts = [node.get_content() for node in batch]

        # Embed the batch
        vectors = embed_batch(texts)

        # Build Qdrant points
        points = []
        for node, vector, text in zip(batch, vectors, texts):
            payload = {"text": text}
            payload.update(node.metadata)
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            ))

        # Upsert directly to Qdrant
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )
        total_upserted += len(points)
        log.info(f"Upserted {total_upserted}/{len(nodes)} chunks")

    log.info(f"Done — {total_upserted} chunks written to Qdrant")


def ingest_pdf(file_path: Path, topic: str, client: QdrantClient):
    """Ingest a PDF — extracts text via LlamaIndex SimpleDirectoryReader."""
    log.info(f"Ingesting PDF: {file_path.name} | topic={topic}")
    docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
    for doc in docs:
        doc.metadata.update({
            "source": file_path.name,
            "content_type": "concept",
            "topic": topic,
            "difficulty": "",
            "tags": [],
        })
    _index_documents(docs, client)
    log.info(f"Done: {len(docs)} pages from {file_path.name}")



def ingest_sat_json(file_path: Path, client: QdrantClient):
    """Ingest SAT questions JSON — resolves templates before indexing."""
    log.info(f"Ingesting SAT JSON: {file_path.name}")
    records = load_sat_questions(file_path)   # calls resolver defined above
    docs = [Document(text=r["text"], metadata=r["metadata"]) for r in records]
    _index_documents(docs, client)
    log.info(f"Done: {len(docs)} resolved question instances")


# =============================================================================
# Entry point
# =============================================================================

def run(file: str | None = None, reset: bool = False):
    client = get_qdrant_client()
    ensure_collection(client, reset=reset)

    files = (
        [DATA_DIR / file] if file
        else [
            f for f in DATA_DIR.iterdir()
            if f.suffix in (".pdf", ".json") and f.name != ".gitkeep"
        ]
    )

    for f in files:
        if f.suffix == ".json":
            ingest_sat_json(f, client)
        elif f.suffix == ".pdf":
            ingest_pdf(f, FILE_TOPIC_MAP.get(f.name, "unknown"), client)

    info = client.get_collection(settings.qdrant_collection)
    log.info(f"Collection total: {info.points_count} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    run(file=args.file, reset=args.reset)