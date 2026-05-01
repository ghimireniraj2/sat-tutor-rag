### SAT Tutor RAG System (Enterprise-Grade AI Learning Assistant)

A production-grade Retrieval-Augmented Generation (RAG) system that acts as an intelligent SAT tutor. Built as a learning project with real production tools and practices — not simplified toy implementations.
The system helps students learn SAT Math, Reading, and Grammar through concept explanations, step-by-step problem solving, and practice question generation.
---
Goals
Primary: Build a working SAT tutoring assistant powered by RAG.
Secondary (equally important): Learn production-grade AI engineering practices by using real tools the way they are used in production — proper observability, eval pipelines, structured outputs, and deployment discipline.
---
System Architecture
```
Vite + React (Vercel)
        ↓
FastAPI Backend (Render)
        ↓
LlamaIndex (RAG Orchestration)
        ↓
Qdrant (Vector Database — local Docker / Qdrant Cloud)
        ↓
sentence-transformers BAAI/bge-small-en-v1.5 (Embeddings)
        ↓
Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
        ↓
Groq API (Production LLM) / Ollama (Local Experiment)
        ↓
Langfuse (Observability + Tracing)
```
---
Tech Stack
Frontend
Vite + React — lightweight, fast builds, pure static output
React Router v6 — client-side routing
TailwindCSS — styling
shadcn/ui — accessible component primitives
EventSource (SSE) — streaming responses from FastAPI (Phase 4)
Why not Next.js: SSR adds complexity with no benefit for a chat interface. Vite builds to static files, which deploy instantly on Vercel with no server-side compute and no function execution limits.
Backend
FastAPI — async Python API framework with native SSE support
Pydantic + Pydantic Settings — request validation and typed config management
Uvicorn — ASGI server
python-dotenv — environment variable loading
FastAPI is structured with proper router separation, dependency injection, and lifespan events for model loading — not flattened into a single file.
RAG Orchestration
LlamaIndex — document ingestion, chunking, index management, query pipeline
Vector Database
Qdrant — local via Docker for development, Qdrant Cloud free tier for deployment
Metadata filtering enabled (topic, difficulty, content type)
Embedding
BAAI/bge-small-en-v1.5 via sentence-transformers
Chosen once, not changed mid-project — switching models requires full re-indexing
For deployment: Hugging Face Inference API to avoid loading the model in memory on the host.
Reranker
cross-encoder/ms-marco-MiniLM-L-6-v2
Applied post-retrieval to reorder top-k results before prompt assembly
Added in Phase 3 after baseline eval establishes a quality benchmark to beat
LLM
Groq API — primary LLM for all deployed environments (fast, cheap, OpenAI-compatible)
O
llama (Llama 3 / Mistral 7B) — local experimentation only, not deployed
Why Ollama stays local: No GPU on the deployment host, ephemeral storage, and meaningful quality gap on multi-step math reasoning vs a hosted model. Ollama's value is educational — running it locally teaches inference, context windows, and the quality ceiling of small models firsthand.
Structured Outputs
Instructor — enforces Pydantic-validated JSON output from LLM responses
Required for Practice mode where frontend renders question, choices, and explanation as distinct UI elements rather than a markdown blob
Observability
Langfuse — trace every query end-to-end: retrieved chunks, prompt content, token counts, latency
Added from Phase 2 onwards — not an afterthought
Without this, debugging retrieval failures means guessing
Deployment
Vercel — frontend static hosting, zero config, auto-deploys on git push
Render — FastAPI backend, Docker container, free tier
Qdrant Cloud — persistent vector index, free 1GB tier, independent of compute restarts
Docker Compose — full local stack (Qdrant + FastAPI + Ollama), ensures dev/prod parity on Windows
---
Development Environment
Development is on Windows targeting Linux deployment. All backend services run inside Docker to avoid Windows/Linux path and dependency inconsistencies.
Add this `.gitattributes` immediately to prevent CRLF line ending issues:
```
* text=auto eol=lf
*.py text eol=lf
*.sh text eol=lf
```
Run the full local stack with:
```bash
docker compose up
```
WSL2 is recommended for running CLI tools and scripts outside Docker.
---
Project Structure
```
sat-math-tutor/
├── backend/
│   ├── main.py          # FastAPI app, lifespan, router registration
│   ├── routers/
│   │   └── query.py     # /query endpoint
│   ├── retrieval.py     # Qdrant queries, reranker
│   ├── prompts.py       # Prompt templates per mode
│   ├── embed.py         # Embedding logic
│   ├── ingest.py        # Ingestion pipeline (rerunnable, idempotent)
│   ├── eval.py          # Offline evaluation script
│   ├── config.py        # Pydantic Settings
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── main.jsx
│   └── vite.config.js
├── data/
│   ├── raw/             # Source documents
│   └── chunks/          # Processed chunks (committed for reproducibility)
├── evals/
│   └── eval_set.json    # 50 question eval set with expected behavior
├── docker-compose.yml
├── .gitattributes
└── .env.example
```
Structure is kept flat and readable. Refactor into deeper modules only when a file genuinely becomes hard to navigate.
---
Data Sources
Only open-license, bulk-downloadable sources are used.
Primary
OpenStax Algebra, Geometry, Statistics (PDFs — genuinely open license)
Public SAT question datasets (GitHub repositories)
MVP target: 20–50 documents, ~300–800 chunks, SAT Math + Reading coverage.
Chunking strategy: Document-aware splitting on question/problem boundaries using LlamaIndex's `SentenceSplitter` with large overlap. Fixed-size chunking is not used — it splits multi-step math problems mid-solution.
---
RAG Pipeline
```
1. User submits question
2. Query embedded with bge-small-en-v1.5
3. Qdrant retrieves top-k chunks (with optional metadata filtering)
4. Cross-encoder reranker reorders results
5. Conversation history (last 4 turns) prepended to context
6. Mode-specific prompt template assembled
7. Groq LLM generates response
8. Structured output validated via Instructor (Practice mode)
9. Full trace logged to Langfuse
```
---
Tutor Modes
Two modes in MVP. Additional modes added in v2.
Explain Mode
Step-by-step concept explanations
Socratic prompt — guide toward understanding, don't just answer
Worked examples from retrieved context
Practice Mode
Generates SAT-style questions
Returns structured JSON: question, answer choices, correct answer, explanation
Instructor enforces output schema so frontend renders components, not markdown
Deferred to v2: Mistake Analysis mode, Test mode, adaptive difficulty. These require either structured error detection logic (Mistake Analysis) or scoring infrastructure (Test) that would dilute focus during the MVP phase.
---
Prompt Design
Each mode has a dedicated prompt template in `prompts.py`. Templates are string constants during development — no database or config-driven prompt system until there is a clear need for one. This keeps iteration fast.
System prompt principles:
Enforce Socratic behavior in Explain mode (guide, don't just solve)
Return structured JSON in Practice mode
Adapt explanation depth based on question complexity
---
Observability
Langfuse is integrated from Phase 2. Every query produces a trace containing:
Retrieved chunk content and scores
Reranked order and score delta
Full prompt sent to LLM
Token counts and latency
LLM response
This is how retrieval quality problems are diagnosed. Without it, debugging is guesswork.
---
Evaluation
An offline eval set of 50 SAT questions with documented expected behavior lives in `evals/eval_set.json`. The eval script in `eval.py` runs against this set and outputs:
Retrieval hit rate (is relevant context in top-k?)
MRR (Mean Reciprocal Rank)
Manual review flags for qualitative failures
The reranker is added in Phase 3 only after baseline eval scores are established. The eval is rerun after adding the reranker to confirm it actually improves retrieval — not assumed.
---
Configuration
Pydantic Settings with `.env` file. All secrets via environment variables — never hardcoded.
```python
class Settings(BaseSettings):
    groq_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    langfuse_secret_key: str
    langfuse_public_key: str
    embed_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 5

    model_config = ConfigDict(env_file=".env")
```
Copy `.env.example` to `.env` and fill in values. Never commit `.env`.
---
CORS
Frontend and backend are on separate domains. FastAPI CORS middleware is configured on day one:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```
`allow_origins=["*"]` is not used even in development.
---
