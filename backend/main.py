# backend/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.query import router as query_router
from retrieval import get_index
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the retrieval index on startup."""
    get_index()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}