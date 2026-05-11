from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Qdrant
    qdrant_url: str
    qdrant_api_key: str = ""
    qdrant_collection: str = "sat_tutor"

    # Embedding
    embed_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5

     # Observability - Langfuse
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"
    
    # LLM
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 3
    reranker_fetch_k: int = 10
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()