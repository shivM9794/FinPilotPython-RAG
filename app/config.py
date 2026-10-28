
import os
from pydantic import BaseModel


class Settings(BaseModel):
    app_env: str = os.getenv("APP_ENV", "development")

    qdrant_url: str = os.getenv(
        "QDRANT_URL", "http://localhost:6333"
    )
    qdrant_collection: str = os.getenv(
        "QDRANT_COLLECTION", "finpilot_knowledge"
    )
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    rag_top_k_dense: int = int(
        os.getenv("RAG_TOP_K_DENSE", "8")
    )
    rag_top_k_sparse: int = int(
        os.getenv("RAG_TOP_K_SPARSE", "8")
    )
    rag_final_top_k: int = int(
        os.getenv("RAG_FINAL_TOP_K", "5")
    )
    max_upload_bytes: int = int(
        os.getenv("MAX_UPLOAD_BYTES", "15000000")
    )


settings = Settings()
