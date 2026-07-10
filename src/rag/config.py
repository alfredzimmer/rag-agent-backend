"""Configuration for the RAG agent.

Every value is read from the environment at instantiation time (after
load_dotenv has run), so the defaults here describe a plain local setup:
Milvus and Ollama on localhost.
"""
import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RAGConfig:
    collection_name: str = field(
        default_factory=lambda: os.getenv("RAG_COLLECTION_NAME", "rag_documents")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("RAG_LLM_MODEL", "qwen3.6:latest")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("DENSE_EMBEDDING_MODEL", "nomic-embed-text:latest")
    )
    llm_num_predict: int = field(
        default_factory=lambda: int(os.getenv("RAG_LLM_NUM_PREDICT", "8192"))
    )
    llm_reasoning: bool = field(
        default_factory=lambda: _env_bool("RAG_LLM_REASONING", False)
    )
    llm_num_ctx: int = field(
        default_factory=lambda: int(os.getenv("RAG_LLM_NUM_CTX", "16384"))
    )
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "10")))
    search_ef: int = field(default_factory=lambda: int(os.getenv("RAG_SEARCH_EF", "96")))
    # Two-stage retrieval (see rag.retrieval). "none" keeps the original
    # single-pass dense search; "mmr" and "llm" over-fetch fetch_k candidates
    # and re-select/reorder down to top_k.
    rerank_backend: str = field(
        default_factory=lambda: os.getenv("RAG_RERANK_BACKEND", "none")
    )
    fetch_k: int = field(default_factory=lambda: int(os.getenv("RAG_FETCH_K", "40")))
    mmr_lambda: float = field(
        default_factory=lambda: float(os.getenv("RAG_MMR_LAMBDA", "0.5"))
    )
    milvus_uri: str = field(
        default_factory=lambda: os.getenv("MILVUS_URI", "http://localhost:19530")
    )
    milvus_db: str = field(default_factory=lambda: os.getenv("MILVUS_DB", "default"))
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
