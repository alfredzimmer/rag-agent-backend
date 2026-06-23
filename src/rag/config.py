"""
Configuration classes for the RAG system.
"""
import os
from dataclasses import dataclass


@dataclass
class RAGConfig:
    vector_store_type: str = "milvus"
    collection_name: str = "HeaderInContentTrial_Dense"
    ranker_type: str = "none"
    dense_embedding_model: str = os.getenv("DENSE_EMBEDDING_MODEL", "nomic-embed-text:latest")
    sparse_embedding_model: str = "none"  # [splade, bm25, bge, none]
    llm_model: str = os.getenv("RAG_LLM_MODEL", "qwen3.6:latest")
    llm_num_predict: int = int(os.getenv("RAG_LLM_NUM_PREDICT", "4096"))
    simple_rag: bool = os.getenv("SIMPLE_RAG", "True").lower() == "true"
    summarization_model: str = os.getenv("SUMMARIZATION_MODEL", "qwen3.6:latest")
    title_llm_model: str = os.getenv("TITLE_LLM_MODEL", "qwen3.6:latest")
    hyde: bool = False
    similarity_threshold: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.35"))
    max_context_tokens: int = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "3000"))
