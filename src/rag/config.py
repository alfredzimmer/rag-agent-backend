"""
Configuration classes for the RAG system.
"""
from dataclasses import dataclass


@dataclass
class RAGConfig:
    vector_store_type: str = "milvus" 
    collection_name: str = "testing" 
    ranker_type: str = "bge"
    dense_embedding_model: str = "qwen3-embedding:8b"
    sparse_embedding_model: str = "splade"  # [splade, bm25, bge]
    llm_model: str = "qwen3:8b"
    memory_llm_model: str = "qwen3:8b"
    memory_embeddings_model: str = "nomic-embed-text"
    memory_embeddings_dims: int = 768
    training_mode: bool = False
    training_llm_model: str = "qwen3:8b"
    hyde: bool = False


@dataclass
class Context:
    user_id: str
