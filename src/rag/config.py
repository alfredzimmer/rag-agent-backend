"""
Configuration classes for the RAG system.
"""
from dataclasses import dataclass


@dataclass
class RAGConfig:
    vector_store_type: str = "milvus" 
    collection_name: str = "HeaderInContentTrial" 
    ranker_type: str = "bge"
    dense_embedding_model: str = "qwen3-embedding:8b"
    sparse_embedding_model: str = "splade"  # [splade, bm25, bge]
    llm_model: str = "qwen3.5:35b"
    summarization_model: str = "qwen3:8b"
    memory_llm_model: str = "qwen3:8b"
    title_llm_model: str = "qwen3:8b"
    memory_embeddings_model: str = "nomic-embed-text"
    memory_embeddings_dims: int = 768
    manager_instructions = """
You are a memory manager. Your job is to extract LONG-TERM knowledge from the conversation.

RULES FOR STORAGE:
1. **IGNORE CHIT-CHAT**: Do not store greetings, pleasantries, or temporary context (e.g., "I'm going to lunch now").
2. **Relevance Filter**: Only store information if it is useful for future tasks (e.g., user preferences, project specs, debugging history).
3. **COMPACTNESS**: Do not store raw quotes. Summarize the user's intent into the smallest possible sentence.
4. **No Duplicates**: If a fact already exists in the provided existing memories, do not create a new one.
"""
    training_llm_model: str = "qwen3:8b"
    hyde: bool = False


@dataclass
class Context:
    user_id: str
