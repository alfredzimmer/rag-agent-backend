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
    llm_num_predict: int = int(os.getenv("RAG_LLM_NUM_PREDICT", "1024"))
    summarization_model: str = os.getenv("SUMMARIZATION_MODEL", "qwen3.6:latest")
    memory_llm_model: str = os.getenv("MEMORY_LLM_MODEL", "qwen3.6:latest")
    title_llm_model: str = os.getenv("TITLE_LLM_MODEL", "qwen3.6:latest")
    memory_embeddings_model: str = os.getenv("MEMORY_EMBEDDINGS_MODEL", "nomic-embed-text:latest")
    memory_embeddings_dims: int = 768
    manager_instructions = """
You are a memory manager. Your job is to extract LONG-TERM knowledge from the conversation.

RULES FOR STORAGE:
1. **IGNORE CHIT-CHAT**: Do not store greetings, pleasantries, or temporary context (e.g., "I'm going to lunch now").
2. **Relevance Filter**: Only store information if it is useful for future tasks (e.g., user preferences, project specs, debugging history).
3. **COMPACTNESS**: Do not store raw quotes. Summarize the user's intent into the smallest possible sentence.
4. **No Duplicates**: If a fact already exists in the provided existing memories, do not create a new one.
"""

    training_llm_model: str = os.getenv("TRAINING_LLM_MODEL", "qwen3.6:latest")
    hyde: bool = False


@dataclass
class Context:
    user_id: str
