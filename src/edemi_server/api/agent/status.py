from fastapi import APIRouter

router = APIRouter(
    prefix="/api/agent/status"
)

@router.get("/")
def get_status():
    """
    Get information about available models and current configuration
    """
    return {
        "status": "ok",
        "available_models": {
            "llm": ["qwen3:8b"],
            "embedding": ["qwen3-embedding:8b"],
            "sparse_embedding": ["splade", "bm25", "bge"]
        },
        "default_config": {
            "vector_store_type": "milvus",
            "ranker_type": "bge",
            "dense_embedding_model": "qwen3-embedding:8b",
            "sparse_embedding_model": "splade",
            "llm_model": "qwen3:8b",
            "hyde": True
        }
    }
