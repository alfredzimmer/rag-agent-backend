from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Lazy initialization of embedding model
_embedding_model = None

def _get_embedding_model() -> OpenAIEmbeddings:
    """
    Get or create the embedding model instance (lazy initialization).
    """
    global _embedding_model
    if _embedding_model is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or as an environment variable."
            )
        _embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=api_key
        )
    return _embedding_model

def embed_chunks(chunks: list[Document]) -> tuple[list[str], list[dict], list[list[float]]]:
    """
    Embed a list of chunks.
    """
    idx_to_content = [chunk.page_content for chunk in chunks]
    idx_to_metadata = [chunk.metadata for chunk in chunks]

    embedding_model = _get_embedding_model()
    return idx_to_content, idx_to_metadata, embedding_model.embed_documents(idx_to_content)