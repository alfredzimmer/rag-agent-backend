from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
)

def embed_chunks(chunks: list[Document]):
    """
    Embed a list of chunks.
    """
    content = [chunk.page_content for chunk in chunks]
    idx_to_metadata = [chunk.metadata for chunk in chunks]

    return content, idx_to_metadata, embedding_model.embed_documents(content)