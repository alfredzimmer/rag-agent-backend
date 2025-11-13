import json
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# Ensure external services can identify our requests when USER_AGENT is unset.
os.environ["USER_AGENT"] = "myagent"

# Debug: see exactly what URL we're using
print("QDRANT_URL =", repr(os.getenv("QDRANT_URL")))

qdrant_client = QdrantClient(
    host="qdrant.ziyutec.com",
    port=443,
    https=True,
    prefer_grpc=False,
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_size = len(embeddings.embed_query("sample text"))

collection_name = os.getenv("QDRANT_COLLECTION", "test")

# Ensure collection exists
if not qdrant_client.collection_exists(collection_name):
    print(f"Creating collection {collection_name} with size={vector_size}")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
else:
    print(f"Collection {collection_name} already exists")

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embeddings,
)


def load_chunk_file(path: str) -> list[Document]:
    """Convert a JSON chunk file into LangChain Documents."""
    chunk_path = Path(path)
    with chunk_path.open("r", encoding="utf-8") as fh:
        raw_chunks = json.load(fh)

    documents = []
    for chunk in raw_chunks:
        metadata = {
            "chunk_id": chunk["chunk_id"],
            "char_count": chunk["char_count"],
            **chunk.get("headers", {}),
        }
        docs.append(Document(page_content=chunk["content"], metadata=metadata))

    return documents

if __name__ == "__main__":
    chunk_file = os.getenv(
        "CHUNK_FILE",
        "src/extraction/outputs/sample_chunks.json",
    )
    documents = load_chunk_file(chunk_file)

    if not documents:
        raise ValueError(f"No documents parsed from {chunk_file}")

    document_ids = vector_store.add_documents(documents=documents)

    print(f"Stored {len(document_ids)} chunks from {chunk_file}")
