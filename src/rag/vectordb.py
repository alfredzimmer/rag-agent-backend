import json
import os
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv

from qdrant_client import QdrantClient, models

from modules.embedder import compute_dense_vec, compute_sparse_vec

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

DENSE_VECTOR_NAME = "text-dense"
SPARSE_VECTOR_NAME = "text-sparse"
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))
collection_name = os.getenv("QDRANT_COLLECTION", "test")

# Ensure collection exists
if not qdrant_client.collection_exists(collection_name):
    print(f"Creating collection {collection_name} with size={VECTOR_SIZE}")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )
else:
    print(f"Collection {collection_name} already exists")


def load_chunk_file(path: str) -> list[dict]:
    """Convert a JSON chunk file into dicts with content + metadata."""
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
        documents.append({"page_content": chunk["content"], "metadata": metadata})

    return documents

def add_documents(documents: list[dict]) -> None:
    points: list[models.PointStruct] = []
    for doc in documents:
        content = doc["page_content"]
        metadata = doc["metadata"]

        dense_vector = compute_dense_vec(content)
        indices, values = compute_sparse_vec(content)

        point = models.PointStruct(
            id=uuid4().hex,
            vector={
                DENSE_VECTOR_NAME: dense_vector,
                SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=list(indices),
                    values=list(values),
                )
            },
            payload={
                "metadata": metadata,
                "page_content": content,
            },
        )
        points.append(point)

    if not points:
        return

    qdrant_client.upsert(collection_name=collection_name, points=points)



if __name__ == "__main__":
    chunk_file = os.getenv(
        "CHUNK_FILE",
        "outputs/30pg_outputs.json",
    )
    documents = load_chunk_file(chunk_file)

    if not documents:
        raise ValueError(f"No documents parsed from {chunk_file}")

    add_documents(documents)
    print(f"Stored {len(documents)} chunks from {chunk_file}")
