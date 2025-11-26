from pymilvus import Collection, MilvusException, connections, db, utility
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_core.documents import Document
import json
import os
from pathlib import Path

conn = connections.connect(host="127.0.0.1", port=19530)

def create_db(db_name):
    try:
        existing_databases = db.list_database()
        if db_name in existing_databases:
            print(f"Database '{db_name}' already exists.")

            # Use the database context
            db.using_database(db_name)

            # Drop all collections in the database
            collections = utility.list_collections()
            for collection_name in collections:
                collection = Collection(name=collection_name)
                collection.drop()
                print(f"Collection '{collection_name}' has been dropped.")

            db.drop_database(db_name)
            print(f"Database '{db_name}' has been deleted.")
        else:
            print(f"Database '{db_name}' does not exist.")
            database = db.create_database(db_name)
            print(f"Database '{db_name}' created successfully.")
    except MilvusException as e:
        print(f"An error occurred: {e}")


def get_vector_store():
    URI = "http://localhost:19530"

    vector_store = Milvus(
        embedding_function=OllamaEmbeddings(model="qwen3-embedding:8b"),
        builtin_function=BM25BuiltInFunction(output_field_names="sparse"),
        connection_args={"uri": URI, "db_name": "milvus_demo"},
        vector_field=["dense", "sparse"],
        drop_old=True,
        auto_id=True,
    )

    return vector_store

def load_chunk_file(path: str) -> list[Document]:
    """Convert a JSON chunk file into dicts with content + metadata."""
    chunk_path = Path(path)
    with chunk_path.open("r", encoding="utf-8") as fh:
        raw_chunks = json.load(fh)

    documents = []
    for chunk in raw_chunks:
        metadata = {
            "char_count": chunk["char_count"],
            **chunk.get("headers", {}),
        }
        documents.append(Document(page_content=chunk["content"], metadata=metadata))

    return documents

if __name__ == "__main__":
    vector_store = get_vector_store()
    for filename in os.listdir("src/rag/outputs/2025"):
        if filename.endswith(".json"):
            chunk_file = os.path.join("src/rag/outputs/2025", filename)
            documents = load_chunk_file(chunk_file)

            if not documents:
                raise ValueError(f"No documents parsed from {chunk_file}")

            vector_store.add_documents(documents=documents)
            print(f"Stored {len(documents)} chunks from {chunk_file}")