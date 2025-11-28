from pymilvus import Collection, MilvusException, connections, db, utility
from typing import Union
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_core.documents import Document
import json
import os
from pathlib import Path
from .modules.sparse_embedder import BGEEmbedder, SpladeEmbedder

conn = connections.connect(host="127.0.0.1", port=19530)

# vector_store = Milvus(
#     embedding_function=OllamaEmbeddings(model="qwen3-embedding:8b"),
#     collection_name="bm25",
#     builtin_function=BM25BuiltInFunction(output_field_names="sparse"),
#     connection_args={"uri": URI, "db_name": "milvus_demo"},
#     vector_field=["dense", "sparse"],
#     drop_old=False,
#     auto_id=True,
# )

# vector_store = Milvus(
#     embedding_function=[OllamaEmbeddings(model="qwen3-embedding:8b"), BGEEmbedder()],
#     collection_name="bge",
#     connection_args={"uri": URI, "db_name": "milvus_demo"},
#     vector_field=["dense", "sparse"],
#     drop_old=False,
#     auto_id=True,
# )

# vector_store = Milvus(
#     embedding_function=[OllamaEmbeddings(model="qwen3-embedding:8b"), SpladeEmbedder()],
#     collection_name="splade",
#     connection_args={"uri": URI, "db_name": "milvus_demo"},
#     vector_field=["dense", "sparse"],
#     drop_old=False,
#     auto_id=True,
# )

class MilvusVectorStore():
    def __init__(self, collection_name: str, embedding_function, buildin_function: Union[BM25BuiltInFunction, None]):
        self.vector_store = Milvus(
            embedding_function=embedding_function,
            builtin_function=buildin_function,
            collection_name=collection_name,
            connection_args={"uri": "http://localhost:19530", "db_name": "milvus_demo"},
            vector_field=["dense", "sparse"],
            drop_old=False,
            auto_id=True,
        )

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
    
    def add_documents(self, folder_path: str):
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                chunk_file = os.path.join(folder_path, filename)
                documents = load_chunk_file(chunk_file)

                if not documents:
                    raise ValueError(f"No documents parsed from {chunk_file}")

                self.vector_store.add_documents(documents=documents)
                print(f"Stored {len(documents)} chunks from {chunk_file}")

    def get_vector_store(self):
        return self.vector_store


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

def create_milvus_store(config) -> MilvusVectorStore:
    """Factory to create MilvusVectorStore from config."""
    
    # Assuming config has these fields or we use defaults
    embedding_model = getattr(config, "dense_embedding_model", "qwen3-embedding:8b")
    sparse_model = getattr(config, "sparse_embedding_model", "splade")
    
    dense_emb = OllamaEmbeddings(model=embedding_model)

    buildin_function = None
    sparse_emb = None
    
    if sparse_model == "splade":
        sparse_emb = SpladeEmbedder()
        embedding_function = [dense_emb, sparse_emb]
    elif sparse_model == "bge":
        sparse_emb = BGEEmbedder()
        embedding_function = [dense_emb, sparse_emb]
    else:  # bm25
        buildin_function = BM25BuiltInFunction(output_field_names="sparse")
        embedding_function = dense_emb  # Only dense embedding, BM25 handles sparse
    
    return MilvusVectorStore(
        collection_name=sparse_model,
        embedding_function=embedding_function,
        buildin_function=buildin_function
    )