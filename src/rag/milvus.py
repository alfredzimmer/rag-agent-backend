from pymilvus import Collection, MilvusException, connections, db, utility
from typing import Union, Optional
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_core.documents import Document
import json
import os
from pathlib import Path
from .modules.sparse_embedder import BGEEmbedder, SpladeEmbedder
from .config import RAGConfig


def milvus_uri() -> str:
    return os.getenv("MILVUS_URI", "http://localhost:19530")


def milvus_db_name() -> str:
    return os.getenv("MILVUS_DB", "rag1")


def connect_milvus():
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    return connections.connect(host=host, port=port)

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

class SafeMilvus(Milvus):
    def _init(self, *args, **kwargs):
        alias = self.alias
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        db_name = os.getenv("MILVUS_DB", "rag1")
        try:
            if not connections.has_connection(alias):
                connections.connect(alias=alias, host=host, port=port, db_name=db_name)
                print(f"SafeMilvus: Successfully registered connection alias '{alias}' in ORM.")
        except Exception as e:
            print(f"SafeMilvus error: {e}")
        super()._init(*args, **kwargs)


class MilvusVectorStore():
    def __init__(self, collection_name: str, embedding_function, buildin_function: Union[BM25BuiltInFunction, None], similarity_threshold: float = 0.0):
        self.similarity_threshold = similarity_threshold
        self.is_hybrid = True if buildin_function or isinstance(embedding_function, list) else False
        self.vector_store = SafeMilvus(
            embedding_function=embedding_function,
            builtin_function=buildin_function,
            collection_name=collection_name,
            connection_args={"uri": milvus_uri(), "db_name": milvus_db_name()},
            vector_field="vector" if collection_name == "ingestion_staging" else (["dense", "sparse"] if buildin_function or isinstance(embedding_function, list) else "dense"),
            text_field="content" if collection_name == "ingestion_staging" else "text",
            drop_old=False,  # Don't drop existing collection
            auto_id=True,
            enable_dynamic_field=True,  # Enable dynamic fields to store all metadata
        )

    def load_chunk_file(self, path: str) -> list[Document]:
        """Convert a JSON chunk file into dicts with content + metadata."""
        chunk_path = Path(path)
        with chunk_path.open("r", encoding="utf-8") as fh:
            raw_chunks = json.load(fh)

        documents = []
        for chunk in raw_chunks:
            metadata = {
                "char_count": chunk["char_count"],
                **chunk.get("metadata", {})  # Merge metadata if present
            }
            to_append = ""
            if "to_append" in chunk and chunk["to_append"]:
                for key, value in chunk["to_append"].items():
                    to_append += f"{key}: {value}, "
            page_content = to_append + chunk["content"]
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def add_documents(self, folder_path: str):
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                chunk_file = os.path.join(folder_path, filename)
                documents = self.load_chunk_file(chunk_file)

                if not documents:
                    raise ValueError(f"No documents parsed from {chunk_file}")

                self.vector_store.add_documents(documents=documents)
                print(f"Stored {len(documents)} chunks from {chunk_file}")

    def add_document(self, chunk_file: str):
        documents = self.load_chunk_file(chunk_file)

        if not documents:
            raise ValueError(f"No documents parsed from {chunk_file}")

        self.vector_store.add_documents(documents=documents)
        print(f"Stored {len(documents)} chunks from {chunk_file}")

    def get_vector_store(self):
        return self.vector_store

    def search_documents(self, query: str, k: int = 30, expr: Optional[str] = None) -> list[Document]:
        """
        Perform hybrid or semantic search based on configuration, filtering by similarity_threshold.
        """
        if self.similarity_threshold > 0.0:
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k, expr=expr)

            is_l2 = False
            if not self.is_hybrid:
                try:
                    col = self.vector_store.col
                    if col:
                        for idx in col.indexes:
                            if idx.to_dict().get("index_param", {}).get("metric_type") == "L2":
                                is_l2 = True
                                break
                except Exception:
                    pass

            filtered_docs = []
            for doc, score in docs_and_scores:
                doc.metadata["similarity_score"] = float(score)
                if is_l2:
                    similarity = 1.0 - (float(score) / 2.0)
                    if similarity >= self.similarity_threshold:
                        filtered_docs.append(doc)
                else:
                    if float(score) >= self.similarity_threshold:
                        filtered_docs.append(doc)
            return filtered_docs
        else:
            return self.vector_store.similarity_search(query, k=k, expr=expr)

    def similarity_search(self, query: str, k: int = 30, expr: Optional[str] = None) -> list[Document]:
        """
        Perform standard vector similarity search, filtering by similarity_threshold.
        """
        if self.similarity_threshold > 0.0:
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k, expr=expr)

            is_l2 = False
            if not self.is_hybrid:
                try:
                    col = self.vector_store.col
                    if col:
                        for idx in col.indexes:
                            if idx.to_dict().get("index_param", {}).get("metric_type") == "L2":
                                is_l2 = True
                                break
                except Exception:
                    pass

            filtered_docs = []
            for doc, score in docs_and_scores:
                doc.metadata["similarity_score"] = float(score)
                if is_l2:
                    similarity = 1.0 - (float(score) / 2.0)
                    if similarity >= self.similarity_threshold:
                        filtered_docs.append(doc)
                else:
                    if float(score) >= self.similarity_threshold:
                        filtered_docs.append(doc)
            return filtered_docs
        else:
            return self.vector_store.similarity_search(query, k=k, expr=expr)



def ensure_db_exists(db_name):
    connect_milvus()
    try:
        existing_databases = db.list_database()
        if db_name not in existing_databases:
            db.create_database(db_name)
            print(f"Database '{db_name}' created successfully.")
        else:
            print(f"Database '{db_name}' already exists.")
    except MilvusException as e:
        print(f"Database setup check: {e}")


def create_db(db_name):
    connect_milvus()
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

    # Ensure the configured database exists safely
    ensure_db_exists(milvus_db_name())

    # Assuming config has these fields or we use defaults
    embedding_model = getattr(config, "dense_embedding_model", "qwen3-embedding:8b")
    sparse_model = getattr(config, "sparse_embedding_model", "splade")
    collection_name = getattr(config, "collection_name", "testing")

    dense_emb = OllamaEmbeddings(model=embedding_model)

    buildin_function = None
    sparse_emb = None

    if sparse_model == "splade":
        sparse_emb = SpladeEmbedder()
        embedding_function = [dense_emb, sparse_emb]
    elif sparse_model == "bge":
        sparse_emb = BGEEmbedder()
        embedding_function = [dense_emb, sparse_emb]
    elif sparse_model == "bm25":
        buildin_function = BM25BuiltInFunction(output_field_names="sparse")
        embedding_function = dense_emb
    else:
        embedding_function = dense_emb

    similarity_threshold = getattr(config, "similarity_threshold", 0.0)
    return MilvusVectorStore(
        collection_name=collection_name,
        embedding_function=embedding_function,
        buildin_function=buildin_function,
        similarity_threshold=similarity_threshold
    )

if __name__ == "__main__":
    vector_store = create_milvus_store(RAGConfig(collection_name="testing"))
    vector_store.add_document("src/rag/outputs/30pg_outputs.json")
