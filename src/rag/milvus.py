import logging
import os
from typing import Union, Optional

from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_core.documents import Document
from pymilvus import MilvusException, connections, db

from .modules.sparse_embedder import BGEEmbedder, SpladeEmbedder
from .config import RAGConfig

logger = logging.getLogger(__name__)


def milvus_uri() -> str:
    return os.getenv("MILVUS_URI", "http://localhost:19530")


def milvus_db_name() -> str:
    return os.getenv("MILVUS_DB", "rag1")


def connect_milvus():
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    return connections.connect(host=host, port=port)

class SafeMilvus(Milvus):
    def _init(self, *args, **kwargs):
        alias = self.alias
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        db_name = os.getenv("MILVUS_DB", "rag1")
        try:
            if not connections.has_connection(alias):
                connections.connect(alias=alias, host=host, port=port, db_name=db_name)
                logger.info("Registered Milvus connection alias", extra={"alias": alias})
        except Exception:
            logger.exception("Failed to register Milvus connection alias")
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
            vector_field=["dense", "sparse"] if self.is_hybrid else "dense",
            text_field="text",
            drop_old=False,
            auto_id=True,
            enable_dynamic_field=True,
        )

    def replace_documents(
        self,
        documents: list[Document],
        *,
        document_id: str,
        scope_id: str,
    ) -> None:
        if not documents:
            raise ValueError("documents cannot be empty")
        collection = self.vector_store.col
        if collection is not None:
            expression = (
                f'document_id == "{document_id}" and scope_id == "{scope_id}"'
            )
            collection.delete(expression)
            collection.flush()
        self.vector_store.add_documents(documents=documents)
        if collection is not None:
            collection.flush()

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
        return self.vector_store.similarity_search(query, k=k, expr=expr)

def ensure_db_exists(db_name):
    connect_milvus()
    try:
        existing_databases = db.list_database()
        if db_name not in existing_databases:
            db.create_database(db_name)
            logger.info("Created Milvus database", extra={"database": db_name})
        else:
            logger.debug("Milvus database already exists", extra={"database": db_name})
    except MilvusException:
        logger.exception("Milvus database setup failed", extra={"database": db_name})
        raise

def create_milvus_store(config) -> MilvusVectorStore:
    """Factory to create MilvusVectorStore from config."""

    ensure_db_exists(milvus_db_name())

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
