"""Milvus vector store access for the RAG agent."""
import logging
from urllib.parse import urlparse

from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from pymilvus import connections, db

from .config import RAGConfig

logger = logging.getLogger(__name__)


def ensure_db_exists(config: RAGConfig) -> None:
    """Create the configured Milvus database on a fresh server."""
    parsed = urlparse(config.milvus_uri)
    alias = "rag-agent-bootstrap"
    connections.connect(
        alias=alias,
        host=parsed.hostname or "localhost",
        port=str(parsed.port or 19530),
    )
    try:
        if config.milvus_db not in db.list_database(using=alias):
            db.create_database(config.milvus_db, using=alias)
            logger.info("Created Milvus database", extra={"database": config.milvus_db})
    finally:
        connections.disconnect(alias)


class ConnectedMilvus(Milvus):
    """langchain-milvus does not register a pymilvus connection when
    connection_args carry a db_name; register the alias before the store
    initializes or every collection lookup fails with ConnectionNotExist."""

    def __init__(self, *args, rag_config: RAGConfig, **kwargs):
        self._rag_config = rag_config
        super().__init__(*args, **kwargs)

    def _init(self, *args, **kwargs):
        if not connections.has_connection(self.alias):
            parsed = urlparse(self._rag_config.milvus_uri)
            connections.connect(
                alias=self.alias,
                host=parsed.hostname or "localhost",
                port=str(parsed.port or 19530),
                db_name=self._rag_config.milvus_db,
            )
        super()._init(*args, **kwargs)


def create_milvus_store(config: RAGConfig) -> Milvus:
    ensure_db_exists(config)
    embeddings = OllamaEmbeddings(
        model=config.embedding_model,
        base_url=config.ollama_host,
    )
    return ConnectedMilvus(
        rag_config=config,
        embedding_function=embeddings,
        collection_name=config.collection_name,
        connection_args={"uri": config.milvus_uri, "db_name": config.milvus_db},
        vector_field="dense",
        text_field="text",
        auto_id=True,
        enable_dynamic_field=True,
        drop_old=False,
    )
