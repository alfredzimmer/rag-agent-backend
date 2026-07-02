from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IngestionConfig:
    redis_url: str
    stream_name: str
    consumer_group: str
    dead_letter_stream: str
    upload_dir: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    max_attempts: int
    stale_after_ms: int
    lock_ttl_ms: int
    job_ttl_seconds: int
    stream_max_length: int
    delete_source_on_success: bool

    @classmethod
    def from_env(cls) -> "IngestionConfig":
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            host = os.getenv("REDIS_HOST", "localhost")
            port = os.getenv("REDIS_PORT", "6380")
            redis_url = f"redis://{host}:{port}/0"

        return cls(
            redis_url=redis_url,
            stream_name=os.getenv("INGESTION_STREAM", "rag-agent:ingestion:jobs"),
            consumer_group=os.getenv("INGESTION_CONSUMER_GROUP", "rag-agent-ingestion-workers"),
            dead_letter_stream=os.getenv(
                "INGESTION_DEAD_LETTER_STREAM",
                "rag-agent:ingestion:dead-letter",
            ),
            upload_dir=Path(os.getenv("INGESTION_UPLOAD_DIR", ".runtime/uploads")).resolve(),
            collection_name=os.getenv("RAG_COLLECTION_NAME", "HeaderInContentTrial"),
            chunk_size=int(os.getenv("INGESTION_CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("INGESTION_CHUNK_OVERLAP", "150")),
            max_attempts=int(os.getenv("INGESTION_MAX_ATTEMPTS", "3")),
            stale_after_ms=int(os.getenv("INGESTION_STALE_AFTER_MS", "900000")),
            lock_ttl_ms=int(os.getenv("INGESTION_LOCK_TTL_MS", "7200000")),
            job_ttl_seconds=int(os.getenv("INGESTION_JOB_TTL_SECONDS", "604800")),
            stream_max_length=int(os.getenv("INGESTION_STREAM_MAX_LENGTH", "10000")),
            delete_source_on_success=os.getenv(
                "INGESTION_DELETE_SOURCE_ON_SUCCESS",
                "false",
            ).lower()
            == "true",
        )
