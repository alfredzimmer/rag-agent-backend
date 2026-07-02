from __future__ import annotations

import hashlib
import logging
import os
import signal
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

from opentelemetry import context, metrics, trace
from opentelemetry.trace import Status, StatusCode

from rag_agent_server.ingestion.chunker import CHUNKER_VERSION, chunk_sections
from rag_agent_server.ingestion.config import IngestionConfig
from rag_agent_server.ingestion.models import IngestionJob, IngestionStatus
from rag_agent_server.ingestion.parser import parse_document
from rag_agent_server.ingestion.queue import IngestionQueue
from rag_agent_server.observability import configure_telemetry
from rag.config import RAGConfig
from rag.milvus import create_milvus_store

logger = logging.getLogger(__name__)
shutdown_event = Event()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class IngestionWorker:
    def __init__(self, config: IngestionConfig | None = None) -> None:
        self.config = config or IngestionConfig.from_env()
        self.queue = IngestionQueue(self.config)
        self.consumer_name = os.getenv(
            "INGESTION_CONSUMER_NAME",
            f"{socket.gethostname()}-{os.getpid()}",
        )
        rag_config = RAGConfig(collection_name=self.config.collection_name)
        self.vector_store = create_milvus_store(rag_config)
        self.embedding_signature = (
            f"{rag_config.dense_embedding_model}:{rag_config.sparse_embedding_model}"
        )

        self.tracer = trace.get_tracer(__name__)
        meter = metrics.get_meter(__name__)
        self.jobs_counter = meter.create_counter(
            "rag_agent.ingestion.jobs",
            description="Ingestion jobs by terminal or retry status",
        )
        self.chunks_counter = meter.create_counter(
            "rag_agent.ingestion.chunks",
            description="Document chunks written to Milvus",
        )
        self.duration_histogram = meter.create_histogram(
            "rag_agent.ingestion.duration",
            unit="s",
            description="End-to-end ingestion processing duration",
        )

    def run(self) -> None:
        self.config.upload_dir.mkdir(parents=True, exist_ok=True)
        self.queue.ensure_consumer_group()
        logger.info(
            "Ingestion worker started",
            extra={"consumer_name": self.consumer_name},
        )

        while not shutdown_event.is_set():
            messages = self.queue.claim_stale(self.consumer_name)
            if not messages:
                messages = self.queue.read(self.consumer_name)
            for message_id, fields in messages:
                if shutdown_event.is_set():
                    break
                try:
                    self._handle_message(message_id, fields)
                except Exception:
                    logger.exception(
                        "Unexpected worker error; message remains pending",
                        extra={"stream_message_id": message_id},
                    )

        logger.info("Ingestion worker stopped")

    def _handle_message(self, message_id: str, fields: dict[str, str]) -> None:
        try:
            job = self.queue.decode_job(fields)
        except Exception as error:
            logger.exception(
                "Malformed ingestion message moved to dead-letter stream",
                extra={"stream_message_id": message_id},
            )
            self.queue.add_malformed_dead_letter(
                message_id,
                fields,
                f"{type(error).__name__}: {error}",
            )
            self.jobs_counter.add(1, {"status": "malformed"})
            return
        parent_context = self.queue.trace_context(fields)
        token = context.attach(parent_context)
        started = time.monotonic()
        try:
            with self.tracer.start_as_current_span("ingestion.process_job") as span:
                span.set_attributes(
                    {
                        "rag_agent.ingestion.job_id": str(job.job_id),
                        "rag_agent.ingestion.attempt": job.attempt,
                        "rag_agent.ingestion.collection": job.collection_name,
                        "rag_agent.ingestion.scope_id": str(job.scope_id),
                    }
                )
                try:
                    self._process(message_id, job, span)
                except Exception as error:
                    span.record_exception(error)
                    span.set_status(Status(StatusCode.ERROR, str(error)))
                    self._handle_failure(message_id, job, error)
        finally:
            self.duration_histogram.record(
                time.monotonic() - started,
                {"collection": job.collection_name},
            )
            context.detach(token)

    def _process(self, message_id: str, job: IngestionJob, span: trace.Span) -> None:
        source_path = job.source_path.resolve()
        if not source_path.is_relative_to(self.config.upload_dir):
            raise ValueError("Job source_path is outside INGESTION_UPLOAD_DIR")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if _sha256(source_path) != job.source_sha256:
            raise ValueError("Document checksum changed after upload")

        fingerprint = self.queue.fingerprint(
            job,
            self.embedding_signature,
            (
                f"{CHUNKER_VERSION}:"
                f"{self.config.chunk_size}:"
                f"{self.config.chunk_overlap}"
            ),
        )
        prior_job_id = self.queue.completed_job(fingerprint)
        if prior_job_id:
            self.queue.update_state(
                job,
                IngestionStatus.DUPLICATE,
                document_id=job.source_sha256,
                error=None,
                completed_at=datetime.now(timezone.utc),
            )
            self.queue.acknowledge(message_id)
            self.jobs_counter.add(1, {"status": "duplicate"})
            logger.info(
                "Skipped duplicate ingestion job",
                extra={"job_id": job.job_id, "document_id": job.source_sha256},
            )
            return

        lock_token = self.queue.acquire_document_lock(
            fingerprint,
            self.consumer_name,
        )
        if lock_token is None:
            logger.info(
                "Document is already being ingested",
                extra={"job_id": job.job_id, "document_id": job.source_sha256},
            )
            return

        try:
            self.queue.update_state(job, IngestionStatus.PROCESSING, error=None)
            sections = parse_document(source_path)
            if not sections:
                raise ValueError("Document contains no extractable text")
            documents = chunk_sections(
                sections,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                original_filename=job.original_filename,
                document_id=job.source_sha256,
                job_id=job.job_id,
                scope_id=job.scope_id,
            )
            if not documents:
                raise ValueError("Document produced no chunks")

            span.set_attribute("rag_agent.ingestion.chunk_count", len(documents))
            self.vector_store.replace_documents(
                documents,
                document_id=job.source_sha256,
                scope_id=str(job.scope_id),
            )
            self.queue.mark_document_completed(fingerprint, str(job.job_id))
            self.queue.update_state(
                job,
                IngestionStatus.COMPLETED,
                chunks_written=len(documents),
                document_id=job.source_sha256,
                error=None,
                completed_at=datetime.now(timezone.utc),
            )
            self.queue.acknowledge(message_id)
            if self.config.delete_source_on_success:
                source_path.unlink(missing_ok=True)

            self.jobs_counter.add(1, {"status": "completed"})
            self.chunks_counter.add(
                len(documents),
                {"collection": job.collection_name},
            )
            logger.info(
                "Document ingestion completed",
                extra={
                    "job_id": job.job_id,
                    "user_id": job.user_id,
                    "conversation_id": job.scope_id,
                    "document_id": job.source_sha256,
                },
            )
        finally:
            self.queue.release_document_lock(fingerprint, lock_token)

    def _handle_failure(
        self,
        message_id: str,
        job: IngestionJob,
        error: Exception,
    ) -> None:
        error_message = f"{type(error).__name__}: {error}"
        logger.exception(
            "Document ingestion failed",
            extra={"job_id": job.job_id, "document_id": job.source_sha256},
        )

        if job.attempt < self.config.max_attempts:
            self.queue.retry(message_id, job, error_message)
            self.jobs_counter.add(1, {"status": "retrying"})
            return

        self.queue.add_dead_letter(job, error_message)
        self.queue.update_state(
            job,
            IngestionStatus.FAILED,
            error=error_message,
            completed_at=datetime.now(timezone.utc),
        )
        self.queue.acknowledge(message_id)
        self.jobs_counter.add(1, {"status": "failed"})


def _request_shutdown(*_: object) -> None:
    shutdown_event.set()


def main() -> None:
    configure_telemetry("rag-agent-ingestion-worker")
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)
    IngestionWorker().run()


if __name__ == "__main__":
    main()
