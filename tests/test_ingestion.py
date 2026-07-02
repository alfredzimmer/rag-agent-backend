from __future__ import annotations

import json
import hashlib
import logging
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from opentelemetry import context, metrics, trace
from langchain_core.documents import Document
from redis.exceptions import TimeoutError as RedisTimeoutError

from rag_agent_server.ingestion.chunker import CHUNKER_VERSION, chunk_sections
from rag_agent_server.ingestion.config import IngestionConfig
from rag_agent_server.ingestion.models import IngestionJob, IngestionStatus
from rag_agent_server.ingestion.parser import parse_document
from rag_agent_server.ingestion.queue import IngestionQueue
from rag_agent_server.ingestion.worker import IngestionWorker
from rag_agent_server.observability import JsonFormatter
from rag.milvus import MilvusVectorStore


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.messages: list[dict[str, str]] = []
        self.acknowledged: list[str] = []
        self.deleted: list[str] = []

    def xadd(self, _stream: str, fields: dict[str, str], **_: object) -> str:
        self.messages.append(fields)
        return f"{len(self.messages)}-0"

    def setex(self, key: str, _ttl: int, value: str) -> None:
        self.values[key] = value

    def xack(self, _stream: str, _group: str, message_id: str) -> None:
        self.acknowledged.append(message_id)

    def xdel(self, _stream: str, message_id: str) -> None:
        self.deleted.append(message_id)

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def pipeline(self, transaction: bool = True):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, client: FakeRedis) -> None:
        self.client = client
        self.commands = []

    def xadd(self, stream: str, fields: dict[str, str], **kwargs: object):
        self.commands.append(("xadd", stream, fields, kwargs))
        return self

    def setex(self, key: str, ttl: int, value: str):
        self.commands.append(("setex", key, ttl, value))
        return self

    def xack(self, stream: str, group: str, message_id: str):
        self.commands.append(("xack", stream, group, message_id))
        return self

    def xdel(self, stream: str, message_id: str):
        self.commands.append(("xdel", stream, message_id))
        return self

    def execute(self) -> list[object]:
        results = []
        for command in self.commands:
            if command[0] == "xadd":
                _, stream, fields, kwargs = command
                results.append(self.client.xadd(stream, fields, **kwargs))
            elif command[0] == "setex":
                _, key, ttl, value = command
                self.client.setex(key, ttl, value)
                results.append(None)
            elif command[0] == "xack":
                _, stream, group, message_id = command
                self.client.xack(stream, group, message_id)
                results.append(None)
            else:
                _, stream, message_id = command
                self.client.xdel(stream, message_id)
                results.append(None)
        return results


class FakeBlockingRedis(FakeRedis):
    def xreadgroup(self, *_: object, **__: object):
        raise RedisTimeoutError("idle blocking read")


class FakeWorkerQueue:
    def __init__(self, job: IngestionJob) -> None:
        self.job = job
        self.statuses: list[IngestionStatus] = []
        self.acknowledged = False

    def decode_job(self, _fields: dict[str, str]) -> IngestionJob:
        return self.job

    def trace_context(self, _fields: dict[str, str]):
        return context.get_current()

    def fingerprint(self, *_: object) -> str:
        return "fingerprint"

    def completed_job(self, _fingerprint: str) -> None:
        return None

    def acquire_document_lock(self, *_: object) -> str:
        return "lock-token"

    def release_document_lock(
        self,
        _fingerprint: str,
        _lock_token: str,
    ) -> None:
        return None

    def update_state(
        self,
        _job: IngestionJob,
        status: IngestionStatus,
        **_: object,
    ) -> None:
        self.statuses.append(status)

    def mark_document_completed(self, *_: object) -> None:
        return None

    def acknowledge(self, _message_id: str) -> None:
        self.acknowledged = True


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents = []

    def replace_documents(self, documents, **_: object) -> None:
        self.documents = documents


class FakeCollection:
    def __init__(self) -> None:
        self.deleted_expression = None
        self.flush_count = 0

    def delete(self, expression: str) -> None:
        self.deleted_expression = expression

    def flush(self) -> None:
        self.flush_count += 1


class FakeLangChainMilvus:
    def __init__(self) -> None:
        self.col = FakeCollection()
        self.documents = []

    def add_documents(self, documents) -> None:
        self.documents = documents


def make_config(upload_dir: Path) -> IngestionConfig:
    return IngestionConfig(
        redis_url="redis://localhost:6380/0",
        stream_name="test:jobs",
        consumer_group="test-workers",
        dead_letter_stream="test:dead-letter",
        upload_dir=upload_dir,
        collection_name="test-collection",
        chunk_size=20,
        chunk_overlap=5,
        max_attempts=3,
        stale_after_ms=1000,
        lock_ttl_ms=60000,
        job_ttl_seconds=60,
        stream_max_length=100,
        delete_source_on_success=False,
    )


class IngestionTests(unittest.TestCase):
    def test_text_parser_and_chunk_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("First paragraph.\n\nSecond paragraph with more text.")
            sections = parse_document(path)
            job_id = uuid4()
            scope_id = uuid4()

            chunks = chunk_sections(
                sections,
                chunk_size=20,
                chunk_overlap=5,
                original_filename=path.name,
                document_id="a" * 64,
                job_id=job_id,
                scope_id=scope_id,
            )

            self.assertGreater(len(chunks), 1)
            self.assertEqual(chunks[0].metadata["document_id"], "a" * 64)
            self.assertEqual(chunks[0].metadata["scope_id"], str(scope_id))
            self.assertEqual(chunks[0].metadata["chunker_version"], CHUNKER_VERSION)

    def test_queue_persists_initial_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            upload_dir = Path(directory).resolve()
            source_path = upload_dir / "sample.txt"
            source_path.write_text("hello")
            job = IngestionJob(
                job_id=uuid4(),
                user_id=uuid4(),
                scope_id=uuid4(),
                source_path=source_path,
                original_filename="sample.txt",
                source_sha256="a" * 64,
                collection_name="test-collection",
            )
            fake_redis = FakeRedis()
            queue = IngestionQueue(make_config(upload_dir), client=fake_redis)

            message_id = queue.enqueue(job)
            state = queue.get_state(str(job.job_id))

            self.assertEqual(message_id, "1-0")
            self.assertIsNotNone(state)
            self.assertEqual(state.status, IngestionStatus.QUEUED)
            self.assertEqual(
                IngestionQueue.decode_job(fake_redis.messages[0]),
                job,
            )

    def test_retry_is_requeued_and_old_message_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            upload_dir = Path(directory).resolve()
            source_path = upload_dir / "sample.txt"
            source_path.write_text("hello")
            job = IngestionJob(
                job_id=uuid4(),
                user_id=uuid4(),
                scope_id=uuid4(),
                source_path=source_path,
                original_filename="sample.txt",
                source_sha256="b" * 64,
                collection_name="test-collection",
            )
            fake_redis = FakeRedis()
            queue = IngestionQueue(make_config(upload_dir), client=fake_redis)
            queue.enqueue(job)

            retry_job = queue.retry("1-0", job, "temporary failure")
            state = queue.get_state(str(job.job_id))

            self.assertEqual(retry_job.attempt, 2)
            self.assertEqual(state.status, IngestionStatus.RETRYING)
            self.assertEqual(state.error, "temporary failure")
            self.assertIn("1-0", fake_redis.acknowledged)
            self.assertIn("1-0", fake_redis.deleted)

    def test_idle_queue_timeout_returns_no_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = IngestionQueue(
                make_config(Path(directory).resolve()),
                client=FakeBlockingRedis(),
            )

            self.assertEqual(queue.read("test-worker"), [])

    def test_json_logs_include_structured_context(self) -> None:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="document processed",
            args=(),
            exc_info=None,
        )
        record.job_id = "job-123"
        payload = json.loads(JsonFormatter().format(record))

        self.assertEqual(payload["message"], "document processed")
        self.assertEqual(payload["job_id"], "job-123")

    def test_worker_processes_document_to_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            upload_dir = Path(directory).resolve()
            source_path = upload_dir / "sample.txt"
            content = b"An ingested document with enough text to create chunks."
            source_path.write_bytes(content)
            job = IngestionJob(
                job_id=uuid4(),
                user_id=uuid4(),
                scope_id=uuid4(),
                source_path=source_path,
                original_filename="sample.txt",
                source_sha256=hashlib.sha256(content).hexdigest(),
                collection_name="test-collection",
            )
            queue = FakeWorkerQueue(job)
            vector_store = FakeVectorStore()
            worker = IngestionWorker.__new__(IngestionWorker)
            worker.config = make_config(upload_dir)
            worker.queue = queue
            worker.consumer_name = "test-worker"
            worker.vector_store = vector_store
            worker.embedding_signature = "test-embedding:none"
            worker.tracer = trace.get_tracer("test")
            meter = metrics.get_meter("test")
            worker.jobs_counter = meter.create_counter("test.jobs")
            worker.chunks_counter = meter.create_counter("test.chunks")
            worker.duration_histogram = meter.create_histogram("test.duration")

            worker._handle_message("1-0", {"payload": job.model_dump_json()})

            self.assertTrue(queue.acknowledged)
            self.assertIn(IngestionStatus.PROCESSING, queue.statuses)
            self.assertIn(IngestionStatus.COMPLETED, queue.statuses)
            self.assertGreater(len(vector_store.documents), 0)

    def test_milvus_replacement_deletes_matching_scope_first(self) -> None:
        langchain_store = FakeLangChainMilvus()
        store = MilvusVectorStore.__new__(MilvusVectorStore)
        store.vector_store = langchain_store
        documents = [Document(page_content="content")]

        store.replace_documents(
            documents,
            document_id="a" * 64,
            scope_id="scope-id",
        )

        self.assertEqual(
            langchain_store.col.deleted_expression,
            f'document_id == "{"a" * 64}" and scope_id == "scope-id"',
        )
        self.assertEqual(langchain_store.documents, documents)
        self.assertEqual(langchain_store.col.flush_count, 2)


if __name__ == "__main__":
    unittest.main()
