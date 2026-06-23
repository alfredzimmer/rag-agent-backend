from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import redis
from opentelemetry import propagate
from redis.exceptions import ResponseError, TimeoutError as RedisTimeoutError

from .config import IngestionConfig
from .models import IngestionJob, IngestionJobState, IngestionStatus


class IngestionQueue:
    def __init__(
        self,
        config: IngestionConfig | None = None,
        client: redis.Redis | None = None,
    ) -> None:
        self.config = config or IngestionConfig.from_env()
        self.client = client or redis.Redis.from_url(
            self.config.redis_url,
            decode_responses=True,
        )

    def ensure_consumer_group(self) -> None:
        try:
            self.client.xgroup_create(
                self.config.stream_name,
                self.config.consumer_group,
                id="0",
                mkstream=True,
            )
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

    def enqueue(self, job: IngestionJob) -> str:
        carrier: dict[str, str] = {}
        propagate.inject(carrier)
        fields = {
            "payload": job.model_dump_json(),
            **carrier,
        }
        state = IngestionJobState.from_job(job)
        pipeline = self.client.pipeline(transaction=True)
        pipeline.xadd(self.config.stream_name, fields)
        pipeline.setex(
            self._state_key(str(job.job_id)),
            self.config.job_ttl_seconds,
            state.model_dump_json(),
        )
        message_id, _ = pipeline.execute()
        return str(message_id)

    def get_state(self, job_id: str) -> IngestionJobState | None:
        raw = self.client.get(self._state_key(job_id))
        return IngestionJobState.model_validate_json(raw) if raw else None

    def set_state(self, state: IngestionJobState) -> None:
        state.updated_at = datetime.now(timezone.utc)
        self.client.setex(
            self._state_key(str(state.job_id)),
            self.config.job_ttl_seconds,
            state.model_dump_json(),
        )

    def update_state(
        self,
        job: IngestionJob,
        status: IngestionStatus,
        **updates: Any,
    ) -> IngestionJobState:
        state = self.get_state(str(job.job_id)) or IngestionJobState.from_job(job)
        state.status = status
        state.attempt = job.attempt
        for key, value in updates.items():
            setattr(state, key, value)
        self.set_state(state)
        return state

    def add_dead_letter(self, job: IngestionJob, error: str) -> str:
        return str(
            self.client.xadd(
                self.config.dead_letter_stream,
                {
                    "payload": job.model_dump_json(),
                    "error": error,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                },
                maxlen=self.config.stream_max_length,
                approximate=True,
            )
        )

    def add_malformed_dead_letter(
        self,
        message_id: str,
        fields: dict[str, str],
        error: str,
    ) -> None:
        pipeline = self.client.pipeline(transaction=True)
        pipeline.xadd(
            self.config.dead_letter_stream,
            {
                "source_message_id": message_id,
                "raw_fields": json.dumps(fields, ensure_ascii=True),
                "error": error,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
            maxlen=self.config.stream_max_length,
            approximate=True,
        )
        pipeline.xack(
            self.config.stream_name,
            self.config.consumer_group,
            message_id,
        )
        pipeline.xdel(self.config.stream_name, message_id)
        pipeline.execute()

    def read(self, consumer_name: str, block_ms: int = 5000) -> list[tuple[str, dict[str, str]]]:
        try:
            entries = self.client.xreadgroup(
                self.config.consumer_group,
                consumer_name,
                {self.config.stream_name: ">"},
                count=1,
                block=block_ms,
            )
        except RedisTimeoutError:
            return []
        return self._flatten(entries)

    def claim_stale(self, consumer_name: str) -> list[tuple[str, dict[str, str]]]:
        result = self.client.xautoclaim(
            self.config.stream_name,
            self.config.consumer_group,
            consumer_name,
            min_idle_time=self.config.stale_after_ms,
            start_id="0-0",
            count=1,
        )
        messages = result[1] if len(result) > 1 else []
        return [(str(message_id), fields) for message_id, fields in messages]

    def acknowledge(self, message_id: str) -> None:
        pipeline = self.client.pipeline(transaction=True)
        pipeline.xack(
            self.config.stream_name,
            self.config.consumer_group,
            message_id,
        )
        pipeline.xdel(self.config.stream_name, message_id)
        pipeline.execute()

    def retry(
        self,
        message_id: str,
        job: IngestionJob,
        error: str,
    ) -> IngestionJob:
        retry_job = job.model_copy(update={"attempt": job.attempt + 1})
        carrier: dict[str, str] = {}
        propagate.inject(carrier)
        retry_state = IngestionJobState.from_job(
            retry_job,
            IngestionStatus.RETRYING,
        )
        retry_state.error = error

        pipeline = self.client.pipeline(transaction=True)
        pipeline.xadd(
            self.config.stream_name,
            {
                "payload": retry_job.model_dump_json(),
                **carrier,
            },
        )
        pipeline.setex(
            self._state_key(str(retry_job.job_id)),
            self.config.job_ttl_seconds,
            retry_state.model_dump_json(),
        )
        pipeline.xack(
            self.config.stream_name,
            self.config.consumer_group,
            message_id,
        )
        pipeline.xdel(self.config.stream_name, message_id)
        pipeline.execute()
        return retry_job

    def acquire_document_lock(
        self,
        fingerprint: str,
        consumer_name: str,
    ) -> str | None:
        lock_token = f"{consumer_name}:{uuid4()}"
        acquired = self.client.set(
            self._lock_key(fingerprint),
            lock_token,
            nx=True,
            px=self.config.lock_ttl_ms,
        )
        return lock_token if acquired else None

    def release_document_lock(self, fingerprint: str, lock_token: str) -> None:
        self.client.eval(
            """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            end
            return 0
            """,
            1,
            self._lock_key(fingerprint),
            lock_token,
        )

    def completed_job(self, fingerprint: str) -> str | None:
        return self.client.get(self._completed_key(fingerprint))

    def mark_document_completed(self, fingerprint: str, job_id: str) -> None:
        self.client.setex(
            self._completed_key(fingerprint),
            self.config.job_ttl_seconds,
            job_id,
        )

    @staticmethod
    def decode_job(fields: dict[str, str]) -> IngestionJob:
        return IngestionJob.model_validate_json(fields["payload"])

    @staticmethod
    def trace_context(fields: dict[str, str]) -> Any:
        carrier = {
            key: value
            for key, value in fields.items()
            if key in {"traceparent", "tracestate", "baggage"}
        }
        return propagate.extract(carrier)

    @staticmethod
    def _flatten(entries: list[Any]) -> list[tuple[str, dict[str, str]]]:
        flattened: list[tuple[str, dict[str, str]]] = []
        for _, messages in entries:
            for message_id, fields in messages:
                flattened.append((str(message_id), fields))
        return flattened

    @staticmethod
    def fingerprint(
        job: IngestionJob,
        embedding_signature: str,
        chunker_signature: str,
    ) -> str:
        value = "|".join(
            (
                job.source_sha256,
                job.collection_name,
                str(job.scope_id),
                embedding_signature,
                chunker_signature,
            )
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _state_key(job_id: str) -> str:
        return f"edemi:ingestion:job:{job_id}"

    @staticmethod
    def _lock_key(fingerprint: str) -> str:
        return f"edemi:ingestion:lock:{fingerprint}"

    @staticmethod
    def _completed_key(fingerprint: str) -> str:
        return f"edemi:ingestion:completed:{fingerprint}"
