from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IngestionStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    DUPLICATE = "duplicate"


class IngestionJob(BaseModel):
    schema_version: int = 1
    job_id: UUID
    user_id: UUID
    scope_id: UUID
    source_path: Path
    original_filename: str = Field(min_length=1, max_length=255)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    collection_name: str = Field(min_length=1, max_length=255)
    attempt: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("source_path")
    @classmethod
    def source_path_must_be_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("source_path must be absolute")
        return value


class IngestionJobState(BaseModel):
    job_id: UUID
    user_id: UUID
    status: IngestionStatus
    attempt: int
    original_filename: str
    collection_name: str
    scope_id: UUID
    chunks_written: int = 0
    document_id: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None

    @classmethod
    def from_job(
        cls,
        job: IngestionJob,
        status: IngestionStatus = IngestionStatus.QUEUED,
    ) -> "IngestionJobState":
        return cls(
            job_id=job.job_id,
            user_id=job.user_id,
            status=status,
            attempt=job.attempt,
            original_filename=job.original_filename,
            collection_name=job.collection_name,
            scope_id=job.scope_id,
            created_at=job.created_at,
        )
