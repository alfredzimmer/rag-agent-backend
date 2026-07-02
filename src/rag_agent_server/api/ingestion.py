from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from opentelemetry import metrics, trace

from rag_agent_server.api.auth import get_current_user
from rag_agent_server.api.dependency import get_agent
from rag_agent_server.ingestion.config import IngestionConfig
from rag_agent_server.ingestion.models import IngestionJob, IngestionJobState
from rag_agent_server.ingestion.parser import SUPPORTED_EXTENSIONS
from rag_agent_server.ingestion.queue import IngestionQueue
from rag.agent import RAGAgent

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)
uploads_counter = meter.create_counter(
    "rag_agent.ingestion.uploads",
    description="Documents accepted for asynchronous ingestion",
)

router = APIRouter(prefix="/api/ingestion", tags=["Ingestion"])


async def _require_owned_session(
    conversation_id: UUID,
    user_id: UUID,
    agent: RAGAgent,
) -> None:
    async with agent.pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT user_id FROM user_sessions WHERE conversation_id = %s",
                (conversation_id,),
            )
            row = await cursor.fetchone()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation does not exist.",
        )
    if row[0] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not own this conversation.",
        )


async def _persist_upload(
    upload: UploadFile,
    destination: Path,
    max_bytes: int,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    try:
        with destination.open("xb") as output:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Document exceeds the {max_bytes} byte upload limit.",
                    )
                digest.update(chunk)
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return digest.hexdigest(), total


@router.post(
    "/documents",
    response_model=IngestionJobState,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_document(
    conversation_id: UUID,
    file: UploadFile = File(...),
    agent: RAGAgent = Depends(get_agent),
    current_user: dict = Depends(get_current_user),
) -> IngestionJobState:
    user_id: UUID = current_user["user_id"]
    await _require_owned_session(conversation_id, user_id, agent)

    original_filename = Path(file.filename or "").name
    extension = Path(original_filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported document type. Supported extensions: {supported}.",
        )

    config = IngestionConfig.from_env()
    config.upload_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid4()
    destination = config.upload_dir / f"{job_id}{extension}"
    max_upload_bytes = int(os.getenv("INGESTION_MAX_UPLOAD_BYTES", "52428800"))

    with tracer.start_as_current_span("ingestion.accept_document") as span:
        source_sha256, size_bytes = await _persist_upload(
            file,
            destination,
            max_upload_bytes,
        )
        span.set_attributes(
            {
                "rag_agent.ingestion.job_id": str(job_id),
                "rag_agent.ingestion.scope_id": str(conversation_id),
                "rag_agent.ingestion.filename": original_filename,
                "rag_agent.ingestion.size_bytes": size_bytes,
            }
        )

        job = IngestionJob(
            job_id=job_id,
            user_id=user_id,
            scope_id=conversation_id,
            source_path=destination,
            original_filename=original_filename,
            source_sha256=source_sha256,
            collection_name=config.collection_name,
        )
        queue = IngestionQueue(config)
        try:
            await asyncio.to_thread(queue.enqueue, job)
        except Exception:
            destination.unlink(missing_ok=True)
            logger.exception(
                "Failed to enqueue ingestion job",
                extra={"job_id": job_id, "user_id": user_id},
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The ingestion queue is unavailable.",
            )

    uploads_counter.add(1, {"collection": config.collection_name})
    logger.info(
        "Document queued for ingestion",
        extra={
            "job_id": job_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "document_id": source_sha256,
        },
    )
    return IngestionJobState.from_job(job)


@router.get("/jobs/{job_id}", response_model=IngestionJobState)
async def get_ingestion_job(
    job_id: UUID,
    current_user: dict = Depends(get_current_user),
) -> IngestionJobState:
    queue = IngestionQueue()
    state = await asyncio.to_thread(queue.get_state, str(job_id))
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingestion job does not exist.",
        )
    if state.user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not own this ingestion job.",
        )
    return state
