from typing import List, Dict, Optional
import traceback
import os
import json
import redis
import asyncio

from uuid import UUID, uuid4
from pydantic import BaseModel

from rag.agent import RAGAgent, Status
from fastapi import HTTPException, APIRouter, Depends, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from edemi_server.api.dependency import get_agent
from edemi_server.api.auth import get_current_user

router = APIRouter(
    prefix="/api/agent/conversation"
)

class CreateSessionResponse(BaseModel):
    conversation_id: UUID

class ChatRequest(BaseModel):
    query: str
    conversation_id: UUID
    user_id: UUID
    file_context: Optional[str] = None
    enable_exa: Optional[bool] = False

class InterruptRequest(BaseModel):
    conversation_id: UUID

class InterruptResponse(BaseModel):
    success: bool
    message: str

class SessionHistoryResponse(BaseModel):
    conversation_id: UUID
    history: List[Dict[str, str]]

class ClearSessionResponse(BaseModel):
    success: bool
    message: str

class ClearSessionRequest(BaseModel):
    conversation_id: UUID

class SessionHistoryRequest(BaseModel):
    conversation_id: UUID


@router.get("/create")
async def create_session(
    agent: RAGAgent = Depends(get_agent),
    current_user: dict = Depends(get_current_user)
):
    conversation_id: UUID = uuid4()
    user_id = current_user["user_id"]
    try:
        # Create mapping in user_sessions on creation
        async with agent.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO user_sessions (conversation_id, user_id, title) VALUES (%s, %s, %s)",
                    (conversation_id, user_id, "New Chat")
                )
        return CreateSessionResponse(
            conversation_id=conversation_id
        )
    except Exception as e:
        print(f"Error in create_session: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error creating session: {str(e)}"
        )

@router.get("/list", response_model=List[dict])
async def list_user_sessions(
    agent: RAGAgent = Depends(get_agent),
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    try:
        async with agent.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT conversation_id, title, created_at
                    FROM user_sessions
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    """,
                    (user_id,)
                )
                rows = await cur.fetchall()
                sessions = []
                for row in rows:
                    sessions.append({
                        "conversation_id": str(row[0]),
                        "title": row[1] or "New Chat",
                        "created_at": row[2].isoformat() if row[2] else ""
                    })
                return sessions
    except Exception as e:
        print(f"Error in list_user_sessions: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error listing sessions: {str(e)}"
        )

@router.post("/chat")
async def chat_with_agent(
    request: ChatRequest,
    agent: RAGAgent = Depends(get_agent),
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    try:
        # Verify and enforce session ownership
        async with agent.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT user_id FROM user_sessions WHERE conversation_id = %s",
                    (request.conversation_id,)
                )
                row = await cur.fetchone()
                if row:
                    if row[0] != user_id:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Forbidden: You do not own this chat session."
                        )
                else:
                    # If this is a new session, associate it with the authenticated user
                    await cur.execute(
                        "INSERT INTO user_sessions (conversation_id, user_id, title) VALUES (%s, %s, %s)",
                        (request.conversation_id, user_id, "New Chat")
                    )

        user_query = request.query
        if request.file_context:
            user_query = (
                f"Attached Document Context:\n"
                f"----------------------\n"
                f"{request.file_context}\n"
                f"----------------------\n\n"
                f"User Question:\n{request.query}"
            )

        async def stream_response():
            # Pass the actual authenticated user ID to the agent execution
            async for response in agent.chat(
                query=user_query,
                conversation_id=str(request.conversation_id),
                user_id=str(user_id),
                enable_exa=request.enable_exa
            ):
                # Intercept generated title to save it in user_sessions
                if response.status == Status.COMPLETE and response.metadata.title:
                    try:
                        async with agent.pool.connection() as conn:
                            async with conn.cursor() as cur:
                                await cur.execute(
                                    "UPDATE user_sessions SET title = %s WHERE conversation_id = %s",
                                    (response.metadata.title, request.conversation_id)
                                )
                    except Exception as db_err:
                        print(f"Error updating session title in DB: {db_err}")
                yield response.model_dump_json(indent=None) + "\n"

        return StreamingResponse(
            content=stream_response(),
            media_type="text/event-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat_with_agent: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )

@router.post("/interrupt", response_model=InterruptResponse)
async def interrupt_chat(
    request: InterruptRequest,
    agent: RAGAgent = Depends(get_agent),
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    try:
        # Verify ownership before interrupting
        async with agent.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT user_id FROM user_sessions WHERE conversation_id = %s",
                    (request.conversation_id,)
                )
                row = await cur.fetchone()
                if row and row[0] != user_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Forbidden: You do not own this chat session."
                    )

        if (agent.interrupt(conversation_id=str(request.conversation_id))):
            return InterruptResponse(
                success=True,
                message="Chat interrupted successfully"
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in interrupt_chat: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error interrupting chat: {str(e)}"
        )

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    ingest_to_milvus: bool = Form(default=False),
    conversation_id: UUID = Form(...),
    agent: RAGAgent = Depends(get_agent),
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    try:
        # Verify ownership of the conversation session
        async with agent.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT user_id FROM user_sessions WHERE conversation_id = %s",
                    (conversation_id,)
                )
                row = await cur.fetchone()
                if row and row[0] != user_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Forbidden: You do not own this chat session."
                    )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error checking session ownership during upload: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error during session check: {str(e)}"
        )

    # Restrict to PDF extension only
    filename = file.filename
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="File upload restricted to PDF (.pdf) extension only."
        )

    # Retrieve upload directory from env
    upload_dir = os.getenv("INGESTION_UPLOAD_DIR", "/Users/alfred/work/cpp-ingestor/files")
    os.makedirs(upload_dir, exist_ok=True)

    # Keep filename safe from traversal
    safe_filename = os.path.basename(filename)
    dest_path = os.path.join(upload_dir, safe_filename)

    # Save the file to host directory
    try:
        # If file exists and is read-only, make it writable before overwriting
        if os.path.exists(dest_path):
            try:
                os.chmod(dest_path, 0o644)
            except Exception as pe:
                print(f"Warning: Could not change permissions for {dest_path}: {pe}")

        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        print(f"Failed to write uploaded file to {dest_path}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )

    redis_enqueued = False
    if ingest_to_milvus:
        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6380"))

            # Connect to Redis
            r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

            job_id = str(uuid4())
            # Path inside the container is /app/files/filename
            container_pdf_path = f"/app/files/{safe_filename}"
            target_collection = str(conversation_id)

            payload = {
                "job_id": job_id,
                "pdf_path": container_pdf_path,
                "config_type": "generic",
                "target_collection": target_collection
            }

            # Enqueue payload to ingestion:jobs queue
            r.rpush("ingestion:jobs", json.dumps(payload))
            redis_enqueued = True
        except Exception as re:
            print(f"Failed to enqueue ingestion job to Redis: {re}")
            raise HTTPException(
                status_code=500,
                detail=f"File uploaded, but failed to queue for ingestion in Redis: {str(re)}"
            )

    # Extract text from PDF for direct injection in the current turn
    extracted_text = ""
    try:
        import pypdf
        def sync_extract():
            reader = pypdf.PdfReader(dest_path)
            # Extract up to first 15 pages to keep prompt context window safe
            text = ""
            for page in reader.pages[:15]:
                text += page.extract_text() or ""
            return text

        extracted_text = await asyncio.to_thread(sync_extract)
    except Exception as ext_err:
        print(f"Warning: Failed to extract text from uploaded PDF: {ext_err}")


    return {
        "success": True,
        "filename": safe_filename,
        "ingested": ingest_to_milvus,
        "redis_enqueued": redis_enqueued,
        "extracted_text": extracted_text,
        "message": "File uploaded and enqueued for database ingestion." if redis_enqueued else "File uploaded successfully."
    }

@router.get("/history", response_model=SessionHistoryResponse)
async def get_session_history(
    conversation_id: UUID,
    agent: RAGAgent = Depends(get_agent),
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    try:
        # Verify ownership
        async with agent.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT user_id FROM user_sessions WHERE conversation_id = %s",
                    (conversation_id,)
                )
                row = await cur.fetchone()
                if row:
                    if row[0] != user_id:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Forbidden: You do not own this chat session."
                        )
                else:
                    # Check if there are checkpoints for this conversation_id
                    await cur.execute(
                        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s",
                        (str(conversation_id),)
                    )
                    chk_count = (await cur.fetchone())[0]
                    if chk_count > 0:
                        # History exists but wasn't registered in user_sessions (e.g. was migrated).
                        # Let's map it under the authenticated user.
                        await cur.execute(
                            "INSERT INTO user_sessions (conversation_id, user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (conversation_id, user_id)
                        )
                    else:
                        # It is a brand new session, just return empty history
                        return SessionHistoryResponse(
                            conversation_id=str(conversation_id),
                            history=[]
                        )

        history = await agent.get_full_history(str(conversation_id))

        return SessionHistoryResponse(
            conversation_id=str(conversation_id),
            history=history
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_chat_history: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chat history: {str(e)}"
        )

@router.delete("/clear", response_model=ClearSessionResponse)
async def clear_session(
    request: ClearSessionRequest,
    agent: RAGAgent = Depends(get_agent),
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    try:
        # Verify ownership
        async with agent.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT user_id FROM user_sessions WHERE conversation_id = %s",
                    (request.conversation_id,)
                )
                row = await cur.fetchone()
                if row and row[0] != user_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Forbidden: You do not own this chat session."
                    )

        success = await agent.clear_session(str(request.conversation_id))

        if success:
            # Delete mapping from user_sessions
            async with agent.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "DELETE FROM user_sessions WHERE conversation_id = %s",
                        (request.conversation_id,)
                    )
            return ClearSessionResponse(
                success=True,
                message=f"Session {request.conversation_id} cleared successfully"
            )
        else:
            return ClearSessionResponse(
                success=False,
                message=f"Session {request.conversation_id} not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in clear_chat_session: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing chat session: {str(e)}"
        )
