import logging
from typing import Dict, List

from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict

from rag.agent import RAGAgent, Status
from fastapi import HTTPException, APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from rag_agent_server.api.dependency import get_agent
from rag_agent_server.api.auth import get_current_user

router = APIRouter(
    prefix="/api/agent/conversation"
)
logger = logging.getLogger(__name__)


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

class CreateSessionResponse(BaseModel):
    conversation_id: UUID

class ChatRequest(StrictRequest):
    query: str
    conversation_id: UUID
    enable_exa: bool = False

class InterruptRequest(StrictRequest):
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

class ClearSessionRequest(StrictRequest):
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
    except Exception as error:
        logger.exception("Failed to create conversation", extra={"user_id": user_id})
        raise HTTPException(
            status_code=500,
            detail="Failed to create conversation.",
        ) from error

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
    except Exception as error:
        logger.exception("Failed to list conversations", extra={"user_id": user_id})
        raise HTTPException(
            status_code=500,
            detail="Failed to list conversations.",
        ) from error

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
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Conversation does not exist.",
                    )

        user_query = request.query

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
                    except Exception:
                        logger.exception(
                            "Failed to update conversation title",
                            extra={"conversation_id": request.conversation_id},
                        )
                yield response.model_dump_json(indent=None) + "\n"

        return StreamingResponse(
            content=stream_response(),
            media_type="application/x-ndjson",
        )

    except HTTPException:
        raise
    except Exception as error:
        logger.exception(
            "Conversation request failed",
            extra={"conversation_id": request.conversation_id, "user_id": user_id},
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to process conversation.",
        ) from error

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
                if row is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Conversation does not exist.",
                    )
                if row[0] != user_id:
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
    except Exception as error:
        logger.exception(
            "Failed to interrupt conversation",
            extra={"conversation_id": request.conversation_id, "user_id": user_id},
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to interrupt conversation.",
        ) from error

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
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Conversation does not exist.",
                    )

        history = await agent.get_full_history(str(conversation_id))

        return SessionHistoryResponse(
            conversation_id=str(conversation_id),
            history=history
        )

    except HTTPException:
        raise
    except Exception as error:
        logger.exception(
            "Failed to retrieve conversation history",
            extra={"conversation_id": conversation_id, "user_id": user_id},
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve conversation history.",
        ) from error

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
                if row is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Conversation does not exist.",
                    )
                if row[0] != user_id:
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
    except Exception as error:
        logger.exception(
            "Failed to clear conversation",
            extra={"conversation_id": request.conversation_id, "user_id": user_id},
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to clear conversation.",
        ) from error
