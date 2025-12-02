from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import traceback

from src.rag.agent import agent_chat, get_session_history, clear_session, RAGConfig
from fastapi import HTTPException, APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(
    prefix="/api/agent/conversation"
)

class Status(Enum):
    RESPONSE = "response"
    USAGE = "usage"
    FUNCTION = "function"
    COMPLETE = "complete"
    CANCEL = "cancel"
    ERROR = "error"

class Metadata(BaseModel):
    session_id: str = Field(..., description="Session ID")
    tokens_used: int = Field(..., description="Number of tokens used")

class ChatRequest(BaseModel):
    input: str = Field(..., description="The question to ask the RAG system")
    session_id: Optional[str] = Field(None, description="Session ID to continue a conversation. If None, creates new session.")

class ChatResponse(BaseModel):
    status: Status
    type: str = Field(..., description="The type of response")
    content: str = Field(..., description="The content of the response")
    metadata: Metadata = Field(..., description="Metadata about the response")

class SessionHistoryResponse(BaseModel):
    session_id: str = Field(..., description="The session ID")
    history: List[Dict[str, str]] = Field(..., description="Conversation history")

class ClearSessionResponse(BaseModel):
    success: bool = Field(..., description="Whether the session was successfully cleared")
    message: str = Field(..., description="Status message")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    try:
        
        # Call the chat agent
        responses = agent_chat(
            query=request.input,
            session_id=request.session_id
        )

        async def stream_response():
            async for response in responses:
                yield response.model_dump_json(indent=None)
        
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        print(f"Error in chat_with_agent: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )


@router.get("/history/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    try:
        history = get_session_history(session_id)
        
        return SessionHistoryResponse(
            session_id=session_id,
            history=history
        )
    
    except Exception as e:
        print(f"Error in get_chat_history: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chat history: {str(e)}"
        )

@router.delete("/clear/{session_id}", response_model=ClearSessionResponse)
async def clear_session(session_id: str):
    try:
        success = clear_session(session_id)
        
        if success:
            return ClearSessionResponse(
                success=True,
                message=f"Session {session_id} cleared successfully"
            )
        else:
            return ClearSessionResponse(
                success=False,
                message=f"Session {session_id} not found"
            )
    
    except Exception as e:
        print(f"Error in clear_chat_session: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing chat session: {str(e)}"
        )