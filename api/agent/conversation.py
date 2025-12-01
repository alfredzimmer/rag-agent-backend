from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import traceback

from requests.models import Response

from src.rag.agent import agent_chat, get_session_history, clear_session, RAGConfig
from fastapi import HTTPException, APIRouter

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
    query: str = Field(..., description="The question to ask the RAG system")
    session_id: Optional[str] = Field(None, description="Session ID to continue a conversation. If None, creates new session.")
    config: Optional[RAGConfig] = Field(None, description="Optional RAG configuration")
    clear_history: Optional[bool] = Field(False, description="If True, clears conversation history for this session")

class SessionHistoryResponse(BaseModel):
    session_id: str = Field(..., description="The session ID")
    history: List[Dict[str, str]] = Field(..., description="Conversation history")

class ClearSessionResponse(BaseModel):
    success: bool = Field(..., description="Whether the session was successfully cleared")
    message: str = Field(..., description="Status message")

@router.post("/", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    try:
        # Build config from request or use defaults
        if request.config:
            config = request.config
        else:
            config = RAGConfig()
        
        # Call the chat agent
        response = agent_chat(
            query=request.query,
            session_id=request.session_id,
            config=config,
            clear_history=request.clear_history
        )
        
        return response
    
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