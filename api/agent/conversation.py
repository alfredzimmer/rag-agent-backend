from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict
import traceback
from uuid import UUID, uuid4

from src.rag.agent import RAGAgent
from fastapi import HTTPException, APIRouter
from fastapi.responses import StreamingResponse
from fastapi import Depends
from api.dependency import get_agent

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

class CreateSessionResponse(BaseModel):
    conversation_id: UUID = Field(..., description="The conversation ID")

class ChatRequest(BaseModel):
    query: str = Field(..., description="The question to ask the RAG system")
    conversation_id: UUID = Field(..., description="The conversation ID")
    user_id: str = Field(..., description="The user ID")

class ChatResponse(BaseModel):
    status: Status
    type: str = Field(..., description="The type of response")
    content: str = Field(..., description="The content of the response")
    metadata: Metadata = Field(..., description="Metadata about the response")

class InterruptRequest(BaseModel):
    conversation_id: UUID = Field(..., description="The conversation ID")

class InterruptResponse(BaseModel):
    success: bool = Field(..., description="Whether the chat was interrupted successfully")
    message: str = Field(..., description="Status message")

class SessionHistoryResponse(BaseModel):
    conversation_id: UUID = Field(..., description="The conversation ID")
    history: List[Dict[str, str]] = Field(..., description="Conversation history")

class ClearSessionResponse(BaseModel):
    success: bool = Field(..., description="Whether the session was successfully cleared")
    message: str = Field(..., description="Status message")

class ClearSessionRequest(BaseModel):
    conversation_id: UUID = Field(..., description="The conversation ID")

class SessionHistoryRequest(BaseModel):
    conversation_id: UUID = Field(..., description="The conversation ID")


@router.get("/create")
async def create_session():
    conversation_id: UUID = uuid4()
    try:
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

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest, agent: RAGAgent = Depends(get_agent)):
    try:
        responses = agent.chat(query=request.query, conversation_id=str(request.conversation_id), user_id=str(request.user_id))
        async def stream_response():
            async for response in responses:
                yield response.model_dump_json(indent=None)
            
        return StreamingResponse(
            content=stream_response(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        print(f"Error in chat_with_agent: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )

@router.post("/interrupt", response_model=InterruptResponse)
async def interrupt_chat(request: InterruptRequest, agent: RAGAgent = Depends(get_agent)):
    try:
        if (agent.interrupt(conversation_id=str(request.conversation_id))):
            return InterruptResponse(
                success=True,
                message="Chat interrupted successfully"
            )
    except Exception as e:
        print(f"Error in interrupt_chat: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error interrupting chat: {str(e)}"
        )



@router.get("/history", response_model=SessionHistoryResponse)
async def get_session_history(request: SessionHistoryRequest, agent: RAGAgent = Depends(get_agent)):
    try:
        history = await agent.get_full_history(str(request.conversation_id))
        
        return SessionHistoryResponse(
            conversation_id=str(request.conversation_id),
            history=history
        )
    
    except Exception as e:
        print(f"Error in get_chat_history: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chat history: {str(e)}"
        )

@router.delete("/clear", response_model=ClearSessionResponse)
async def clear_session(request: ClearSessionRequest, agent: RAGAgent = Depends(get_agent)):
    try:
        success = await agent.clear_session(str(request.conversation_id))
        
        if success:
            return ClearSessionResponse(
                success=True,
                message=f"Session {request.conversation_id} cleared successfully"
            )
        else:
            return ClearSessionResponse(
                success=False,
                message=f"Session {request.conversation_id} not found"
            )
    
    except Exception as e:
        print(f"Error in clear_chat_session: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing chat session: {str(e)}"
        )