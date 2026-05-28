from typing import List, Dict
import traceback
from uuid import UUID, uuid4
from pydantic import BaseModel

from rag.agent import RAGAgent, ChatResponse, Status, Metadata
from fastapi import HTTPException, APIRouter
from fastapi.responses import StreamingResponse
from fastapi import Depends
from pyapi.api.dependency import get_agent

router = APIRouter(
    prefix="/api/agent/conversation"
)

class CreateSessionResponse(BaseModel):
    conversation_id: UUID

class ChatRequest(BaseModel):
    query: str
    conversation_id: UUID
    user_id: UUID

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

@router.post("/chat")
async def chat_with_agent(request: ChatRequest, agent: RAGAgent = Depends(get_agent)):
    try:
        async def stream_response():
            async for response in agent.chat(query=request.query, conversation_id=str(request.conversation_id), user_id=str(request.user_id)):
                yield response.model_dump_json(indent=None) + "\n"
            
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
async def get_session_history(conversation_id: UUID, agent: RAGAgent = Depends(get_agent)):
    try:
        history = await agent.get_full_history(str(conversation_id))
        
        return SessionHistoryResponse(
            conversation_id=str(conversation_id),
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
