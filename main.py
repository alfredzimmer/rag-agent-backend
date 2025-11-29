from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import traceback

from fastapi.middleware.cors import CORSMiddleware

# Import RAG components
from src.rag.agent import agent_call, agent_chat, get_session_history, clear_session, RAGConfig

# Initialize the app
app = FastAPI(
    title="RAG API",
    description="API for querying the RAG (Retrieval-Augmented Generation) system",
    version="1.0.0"
)

# CORS configuration - allow your frontend server and Tailscale access
origins = [
    "https://pis3.aempro.ca",  # Your main Node app
    "http://localhost:3000",   # Local testing
    "http://localhost:5173",   # Vite default port
    "*",                       # Allow all origins for Tailscale Funnel (you can restrict this later)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class RAGConfigRequest(BaseModel):
    """Configuration for RAG system"""
    vector_store_type: Optional[str] = "milvus"
    ranker_type: Optional[str] = "bge"
    dense_embedding_model: Optional[str] = "qwen3-embedding:8b"
    sparse_embedding_model: Optional[str] = "splade"
    llm_model: Optional[str] = "qwen3:8b"
    hyde: Optional[bool] = True

class QueryRequest(BaseModel):
    """Single query request"""
    query: str = Field(..., description="The question to ask the RAG system")
    config: Optional[RAGConfigRequest] = Field(None, description="Optional RAG configuration")

class BatchQueryRequest(BaseModel):
    """Batch query request"""
    queries: List[str] = Field(..., description="List of questions to ask the RAG system")
    config: Optional[RAGConfigRequest] = Field(None, description="Optional RAG configuration")

class QueryResponse(BaseModel):
    """Single query response"""
    response: str = Field(..., description="The generated response from the RAG system")
    sources: List[str] = Field(..., description="Source documents used to generate the response")

class ChatRequest(BaseModel):
    """Continuous chat request"""
    query: str = Field(..., description="The question to ask the RAG system")
    session_id: Optional[str] = Field(None, description="Session ID to continue a conversation. If None, creates new session.")
    config: Optional[RAGConfigRequest] = Field(None, description="Optional RAG configuration")
    clear_history: Optional[bool] = Field(False, description="If True, clears conversation history for this session")

class ChatResponse(BaseModel):
    """Continuous chat response"""
    response: str = Field(..., description="The generated response from the RAG system")
    sources: List[str] = Field(..., description="Source documents used to generate the response")
    session_id: str = Field(..., description="Session ID for continuing the conversation")

class SessionHistoryResponse(BaseModel):
    """Session history response"""
    session_id: str = Field(..., description="The session ID")
    history: List[Dict[str, str]] = Field(..., description="Conversation history")

class ClearSessionResponse(BaseModel):
    """Clear session response"""
    success: bool = Field(..., description="Whether the session was successfully cleared")
    message: str = Field(..., description="Status message")

# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/")
def read_root():
    """
    Root endpoint - API health check
    """
    return {
        "status": "ok", 
        "message": "RAG API is running!",
        "endpoints": {
            "health": "/health",
            "model_status": "/api/model-status",
            "query": "/api/query",
            "batch_query": "/api/batch-query",
            "chat": "/api/chat",
            "chat_history": "/api/chat/history/{session_id}",
            "chat_clear": "/api/chat/clear/{session_id}"
        }
    }

@app.get("/health")
def check_health():
    """
    Health check endpoint to confirm the API is live and reachable
    """
    return {"status": "ok", "message": "RAG API is healthy!"}

@app.get("/api/model-status")
def get_model_status():
    """
    Get information about available models and current configuration
    """
    return {
        "status": "ok",
        "available_models": {
            "llm": ["qwen3:8b"],
            "embedding": ["qwen3-embedding:8b"],
            "sparse_embedding": ["splade", "bm25", "bge"]
        },
        "default_config": {
            "vector_store_type": "milvus",
            "ranker_type": "bge",
            "dense_embedding_model": "qwen3-embedding:8b",
            "sparse_embedding_model": "splade",
            "llm_model": "qwen3:8b",
            "hyde": True
        }
    }

# ============================================================================
# RAG Query Endpoints
# ============================================================================

@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a single question.
    
    This endpoint accepts a question and optional configuration,
    then returns the generated response along with source documents.
    
    Example:
    ```json
    {
        "query": "What is machine learning?",
        "config": {
            "llm_model": "qwen3:8b",
            "hyde": true
        }
    }
    ```
    """
    try:
        # Build config from request or use defaults
        if request.config:
            config = RAGConfig(**request.config.model_dump())
        else:
            config = RAGConfig()
        
        # Call the RAG agent
        response, docs = agent_call(request.query, config)
        
        return QueryResponse(
            response=response,
            sources=docs
        )
    
    except Exception as e:
        print(f"Error in query_rag: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


# ============================================================================
# Continuous Conversation Endpoints
# ============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Chat with the RAG agent in a continuous conversation.
    
    This endpoint maintains conversation history across multiple requests using session IDs.
    The first request creates a new session, and subsequent requests with the same session_id
    continue the conversation.
    
    Example (New conversation):
    ```json
    {
        "query": "What is machine learning?"
    }
    ```
    
    Example (Continue conversation):
    ```json
    {
        "query": "Can you explain more about supervised learning?",
        "session_id": "abc-123-def-456"
    }
    ```
    
    Example (Clear history and start fresh):
    ```json
    {
        "query": "New topic: what is deep learning?",
        "session_id": "abc-123-def-456",
        "clear_history": true
    }
    ```
    """
    try:
        # Build config from request or use defaults
        if request.config:
            config = RAGConfig(**request.config.model_dump())
        else:
            config = RAGConfig()
        
        # Call the chat agent
        response, docs, session_id = agent_chat(
            query=request.query,
            session_id=request.session_id,
            config=config,
            clear_history=request.clear_history
        )
        
        return ChatResponse(
            response=response,
            sources=docs,
            session_id=session_id
        )
    
    except Exception as e:
        print(f"Error in chat_with_agent: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )

@app.get("/api/chat/history/{session_id}", response_model=SessionHistoryResponse)
async def get_chat_history(session_id: str):
    """
    Get the conversation history for a specific session.
    
    This endpoint returns all messages in the conversation, including user queries,
    assistant responses, and tool calls.
    
    Example:
    ```
    GET /api/chat/history/abc-123-def-456
    ```
    """
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

@app.delete("/api/chat/clear/{session_id}", response_model=ClearSessionResponse)
async def clear_chat_session(session_id: str):
    """
    Clear the conversation history for a specific session.
    
    This endpoint deletes all messages in the conversation and removes the session.
    
    Example:
    ```
    DELETE /api/chat/clear/abc-123-def-456
    ```
    """
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