from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from src.rag.agent import RAGAgent, RAGConfig
import traceback 

router = APIRouter(
    prefix="/api/agent/response"
)

class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask the RAG system")
    config: Optional[RAGConfig] = Field(None, description="Optional RAG configuration")

class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., description="List of questions to ask the RAG system")
    config: Optional[RAGConfig] = Field(None, description="Optional RAG configuration")

class QueryResponse(BaseModel):
    response: str = Field(..., description="The generated response from the RAG system")
    sources: List[str] = Field(..., description="Source documents used to generate the response")

@router.post("/", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the agent with a single question.
    """
    try:
        # Build config from request or use defaults
        if request.config:
            config = request.config
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
