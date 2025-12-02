from fastapi import Request
from src.rag.agent import RAGAgent, RAGConfig

def get_agent(request: Request) -> RAGAgent:
    return request.app.state.agent