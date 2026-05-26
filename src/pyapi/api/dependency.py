from fastapi import Request
from rag.agent import RAGAgent

def get_agent(request: Request) -> RAGAgent:
    return request.app.state.agent
