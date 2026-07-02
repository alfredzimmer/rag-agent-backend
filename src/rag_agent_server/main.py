"""FastAPI server for the RAG agent."""
import logging
import os
import socket
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from uuid import UUID

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

load_dotenv()

from rag.agent import RAGAgent
from rag.config import RAGConfig
from rag_agent_server.config import get_cors_origins

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def service_targets(config: RAGConfig | None = None) -> dict[str, tuple[str, int]]:
    """Host/port of each runtime dependency, resolved from the environment."""
    config = config or RAGConfig()
    milvus = urlparse(config.milvus_uri)
    ollama = urlparse(config.ollama_host)
    return {
        "milvus": (milvus.hostname or "localhost", milvus.port or 19530),
        "ollama": (ollama.hostname or "localhost", ollama.port or 11434),
    }


def check_dependencies(timeout: float = 1.5) -> dict[str, str]:
    statuses = {}
    for name, (host, port) in service_targets().items():
        try:
            with socket.create_connection((host, port), timeout=timeout):
                statuses[name] = "online"
        except OSError:
            statuses[name] = "offline"
            logger.warning(
                "Dependency is unreachable",
                extra={"dependency": name, "host": host, "port": port},
            )
    return statuses


@asynccontextmanager
async def lifespan(app: FastAPI):
    statuses = check_dependencies()
    logger.info("Dependency connectivity", extra={"statuses": statuses})
    app.state.agent = RAGAgent(RAGConfig())
    logger.info("RAG agent ready")
    yield


app = FastAPI(
    title="RAG Agent Backend",
    description="Minimal RAG agent: Milvus retrieval + Ollama generation.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


def get_agent(request: Request) -> RAGAgent:
    return request.app.state.agent


def require_session(agent: RAGAgent, conversation_id: UUID) -> str:
    conversation_id = str(conversation_id)
    if not agent.has_session(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation does not exist.")
    return conversation_id


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatRequest(StrictRequest):
    query: str
    conversation_id: UUID


class ConversationRequest(StrictRequest):
    conversation_id: UUID


@app.get("/health", include_in_schema=False)
def health() -> JSONResponse:
    statuses = check_dependencies()
    ok = all(status == "online" for status in statuses.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": "ok" if ok else "degraded",
            "service": "rag-agent-api",
            "dependencies": statuses,
        },
    )


@app.get("/api/agent/conversation/create")
def create_session(request: Request):
    return {"conversation_id": get_agent(request).create_session()}


@app.get("/api/agent/conversation/list")
def list_sessions(request: Request):
    return get_agent(request).list_sessions()


@app.post("/api/agent/conversation/chat")
def chat(body: ChatRequest, request: Request):
    agent = get_agent(request)
    conversation_id = require_session(agent, body.conversation_id)

    async def stream():
        async for response in agent.chat(body.query, conversation_id):
            yield response.model_dump_json() + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/agent/conversation/interrupt")
def interrupt(body: ConversationRequest, request: Request):
    agent = get_agent(request)
    conversation_id = require_session(agent, body.conversation_id)
    agent.interrupt(conversation_id)
    return {"success": True, "message": "Chat interrupted successfully"}


@app.get("/api/agent/conversation/history")
def history(conversation_id: UUID, request: Request):
    agent = get_agent(request)
    conversation_id = require_session(agent, conversation_id)
    return {
        "conversation_id": conversation_id,
        "history": agent.get_history(conversation_id),
    }


@app.delete("/api/agent/conversation/clear")
def clear(body: ConversationRequest, request: Request):
    agent = get_agent(request)
    conversation_id = require_session(agent, body.conversation_id)
    agent.clear_session(conversation_id)
    return {"success": True, "message": f"Session {conversation_id} cleared successfully"}


def run() -> None:
    import uvicorn

    port = int(os.getenv("RAG_AGENT_PORT", "9229"))
    reload_enabled = os.getenv("RAG_AGENT_RELOAD", "false").lower() == "true"
    logger.info("Starting RAG Agent API", extra={"port": port})
    uvicorn.run(
        "rag_agent_server.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    run()
