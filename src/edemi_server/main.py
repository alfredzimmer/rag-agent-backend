import os
import logging
import socket
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

load_dotenv()

from edemi_server.observability import configure_telemetry, instrument_fastapi

configure_telemetry("edemi-api")

from edemi_server.api.agent.conversation import router as conversation_router
from edemi_server.api.auth import router as auth_router, init_auth_db
from edemi_server.api.ingestion import router as ingestion_router
from edemi_server.config import get_cors_origins
from rag.agent import RAGAgent, RAGConfig

logger = logging.getLogger(__name__)


def ping_services():
    services = {}

    # 1. Postgres
    pg_uri = os.getenv("PG_URI", "postgresql://edemi:edemi@localhost:5433/edemi")
    try:
        parsed = urlparse(pg_uri)
        pg_host = parsed.hostname or "localhost"
        pg_port = parsed.port or 5433
    except Exception:
        pg_host = "localhost"
        pg_port = 5433
    services["Postgres"] = (pg_host, pg_port)

    # 2. Milvus
    milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    try:
        parsed = urlparse(milvus_uri)
        milvus_host = parsed.hostname or os.getenv("MILVUS_HOST", "localhost")
        milvus_port = parsed.port or int(os.getenv("MILVUS_PORT", "19530"))
    except Exception:
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
    services["Milvus"] = (milvus_host, milvus_port)

    # 3. Redis
    redis_url = os.getenv("REDIS_URL")
    try:
        parsed = urlparse(redis_url) if redis_url else None
        redis_host = (
            parsed.hostname
            if parsed and parsed.hostname
            else os.getenv("REDIS_HOST", "localhost")
        )
        redis_port = (
            parsed.port
            if parsed and parsed.port
            else int(os.getenv("REDIS_PORT", "6380"))
        )
    except (TypeError, ValueError):
        redis_host = "localhost"
        redis_port = 6380
    services["Redis"] = (redis_host, redis_port)

    # 4. Ollama
    ollama_host_env = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        parsed = urlparse(ollama_host_env)
        ollama_host = parsed.hostname or "localhost"
        ollama_port = parsed.port or 11434
    except Exception:
        ollama_host = "localhost"
        ollama_port = 11434
    services["Ollama"] = (ollama_host, ollama_port)

    # 5. MinIO
    services["MinIO"] = (
        os.getenv("MINIO_HOST", "localhost"),
        int(os.getenv("MINIO_PORT", "9000")),
    )

    all_online = True
    for name, (host, port) in services.items():
        try:
            with socket.create_connection((host, port), timeout=1.5):
                status = "ONLINE"
        except Exception:
            status = "OFFLINE"
            all_online = False

        logger.log(
            logging.INFO if status == "ONLINE" else logging.WARNING,
            "Dependency connectivity check",
            extra={"dependency": name, "host": host, "port": port, "status": status},
        )
    return all_online


@asynccontextmanager
async def lifespan(app: FastAPI):
    ping_services()

    logger.info("Initializing RAG agent")
    config = RAGConfig(
        llm_model=os.getenv("RAG_LLM_MODEL", "qwen3.6"),
        collection_name=os.getenv("RAG_COLLECTION_NAME", "HeaderInContentTrial"),
    )
    app.state.agent = await RAGAgent.create(config)

    await init_auth_db(app.state.agent.pool)

    yield
    logger.info("Closing RAG agent")
    await app.state.agent.close()


app = FastAPI(
    title="Edemi Backend",
    description="API for the Edemi agent and retrieval runtime",
    version="1.0.0",
    lifespan=lifespan
)
instrument_fastapi(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(conversation_router)
app.include_router(ingestion_router)


@app.get("/health", include_in_schema=False)
def health():
    return {
        "status": "ok",
        "service": "edemi-api",
    }


def run() -> None:
    import uvicorn

    port = int(os.getenv("EDEMI_PORT", "9229"))
    reload_enabled = os.getenv("EDEMI_RELOAD", "false").lower() == "true"
    logger.info("Starting Edemi API", extra={"port": port})
    uvicorn.run(
        "edemi_server.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload_enabled,
        log_config=None,
    )


if __name__ == "__main__":
    run()
