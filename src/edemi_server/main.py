import os
import socket
from urllib.parse import urlparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from edemi_server.api.agent.conversation import router as conversation_router
from edemi_server.api.agent.status import router as status_router
from edemi_server.api.auth import router as auth_router, init_auth_db
from rag.agent import RAGAgent, RAGConfig

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
    redis_host = os.getenv("REDIS_HOST", "localhost")
    try:
        redis_port = int(os.getenv("REDIS_PORT", "6380"))
    except ValueError:
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
    services["MinIO"] = ("localhost", 9000)

    print("\n" + "=" * 50)
    print("      DIAGNOSTICS: PINGING HOSTED SERVICES      ")
    print("=" * 50)

    all_online = True
    for name, (host, port) in services.items():
        try:
            with socket.create_connection((host, port), timeout=1.5):
                status = "ONLINE"
        except Exception:
            status = "OFFLINE"
            all_online = False

        print(f" {name:<12} ({host}:{port})".ljust(35) + f": [ {status} ]")

    print("=" * 50)
    if not all_online:
        print(" WARNING: Some dependent services are OFFLINE.")
        print(" Please verify that all Docker containers and Ollama are running.")
    print("=" * 50 + "\n")
    return all_online

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Perform port-connectivity diagnostics prior to initializing the agent
    ping_services()

    print("AGENT:     Initializing agent...")
    config = RAGConfig(
        llm_model=os.getenv("RAG_LLM_MODEL", "qwen3.6"),
        collection_name=os.getenv("RAG_COLLECTION_NAME", "HeaderInContentTrial"),
    )
    app.state.agent = await RAGAgent.create(config)

    # Initialize multi-user auth database tables & run migrations
    await init_auth_db(app.state.agent.pool)


    yield
    print("AGENT:     Cleaning up agent...")
    await app.state.agent.close()

app = FastAPI(
    title="Edemi Backend",
    description="API for the Edemi agent and retrieval runtime",
    version="1.0.0",
    lifespan=lifespan
)

origins = [
    "https://chat.edemi.org",
    "https://pis3.aempro.ca",
    "http://localhost:3000",
    "http://localhost:5173",
    "https://chat.edemi.org",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(conversation_router)
# app.include_router(response_router)
app.include_router(status_router)

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "API is running!"
    }

if __name__ == "__main__":
    import uvicorn
    # Retrieve port from env (default to 9229)
    port = int(os.getenv("EDEMI_PORT", "9229"))
    print(f"Starting server on port {port}...")
    uvicorn.run("edemi_server.main:app", host="0.0.0.0", port=port, reload=True)
