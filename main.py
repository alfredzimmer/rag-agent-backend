from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.agent.conversation import router as conversation_router
from api.agent.response import router as response_router
from api.agent.status import router as status_router
from src.rag.agent import RAGAgent, RAGConfig

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing agent...")
    app.state.agent = RAGAgent(config=RAGConfig(), session_id="1")
    yield
    print("Cleaning up agent...")
    del app.state.agent

app = FastAPI(
    title="EC Master Agent API",
    description="API for querying the EC Master Agent",
    version="1.0.0",
    lifespan=lifespan
)

origins = [
    "https://pis3.aempro.ca",
    "http://localhost:3000",
    "http://localhost:5173",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(conversation_router)
app.include_router(response_router)
app.include_router(status_router)

@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "message": "API is running!"
    }