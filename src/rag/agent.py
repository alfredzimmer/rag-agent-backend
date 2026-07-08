"""Minimal RAG agent.

One flow: retrieve top-k documents from Milvus, put them in the system
prompt, stream the answer from Ollama. Conversation history is kept
in memory per conversation id.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncIterator, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from .config import RAGConfig
from .milvus import create_milvus_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an assistant for engineering standards and internal reference material. Answer the user's question using the numbered sources below.

Rules:
- Base every claim on the sources and cite them inline as [1], [2], etc.
- If the sources do not contain the information needed, say so and name what is missing — do not guess or invent code requirements.
- If you supplement with general knowledge, put it after the grounded answer under a line reading "Beyond the sources:" so the user can tell them apart.

Sources:
{context}"""

NO_RESULTS = "No relevant documents found in the knowledge base."


class Status(Enum):
    CREATED = "created"
    RESPONSE = "response"
    USAGE = "usage"
    FUNCTION = "function"
    COMPLETE = "complete"
    CANCEL = "cancel"
    ERROR = "error"


class Metadata(BaseModel):
    conversation_id: str = Field(..., description="Session ID")
    rating: float = Field(0.0, description="Rating result")
    title: Optional[str] = Field(None, description="Title of the conversation")
    input_tokens_used: int = Field(0, description="Number of input tokens used")
    output_tokens_used: int = Field(0, description="Number of output tokens used")


class ChatResponse(BaseModel):
    status: Status
    type: str = Field(..., description="The type of response")
    content: str = Field(..., description="The content of the response")
    metadata: Metadata = Field(..., description="Metadata about the response")


@dataclass
class Session:
    title: str = "New Chat"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    messages: list[BaseMessage] = field(default_factory=list)


class RAGAgent:
    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.vector_store = create_milvus_store(self.config)
        self.llm = ChatOllama(
            model=self.config.llm_model,
            base_url=self.config.ollama_host,
            temperature=0.15,
            num_ctx=self.config.llm_num_ctx,
            reasoning=self.config.llm_reasoning,
            num_predict=self.config.llm_num_predict,
        )
        self.sessions: dict[str, Session] = {}
        self._interrupted: set[str] = set()

    # -- Sessions ----------------------------------------------------------

    def create_session(self) -> str:
        conversation_id = str(uuid4())
        self.sessions[conversation_id] = Session()
        return conversation_id

    def has_session(self, conversation_id: str) -> bool:
        return conversation_id in self.sessions

    def list_sessions(self) -> list[dict]:
        return [
            {
                "conversation_id": conversation_id,
                "title": session.title,
                "created_at": session.created_at.isoformat(),
            }
            for conversation_id, session in sorted(
                self.sessions.items(), key=lambda item: item[1].created_at, reverse=True
            )
        ]

    def get_history(self, conversation_id: str) -> list[dict]:
        session = self.sessions.get(conversation_id)
        if session is None:
            return []
        roles = {HumanMessage: ("user", "human"), AIMessage: ("assistant", "ai")}
        history = []
        for message in session.messages:
            role, kind = roles.get(type(message), ("unknown", "unknown"))
            history.append({"role": role, "type": kind, "content": str(message.content)})
        return history

    def clear_session(self, conversation_id: str) -> bool:
        return self.sessions.pop(conversation_id, None) is not None

    def interrupt(self, conversation_id: str) -> bool:
        if not self.has_session(conversation_id):
            return False
        self._interrupted.add(conversation_id)
        return True

    # -- RAG ---------------------------------------------------------------

    @staticmethod
    def format_source(index: int, doc) -> str:
        """One numbered source: [n] document › header path, then the text."""
        meta = doc.metadata
        name = meta.get("name") or meta.get("original_filename") or "unknown"
        headers = [meta[k] for k in sorted(meta) if k.startswith("Header") and meta[k]]
        title = " › ".join([str(name), *map(str, headers)])
        return f"[{index}] {title}\n{doc.page_content}"

    async def retrieve(self, query: str) -> str:
        """Search Milvus and serialize the top documents for the prompt."""
        docs = await asyncio.to_thread(
            self.vector_store.similarity_search, query, self.config.top_k
        )
        if not docs:
            return NO_RESULTS
        return "\n\n".join(self.format_source(i, doc) for i, doc in enumerate(docs, 1))

    async def chat(self, query: str, conversation_id: str) -> AsyncIterator[ChatResponse]:
        """Stream one conversation turn as ChatResponse events."""
        session = self.sessions[conversation_id]
        self._interrupted.discard(conversation_id)

        def event(status: Status, type_: str, content: str, **metadata) -> ChatResponse:
            return ChatResponse(
                status=status,
                type=type_,
                content=content,
                metadata=Metadata(conversation_id=conversation_id, **metadata),
            )

        yield event(
            Status.RESPONSE,
            "response.function_call_arguments.delta",
            f"Calling retrieve with args: {{'query': '{query}'}}",
        )

        try:
            context = await self.retrieve(query)
        except Exception:
            logger.exception("Retrieval failed", extra={"conversation_id": conversation_id})
            context = NO_RESULTS

        yield event(Status.FUNCTION, "retrieve", context)

        messages: list[BaseMessage] = [
            *session.messages,
            HumanMessage(content=f"{SYSTEM_PROMPT.format(context=context)}\n\nQuestion: {query}"),
        ]

        input_tokens = 0
        output_tokens = 0
        answer = ""

        async for chunk in self.llm.astream(messages):
            if conversation_id in self._interrupted:
                self._interrupted.discard(conversation_id)
                yield event(
                    Status.CANCEL,
                    "chat.cancel",
                    "",
                    input_tokens_used=input_tokens,
                    output_tokens_used=output_tokens,
                )
                return

            reasoning = chunk.additional_kwargs.get("reasoning_content", "")
            if reasoning:
                yield event(Status.RESPONSE, "response.reasoning.delta", reasoning)

            if chunk.content:
                answer += str(chunk.content)
                yield event(Status.RESPONSE, "response.output_text.delta", str(chunk.content))

            if chunk.usage_metadata:
                input_tokens += chunk.usage_metadata.get("input_tokens", 0)
                output_tokens += chunk.usage_metadata.get("output_tokens", 0)

        session.messages.extend([HumanMessage(content=query), AIMessage(content=answer)])
        if session.title == "New Chat":
            session.title = query.strip()[:60] or "New Chat"

        yield event(
            Status.COMPLETE,
            "completion",
            "",
            title=session.title,
            input_tokens_used=input_tokens,
            output_tokens_used=output_tokens,
        )
