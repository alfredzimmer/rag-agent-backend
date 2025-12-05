"""
Self-RAG Agent implementation.

This module provides a RAG agent that uses the Self-RAG strategy with
self-reflection and self-grading on retrieved documents and generations.
"""

import os
import asyncio
from typing import Union, List, Optional
from dataclasses import dataclass
from enum import Enum

# --- LangChain / LangGraph Imports ---
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document

# --- Database / Store Imports ---
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

# --- Memory Imports ---
from langmem import create_memory_store_manager

# --- Local Imports ---
from .milvus import create_milvus_store
from .modules.reranker import BGERanker
from .modules.hyde import HyDEGenerator
from .self_rag_graph import create_self_rag_graph

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
DB_URI: str = os.getenv("PG_URI", "")


@dataclass
class RAGConfig:
    vector_store_type: str = "milvus"
    ranker_type: str = "bge"
    dense_embedding_model: str = "qwen3-embedding:8b"
    sparse_embedding_model: str = "splade"  # [splade, bm25, bge]
    llm_model: str = "qwen3:8b"
    memory_llm_model: str = "qwen3:8b"
    memory_embeddings_model: str = "nomic-embed-text"
    memory_embeddings_dims: int = 768
    training_mode: bool = False
    training_llm_model: str = "qwen3:8b"
    hyde: bool = False
    # Self-RAG specific settings
    retrieval_k: int = 30  # Number of documents to retrieve initially
    rerank_k: int = 5  # Number of documents after reranking
    grade_documents: bool = True  # Whether to grade documents (disable for faster response)


@dataclass
class Context:
    user_id: str


class Status(Enum):
    CREATED = "created"
    RESPONSE = "response"
    USAGE = "usage"
    FUNCTION = "function"
    RETRIEVAL = "retrieval"  # New status for Self-RAG retrieval info
    GRADING = "grading"  # New status for Self-RAG grading info
    COMPLETE = "complete"
    CANCEL = "cancel"
    ERROR = "error"


class Metadata(BaseModel):
    conversation_id: str = Field(..., description="Session ID")
    rating: float = Field(..., description="Rating result")
    input_tokens_used: int = Field(..., description="Number of input tokens used")
    output_tokens_used: int = Field(..., description="Number of output tokens used")

2
class ChatResponse(BaseModel):
    status: Status
    type: str = Field(..., description="The type of response")
    content: str = Field(..., description="The content of the response")
    metadata: Metadata = Field(..., description="Metadata about the response")


# Model used for long-term memory
class CompactMemory(BaseModel):
    category: str = Field(
        description="One word category: e.g., 'Technical', 'Personal', 'Preference'"
    )
    fact: str = Field(
        description="A concise, single-sentence summary of the new information. Max 15 words."
    )
    importance: int = Field(
        description="1-10 scale of how important this is to remember long-term."
    )


VECTOR_STORES = {
    "milvus": create_milvus_store,
}

RANKERS = {
    "bge": BGERanker,
}


class RerankedRetriever:
    """
    Custom retriever that wraps a vector store and applies reranking.
    This is used by the Self-RAG graph for retrieval.
    """

    def __init__(
        self,
        vector_store,
        ranker,
        hyde_generator: Optional[HyDEGenerator] = None,
        retrieval_k: int = 30,
        rerank_k: int = 5,
    ):
        self.vector_store = vector_store
        self.ranker = ranker
        self.hyde_generator = hyde_generator
        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k

    def invoke(self, query: str) -> List[Document]:
        """Synchronous retrieval with reranking."""
        # Apply HyDE if configured
        search_query = (
            self.hyde_generator.generate(query) if self.hyde_generator else query
        )

        # Retrieve documents using hybrid search
        vs = self.vector_store.get_vector_store()
        retrieved_docs = vs.similarity_search(search_query, k=self.retrieval_k)

        # Rerank the documents
        reranked_docs = self.ranker.rerank(query, retrieved_docs, self.rerank_k)

        return reranked_docs

    async def ainvoke(self, query: str) -> List[Document]:
        """Async retrieval with reranking."""
        # For now, run sync in executor since most vector stores are sync
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, query)


class SelfRAGAgent:
    """
    RAG Agent using the Self-RAG strategy.

    Self-RAG incorporates self-reflection on:
    - Document relevance (grades retrieved documents)
    - Hallucination detection (checks if generation is grounded)
    - Answer quality (checks if answer addresses the question)
    """

    def __init__(self, config: RAGConfig, checkpointer=None, store=None):
        """
        Initialize SelfRAGAgent synchronously.
        """
        self.config = config

        # RAG Setup #########################################################
        store_factory = VECTOR_STORES.get(config.vector_store_type)
        ranker_factory = RANKERS.get(config.ranker_type)

        if not store_factory or not ranker_factory:
            raise ValueError("Invalid component type in config")

        vector_store = store_factory(config)
        ranker = ranker_factory()
        hyde_generator = HyDEGenerator() if config.hyde else None

        # Create the reranked retriever for Self-RAG
        retriever = RerankedRetriever(
            vector_store=vector_store,
            ranker=ranker,
            hyde_generator=hyde_generator,
            retrieval_k=config.retrieval_k,
            rerank_k=config.rerank_k,
        )

        # LLM Setup #########################################################
        # Base LLM without tools (Self-RAG doesn't use tool calling)
        llm = ChatOllama(model=config.llm_model, temperature=0)

        # Memory Setup #########################################################
        memory_llm = ChatOllama(model=config.memory_llm_model, temperature=0)
        manager_instructions = """
You are a memory manager. Your job is to extract LONG-TERM knowledge from the conversation.

RULES FOR STORAGE:
1. **IGNORE CHIT-CHAT**: Do not store greetings, pleasantries, or temporary context (e.g., "I'm going to lunch now").
2. **Relevance Filter**: Only store information if it is useful for future tasks (e.g., user preferences, project specs, debugging history).
3. **COMPACTNESS**: Do not store raw quotes. Summarize the user's intent into the smallest possible sentence.
4. **No Duplicates**: If a fact already exists in the provided existing memories, do not create a new one.
"""
        self.memory_manager = create_memory_store_manager(
            memory_llm,
            namespace=("memories", "{user_id}"),
            schemas=[CompactMemory],
            instructions=manager_instructions,
        )

        # Training LLM Setup for evaluation ####################################
        training_llm = (
            ChatOllama(model=config.training_llm_model, temperature=0, format="json")
            if config.training_mode
            else None
        )

        # Create Self-RAG workflow #############################################
        workflow = create_self_rag_graph(
            llm=llm,
            retriever=retriever,
            memory_manager=self.memory_manager,
            training_llm=training_llm,
            debug=True,
            grade_documents=config.grade_documents,
        )

        # To be initialized by create()
        self.pool = None
        self.checkpointer = checkpointer
        self.store = store

        # Graph Compilation ##################################################
        self.agent = workflow.compile(checkpointer=checkpointer, store=store)

        self.interrupted_ids = set()

    @classmethod
    async def create(cls, config: RAGConfig):
        """
        Async factory method to create SelfRAGAgent with checkpointing support.
        """
        # Create async pool and checkpointer #####################################
        pool = AsyncConnectionPool(
            conninfo=DB_URI, max_size=20, kwargs={"autocommit": True}, open=False
        )
        await pool.open()

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        # Setup AsyncPostgresStore #############################################
        memory_embeddings = OllamaEmbeddings(model=config.memory_embeddings_model)
        store = AsyncPostgresStore(
            pool,
            index={
                "dims": config.memory_embeddings_dims,
                "embed": memory_embeddings,
                "fields": ["content"],
            },
        )

        await store.setup()

        agent = cls(config, checkpointer=checkpointer, store=store)
        agent.pool = pool

        return agent

    async def close(self):
        """Close the async connection pool."""
        if self.pool:
            await self.pool.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def chat(
        self, query: str, conversation_id: str, user_id: str, stream: bool = True
    ):
        """
        Stream responses from the Self-RAG agent.

        Self-RAG doesn't use tool calls, so the streaming is simpler.
        The graph automatically:
        1. Retrieves documents
        2. Grades them for relevance
        3. Generates an answer
        4. Checks for hallucinations and answer quality

        Yields ChatResponse objects for:
        - AI message content (RESPONSE status)
        - Completion signal (COMPLETE status)
        """
        config = {"configurable": {"thread_id": conversation_id, "user_id": user_id}}
        messages = [HumanMessage(content=query)]

        # Initialize state with new message
        initial_state = {"messages": messages}

        # Track cumulative token usage
        total_input_tokens = 0
        total_output_tokens = 0

        if stream:
            # Stream using messages mode
            async for chunk in self.agent.astream(
                initial_state, config=config, stream_mode="messages"
            ):
                if self.is_interrupted(conversation_id):
                    self.interrupted_ids.remove(conversation_id)
                    yield ChatResponse(
                        status=Status.CANCEL,
                        type="chat.cancel",
                        content="",
                        metadata=Metadata(
                            conversation_id=conversation_id,
                            input_tokens_used=total_input_tokens,
                            output_tokens_used=total_output_tokens,
                            rating=0.0,
                        ),
                    )
                    return

                # chunk is a tuple: (message_chunk, metadata)
                msg, chunk_metadata = chunk

                # Get the current node from metadata
                current_node = (
                    chunk_metadata.get("langgraph_node", "")
                    if isinstance(chunk_metadata, dict)
                    else ""
                )

                # Skip internal nodes (grading, evaluation, etc.)
                skip_nodes = {
                    "evaluator",
                    "grade_documents",
                    "grade_generation",
                    "extract_question",
                    "save_memory",
                    "summarize",
                }
                if current_node in skip_nodes:
                    continue

                # Handle AI message chunks from the generate node
                if isinstance(msg, AIMessage):

                    if isinstance(msg, AIMessageChunk):
                        content = msg.content
                        if content:
                            yield ChatResponse(
                                    status=Status.RESPONSE,
                                    type="response.output_text.delta",
                                    content=content if isinstance(content, str) else "",
                                    metadata=Metadata(
                                        conversation_id=conversation_id,
                                        input_tokens_used=total_input_tokens,
                                        output_tokens_used=total_output_tokens,
                                        rating=0.0,
                                    ),
                                )

                    # Update token usage if available
                    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                        total_input_tokens += msg.usage_metadata.get("input_tokens", 0)
                        total_output_tokens += msg.usage_metadata.get(
                            "output_tokens", 0
                        )

            # Get final state to retrieve the rating from the evaluator
            final_state = await self.agent.aget_state(config)
            rating = final_state.values.get("rating", 0.0)

            yield ChatResponse(
                status=Status.COMPLETE,
                type="completion",
                content="",
                metadata=Metadata(
                    conversation_id=conversation_id,
                    input_tokens_used=total_input_tokens,
                    output_tokens_used=total_output_tokens,
                    rating=rating,
                ),
            )

        else:
            final_state = await self.agent.ainvoke(initial_state, config=config)

            # Yield the final state as a single response
            yield final_state
            return

    def call(self, query: Union[str, List[str]]):
        """
        Synchronous call method for single or batch queries using Self-RAG.

        Args:
            query: Single query string or list of query strings

        Returns:
            Single result or list of results.
            Each result is a tuple: (response_text, list_of_retrieved_docs)
        """
        is_batch = isinstance(query, list)
        queries = query if is_batch else [query]

        # Prepare inputs for the graph
        inputs = [{"messages": [HumanMessage(content=q)]} for q in queries]

        try:
            # Execute via graph
            if is_batch:
                results = self.agent.batch(inputs)
            else:
                results = [self.agent.invoke(inputs[0])]

            final_results = []
            for state in results:
                messages = state.get("messages", [])
                documents = state.get("documents", [])
                generation = state.get("generation", "")

                # Extract final response from generation or last AI message
                response_text = generation
                if not response_text:
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage):
                            content = msg.content
                            if isinstance(content, str):
                                response_text = content
                            elif isinstance(content, list):
                                # Handle list content (e.g., multimodal messages)
                                response_text = " ".join(
                                    str(item) for item in content if item
                                )
                            else:
                                response_text = str(content) if content else ""
                            break

                # Extract retrieved docs from state
                retrieved_docs = [doc.page_content for doc in documents]

                final_results.append((response_text, retrieved_docs))

            if is_batch:
                return final_results
            else:
                return final_results[0]

        except Exception as e:
            print(f"Error in agent_call: {e}")
            if is_batch:
                return [(f"Error: {str(e)}", [])] * len(queries)
            return (f"Error: {str(e)}", [])

    def interrupt(self, conversation_id):
        self.interrupted_ids.add(conversation_id)
        return True

    def is_interrupted(self, conversation_id):
        return conversation_id in self.interrupted_ids

    async def get_full_history(self, conversation_id: str):
        config = {"configurable": {"thread_id": conversation_id}}
        state = await self.agent.aget_state(config=config)
        messages = state.values.get("messages", [])

        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                history.append({"role": "system", "content": msg.content})
            elif isinstance(msg, BaseMessage):
                history.append({"role": "unknown", "content": msg.content})

        return history

    async def clear_session(self, conversation_id: str) -> bool:
        if self.checkpointer is not None:
            await self.checkpointer.adelete_thread(conversation_id)
            return True
        return False

    async def get_retrieval_info(self, conversation_id: str) -> dict:
        """
        Get information about the last retrieval for debugging/inspection.

        Returns:
            dict with keys:
            - question: The processed question
            - documents: List of retrieved documents
            - generation: The generated answer
        """
        config = {"configurable": {"thread_id": conversation_id}}
        state = await self.agent.aget_state(config=config)

        return {
            "question": state.values.get("question", ""),
            "documents": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in state.values.get("documents", [])
            ],
            "generation": state.values.get("generation", "")
        }


async def main():
    config = RAGConfig()

    # Use the async factory method to create agent with checkpointing
    agent = await SelfRAGAgent.create(config)

    query = input("Enter your query: ")

    # Stream responses
    async for response in agent.chat(
        query, conversation_id=str(29), user_id="1", stream=True
    ):
        print(f"[{response.status.value}] {response.type}: {response.content}")
        if response.status == Status.COMPLETE:
            print(
                f"\nFinal token usage - Input: {response.metadata.input_tokens_used}, Output: {response.metadata.output_tokens_used}"
            )
            print(f"Rating: {response.metadata.rating}")

    # Optionally get retrieval info for debugging
    retrieval_info = await agent.get_retrieval_info(str(29))
    print(f"\nRetrieval Info:")
    print(f"  Question: {retrieval_info['question']}")
    print(f"  Documents retrieved: {len(retrieval_info['documents'])}")

    # Clean up
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
