import os
import asyncio
from typing import Union, List, Optional
from enum import Enum

# --- LangChain / LangGraph Imports ---
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig

# --- Database / Store Imports ---
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

# --- Memory Imports ---
# NOTE: langmem has renamed create_memory_store_manager to create_memory_store_enricher.
# For now, we'll disable memory management until we can refactor the code.
# from langmem import create_memory_manager

# --- Local Imports ---
from .config import RAGConfig
from .milvus import create_milvus_store
from .modules.reranker import BGERanker
from .modules.hyde import HyDEGenerator
from .graph import create_agent_graph

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
DB_URI: str = os.getenv("PG_URI") or ""

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
   rating: float = Field(..., description="Rating result")
   title: Optional[str] = Field(None, description="Title of the conversation")
   input_tokens_used: int = Field(..., description="Number of input tokens used")
   output_tokens_used: int = Field(..., description="Number of output tokens used")

class ChatResponse(BaseModel):
   status: Status
   type: str = Field(..., description="The type of response")
   content: str = Field(..., description="The content of the response")
   metadata: Metadata = Field(..., description="Metadata about the response")

# Model used for long-term memory
class CompactMemory(BaseModel):
    category: str = Field(description="One word category: e.g., 'Technical', 'Personal', 'Preference'")
    fact: str = Field(description="A concise, single-sentence summary of the new information. Max 15 words.")
    importance: int = Field(description="1-10 scale of how important this is to remember long-term.")

# Register available stores and rankers
VECTOR_STORES = {
    "milvus": create_milvus_store,
}

RANKERS = {
    "bge": BGERanker,
}

class RAGAgent:
    def __init__(self, config: RAGConfig, checkpointer=None, store=None):
        """
        Initialize RAGAgent synchronously.
        """
        self.config = config

        # RAG Setup
        store_factory = VECTOR_STORES.get(config.vector_store_type)
        ranker_factory = RANKERS.get(config.ranker_type)

        if not store_factory:
            raise ValueError("Invalid vector store type in config")

        # Initialize actual Milvus vector store
        vector_store = store_factory(config)
        ranker = ranker_factory() if ranker_factory else None
        hyde_generator = HyDEGenerator(model=config.llm_model) if config.hyde else None

        rag_tool = create_rag_tool(vector_store, ranker, hyde_generator, config.similarity_threshold)

        # LLM Setup and tool schema binding
        if not config.llm_model:
            raise ValueError("LLM model not specified")
        llm = ChatOllama(
            model=config.llm_model,
            temperature=0.4,
            num_ctx=8192,
            streaming=True,
            reasoning=True,
            num_predict=config.llm_num_predict,
        )
        llm_with_tools = llm.bind_tools([rag_tool])

        # Memory Setup
        memory_manager = None

        # Summarization Setup
        sum_model = config.summarization_model if (config.summarization_model and config.summarization_model != "qwen3:8b") else config.llm_model
        summarization_llm = ChatOllama(model=sum_model, temperature=0, num_ctx=8192, reasoning=False, num_predict=1024)

        # Training LLM Setup for evaluation - Disabled to optimize speed
        training_llm = None

        # Title LLM Setup
        title_model = config.title_llm_model if (config.title_llm_model and config.title_llm_model != "qwen3:8b") else config.llm_model
        title_llm = ChatOllama(model=title_model, temperature=0, num_ctx=4096, reasoning=False, num_predict=100)
        self.title_llm = title_llm

        # Exa AI Search setup
        exa_tool = None
        if os.getenv("EXA_API_KEY"):
            try:
                from langchain_exa import ExaSearchResults
                exa_tool = ExaSearchResults(num_results=3)
            except Exception as e:
                print(f"Warning: Failed to initialize Exa Search: {e}")

        # Constucting actual agent
        workflow = create_agent_graph(
            llm=llm,
            llm_with_tools=llm_with_tools,
            rag_tool=rag_tool,
            exa_tool=exa_tool,
            memory_manager=memory_manager,
            summarization_llm=summarization_llm,
            training_llm=training_llm,
            title_llm=title_llm,
            max_context_tokens=config.max_context_tokens,
            debug=True
        )

        # To be initialized by create()
        self.pool = None
        self.checkpointer = checkpointer
        self.store = store
        self.vector_store = vector_store

        # Graph Compilation
        self.agent = workflow.compile(
            checkpointer=checkpointer,
            store=store
        )

        self.interrupted_ids = set()

    @classmethod
    async def create(cls, config: RAGConfig):
        """
        Async factory method to create RAGAgent with checkpointing support.
        """
        # Create async pool and checkpointer
        pool = AsyncConnectionPool(conninfo=DB_URI, max_size=5, kwargs={"autocommit": True}, open=False)
        await pool.open()

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        # Setup AsyncPostgresStore (index removed because pgvector is not available and memory is disabled)
        store = AsyncPostgresStore(pool)

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
        self,
        query: str,
        conversation_id: str,
        user_id: str,
        stream: bool = True,
        enable_exa: bool = False
    ):
        """
        Stream responses from the agent.

        Yields ChatResponse objects for:
        - AI message content (RESPONSE status, type: response.output_text.delta)
        - Tool calls (RESPONSE status, type: response.function_call_arguments.delta)
        - Tool results (FUNCTION status, type: function)
        - Completion signal (COMPLETE status)
        """
        if self.config.simple_rag:
            async for chunk in self.simple_rag_flow(query, conversation_id, user_id):
                yield chunk
            return

        config = {"configurable": {"thread_id": conversation_id, "user_id": user_id, "enable_exa": enable_exa}}

        # Pass the clean query directly to the conversation history to avoid polluting history with instructions.
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }

        # Track cumulative token usage
        total_input_tokens = 0
        total_output_tokens = 0

        if stream:
            # We use astream_events to capture real-time streaming tokens and tool calls from inside graph nodes
            async for event in self.agent.astream_events(initial_state, config=config, version="v2"):

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
                            title=None,
                        )
                    )
                    return

                event_type = event["event"]

                if event_type == "on_chat_model_stream":
                    # Only stream responses from the synthesizer node to avoid leaking helper outputs (e.g. evaluator, title generation)
                    node_name = event.get("metadata", {}).get("langgraph_node")
                    if node_name != "synthesizer":
                        continue

                    chunk = event["data"].get("chunk")
                    if chunk:
                        # Extract reasoning if present
                        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
                        if reasoning:
                            yield ChatResponse(
                                status=Status.RESPONSE,
                                type="response.reasoning.delta",
                                content=reasoning,
                                metadata=Metadata(
                                    conversation_id=conversation_id,
                                    input_tokens_used=total_input_tokens,
                                    output_tokens_used=total_output_tokens,
                                    rating=0.0,
                                    title=None,
                                )
                            )

                        if hasattr(chunk, "content") and chunk.content:
                            content = chunk.content
                            yield ChatResponse(
                                status=Status.RESPONSE,
                                type="response.output_text.delta",
                                content=content,
                                metadata=Metadata(
                                    conversation_id=conversation_id,
                                    input_tokens_used=total_input_tokens,
                                    output_tokens_used=total_output_tokens,
                                    rating=0.0,
                                    title=None,
                                )
                            )

                        # Accumulate tokens if usage metadata exists in chunk
                        if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                            total_input_tokens += chunk.usage_metadata.get('input_tokens', 0)
                            total_output_tokens += chunk.usage_metadata.get('output_tokens', 0)


                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "tool")
                    tool_input = event["data"].get("input", "")
                    yield ChatResponse(
                        status=Status.RESPONSE,
                        type="response.function_call_arguments.delta",
                        content=f"Calling {tool_name} with args: {tool_input}",
                        metadata=Metadata(
                            conversation_id=conversation_id,
                            input_tokens_used=total_input_tokens,
                            output_tokens_used=total_output_tokens,
                            rating=0.0,
                            title=None,
                        )
                    )

                elif event_type == "on_tool_end":
                    tool_output = event["data"].get("output", "")
                    # Extract string representation of the output
                    content_str = ""
                    if isinstance(tool_output, str):
                        content_str = tool_output
                    elif hasattr(tool_output, "content"):
                        content_str = str(tool_output.content)
                    else:
                        content_str = str(tool_output)

                    yield ChatResponse(
                        status=Status.FUNCTION,
                        type=event.get("name", "function"),
                        content=content_str,
                        metadata=Metadata(
                            conversation_id=conversation_id,
                            input_tokens_used=total_input_tokens,
                            output_tokens_used=total_output_tokens,
                            rating=0.0,
                            title=None,
                        )
                    )

            # Retrieve final state to get ratings, generated titles, and accurate token usage
            final_state = await self.agent.aget_state(config)
            rating = final_state.values.get("rating", 0.0)
            title = final_state.values.get("title", None)
            total_input_tokens = final_state.values.get("input_tokens_used", total_input_tokens)
            total_output_tokens = final_state.values.get("output_tokens_used", total_output_tokens)

            completion_metadata = Metadata(
                conversation_id=conversation_id,
                input_tokens_used=total_input_tokens,
                output_tokens_used=total_output_tokens,
                rating=rating,
                title=title,
            )

            yield ChatResponse(
                status=Status.COMPLETE,
                type="completion",
                content="",
                metadata=completion_metadata
            )

        else:
            final_state = await self.agent.ainvoke(initial_state, config=config)

            # Yield the final state as a single response
            yield final_state
            return

    async def simple_rag_flow(self, query: str, conversation_id: str, user_id: str):
        """
        Stream responses from the agent using a simple direct RAG pipeline:
        1. Retrieve context once from Milvus.
        2. Format a system instruction context.
        3. Stream answers from Ollama.
        """
        # Yield Tool Start
        yield ChatResponse(
            status=Status.RESPONSE,
            type="response.function_call_arguments.delta",
            content=f"Calling hybrid_RAG_retrieve with args: {{'query': '{query}'}}",
            metadata=Metadata(
                conversation_id=conversation_id,
                input_tokens_used=0,
                output_tokens_used=0,
                rating=0.0,
                title=None,
            )
        )

        retrieved_context = ""
        if hasattr(self, "vector_store") and self.vector_store:
            try:
                # Retrieve from Milvus using search_documents, filtering by conversation_id or global collection
                expr = None
                if self.vector_store.vector_store.collection_name == "ingestion_staging":
                    expr = f"target_collection == '{conversation_id}' or target_collection == 'HeaderInContentTrial'"
                docs = self.vector_store.search_documents(query, k=3, expr=expr)
                if docs:
                    retrieved_context = "\n\n".join(
                        f"Source: {doc.metadata}\nContent: {doc.page_content}"
                        for doc in docs
                    )
            except Exception as e:
                print(f"Simple RAG retrieval error: {e}")

        tool_output = retrieved_context if retrieved_context else "No relevant documents found in the knowledge base."

        # Yield Tool End
        yield ChatResponse(
            status=Status.FUNCTION,
            type="function",
            content=tool_output,
            metadata=Metadata(
                conversation_id=conversation_id,
                input_tokens_used=0,
                output_tokens_used=0,
                rating=0.0,
                title=None,
            )
        )

        # Load existing history messages from checkpointer to maintain context
        history_messages = []
        try:
            config = {"configurable": {"thread_id": conversation_id, "user_id": user_id}}
            state = await self.agent.aget_state(config=config)
            if state and "messages" in state.values:
                # Filter out any old SystemMessages to avoid duplicate system instructions in the context
                history_messages = [msg for msg in state.values["messages"] if not isinstance(msg, SystemMessage)]
        except Exception as e:
            print(f"Error loading history for simple RAG context: {e}")

        # Format system instruction and context
        system_prompt = f"""You are a helpful assistant. Use the retrieved context below as your primary reference to answer the user's question.
If the retrieved context does not contain relevant information to answer the question, you may answer using your general knowledge directly.

Retrieved Context:
{tool_output}"""

        # Construct messages
        messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=query)]

        # Call the LLM (ChatOllama) to stream response
        llm = ChatOllama(
            model=self.config.llm_model,
            temperature=0.4,
            num_ctx=8192,
            streaming=True,
            reasoning=True,
            num_predict=self.config.llm_num_predict,
        )

        total_input_tokens = 0
        total_output_tokens = 0
        full_ai_response = ""

        async for chunk in llm.astream(messages):
            reasoning = chunk.additional_kwargs.get("reasoning_content", "")
            if reasoning:
                yield ChatResponse(
                    status=Status.RESPONSE,
                    type="response.reasoning.delta",
                    content=reasoning,
                    metadata=Metadata(
                        conversation_id=conversation_id,
                        input_tokens_used=total_input_tokens,
                        output_tokens_used=total_output_tokens,
                        rating=0.0,
                        title=None,
                    )
                )

            content = chunk.content
            if content:
                full_ai_response += content
                yield ChatResponse(
                    status=Status.RESPONSE,
                    type="response.output_text.delta",
                    content=content,
                    metadata=Metadata(
                        conversation_id=conversation_id,
                        input_tokens_used=total_input_tokens,
                        output_tokens_used=total_output_tokens,
                        rating=0.0,
                        title=None,
                    )
                )

        # Persist messages to LangGraph checkpointer so history is correctly displayed
        try:
            config = {"configurable": {"thread_id": conversation_id, "user_id": user_id}}
            await self.agent.aupdate_state(
                config,
                {
                    "messages": [
                        HumanMessage(content=query),
                        ToolMessage(content=tool_output, name="hybrid_RAG_retrieve", tool_call_id="call_simple_rag"),
                        AIMessage(content=full_ai_response)
                    ]
                },
                as_node="synthesizer"
            )
        except Exception as save_err:
            print(f"Error saving simple RAG conversation history: {save_err}")

        # Complete turn
        generated_title = None
        if hasattr(self, "title_llm") and self.title_llm:
            try:
                title_prompt = f"Generate a distinct, 3-5 word title for a conversation starting with this user message. Return ONLY the title itself with no quotes, preamble or extra text: '{query}'"
                title_res = await self.title_llm.ainvoke(title_prompt)
                generated_title = title_res.content.strip().strip('"').strip("'").strip()
            except Exception as e:
                print(f"Error generating title in simple RAG: {e}")

        yield ChatResponse(
            status=Status.COMPLETE,
            type="completion",
            content="",
            metadata=Metadata(
                conversation_id=conversation_id,
                input_tokens_used=total_input_tokens,
                output_tokens_used=total_output_tokens,
                rating=0.0,
                title=generated_title,
            )
        )

    def call(self, query: Union[str, List[str]]):
        """
        Synchronous call method for single or batch queries using the LangGraph agent.

        Args:
            query: Single query string or list of query strings

        Returns:
            Single result or list of results.
            Each result is a tuple: (response_text, list_of_retrieved_docs)
        """
        is_batch = isinstance(query, list)
        queries = query if is_batch else [query]

        # Prepare inputs for the graph
        # We use HumanMessage to represent the user query
        inputs = [{"messages": [HumanMessage(content=q)]} for q in queries]

        try:
            # Execute via graph
            # We use invoke/batch which runs the graph to completion
            if is_batch:
                results = self.agent.batch(inputs)
            else:
                results = [self.agent.invoke(inputs[0])]

            final_results = []
            for state in results:
                messages = state["messages"]

                # Extract final response
                final_message = messages[-1]
                response_text = final_message.content if isinstance(final_message, AIMessage) else ""

                # Extract retrieved docs from ToolMessages
                retrieved_docs = []
                for msg in messages:
                    if isinstance(msg, ToolMessage) and msg.artifact:
                        # We expect the artifact to be the list of documents
                        if isinstance(msg.artifact, list):
                            retrieved_docs.extend([doc.page_content for doc in msg.artifact])

                final_results.append((response_text, retrieved_docs))

            if is_batch:
                return final_results
            else:
                return final_results[0]

        except Exception as e:
            print(f"Error in agent_call: {e}")
            if is_batch:
                return [], [f"Error: {str(e)}"] * len(queries)
            return [], f"Error: {str(e)}"

    def interrupt(self, conversation_id):
        self.interrupted_ids.add(conversation_id)
        return True

    def is_interrupted(self, conversation_id):
        return conversation_id in self.interrupted_ids

    async def get_full_history(self, conversation_id: str):
        try:
            config = {"configurable": {"thread_id": conversation_id}}
            state = await self.agent.aget_state(config=config)
            messages = state.values["messages"]
        except Exception:
            return []

        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "type": "ai", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                history.append({"role": "tool", "type": "tool", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                history.append({"role": "system", "type": "system", "content": msg.content})
            elif isinstance(msg, BaseMessage):
                history.append({"role": "Unknown", "type": "unknown", "content": msg.content})

        return history

    async def clear_session(self, conversation_id: str) -> bool:
        try:
            config = {"configurable": {"thread_id": conversation_id}}
            state = await self.agent.aget_state(config=config)
            if state and "messages" in state.values:
                from langchain_core.messages import RemoveMessage
                # Create a RemoveMessage for every message in history to clear it in the checkpointer
                removals = [RemoveMessage(id=msg.id) for msg in state.values["messages"] if msg.id]
                if removals:
                    await self.agent.aupdate_state(config=config, values={"messages": removals})

            # Also try direct checkpointer delete method if it exists
            if hasattr(self.checkpointer, "adelete_thread"):
                await self.checkpointer.adelete_thread(conversation_id)
        except Exception as e:
            print(f"Error clearing session: {e}")
        return True

    async def get_state_history(self, conversation_id: str):
        config = {"configurable": {"thread_id": conversation_id}}
        history = []
        async for state in self.agent.aget_state_history(config=config):
            history.append(state)
        return history




def create_rag_tool(vector_store, ranker, hyde_generator: Optional[HyDEGenerator], similarity_threshold: float = 0.0):
    @tool
    def hybrid_RAG_retrieve(query: str, config: Optional[RunnableConfig] = None):
        """
        Retrieve relevant context from the knowledge base using hybrid search (dense + sparse) and reranking.

        Uses BM25 sparse search and dense vector search simultaneously, then reranks with BGE for best relevance.
        Falls back to a second pass with expanded query terms if no results are found.

        Args:
            query: The search query to find relevant information (raw user intent)

        Returns:
            Serialized string with source metadata and content of top-ranked documents
        """
        if "Please use the hybrid_RAG_retrieve tool to answer if needed. If retrieval yields no relevant results, DO NOT hallucinate. " in query:
            query = query.replace("Please use the hybrid_RAG_retrieve tool to answer if needed. If retrieval yields no relevant results, DO NOT hallucinate. ", "", 1)

        if not query.strip():
            return "No query provided."

        # 1. Expand query: use HYDE-generated passage for retrieval if available
        # This helps when the user's query is too short or uses different terminology than the docs
        retrieval_queries = [query]
        if hyde_generator:
            hyde_passage = hyde_generator.generate(query)
            if hyde_passage and len(hyde_passage) > 20:
                retrieval_queries.append(hyde_passage)

        if not vector_store:
            return "Source: {'doc_id': 'mock1'}\nContent: This is a mock document retrieved for testing purposes. It states that Milvus is currently bypassed, but the retrieval tool was successfully invoked!"

        vs = vector_store

        configurable = config.get("configurable", {}) if config else {}
        conversation_id = configurable.get("thread_id", "")

        expr = None
        if vs.vector_store.collection_name == "ingestion_staging" and conversation_id:
            expr = f"target_collection == '{conversation_id}' or target_collection == 'HeaderInContentTrial'"

        all_retrieved = {}

        # 2. Hybrid search: use search_documents which invokes BOTH dense and sparse search
        # similarity_search() only uses dense vectors — that's the bug we're fixing
        for q in retrieval_queries:
            try:
                docs = vs.search_documents(q, k=30, expr=expr)
                for doc in docs:
                    doc_hash = hash(doc.page_content[:128])
                    if doc_hash not in all_retrieved:
                        all_retrieved[doc_hash] = doc
            except Exception:
                # If search_documents fails, fall back to dense vector search only
                docs = vs.similarity_search(q, k=30, expr=expr)
                for doc in docs:
                    doc_hash = hash(doc.page_content[:128])
                    if doc_hash not in all_retrieved:
                        all_retrieved[doc_hash] = doc

        retrieved_docs = list(all_retrieved.values())

        if not retrieved_docs:
            return "No relevant documents found in the knowledge base."

        # 3. Rerank for precision if a ranker is available
        k = 3
        if ranker:
            reranked_docs = ranker.rerank(query, retrieved_docs, k)
            if similarity_threshold > 0.0:
                reranked_docs = [doc for doc in reranked_docs if doc.metadata.get("rerank_score", 0.0) >= similarity_threshold]
        else:
            reranked_docs = retrieved_docs[:k]

        if not reranked_docs:
            return "No relevant documents found in the knowledge base."

        # 4. Serialize
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in reranked_docs
        )
        return serialized

    return hybrid_RAG_retrieve

async def main():
    config = RAGConfig()

    # Use the async factory method to create agent with checkpointing
    agent = await RAGAgent.create(config)

    query = input("Enter your query: ")

    # Invoke the agent
    # async for response in agent.chat(query, conversation_id=str(29), stream=False):
    #     print("Messages:")
    #     print(response["messages"])
    #     print()

    #     # Context might not exist if summarization hasn't triggered yet
    #     if "context" in response and "running_summary" in response["context"]:
    #         print("Summary:")
    #         print(response["context"]["running_summary"].summary)
    #         print()

    #     print(f"Input tokens: {response.get('input_tokens_used', 0)}")
    #     print(f"Output tokens: {response.get('output_tokens_used', 0)}")

    # Stream responses
    async for response in agent.chat(query, conversation_id=str(777), user_id="1", stream=True):
        print(f"[{response.status.value}] {response.type}: {response.content}")
        if response.status == Status.COMPLETE:
            print(f"\nFinal token usage - Input: {response.metadata.input_tokens_used}, Output: {response.metadata.output_tokens_used}")

    # Clean up
    await agent.close()

if __name__ == "__main__":
    asyncio.run(main())
