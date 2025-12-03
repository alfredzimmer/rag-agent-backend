from typing import Union, List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from psycopg_pool import ConnectionPool, AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import uuid
from enum import Enum
from pydantic import BaseModel, Field
import asyncio
import os

from .milvus import create_milvus_store
from .modules.reranker import BGERanker
from .modules.hyde import HyDEGenerator
from .graph import create_agent_graph
from .session_storage import get_storage

from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("DB_URI")

@dataclass
class RAGConfig:
    vector_store_type: str = "milvus"  
    ranker_type: str = "bge"
    dense_embedding_model: str = "qwen3-embedding:8b"
    sparse_embedding_model: str = "splade" #[splade, bm25, bge]
    llm_model: str = "qwen3:8b"
    hyde: bool = False

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
   input_tokens_used: int = Field(..., description="Number of input tokens used")
   output_tokens_used: int = Field(..., description="Number of output tokens used")

class ChatResponse(BaseModel):
   status: Status
   type: str = Field(..., description="The type of response")
   content: str = Field(..., description="The content of the response")
   metadata: Metadata = Field(..., description="Metadata about the response")


# Registry of component creators
VECTOR_STORES = {
    "milvus": create_milvus_store,
}

RANKERS = {
    "bge": BGERanker,
}

class RAGAgent:
    def __init__(self, config: RAGConfig, checkpointer=None):
        """
        Initialize RAGAgent synchronously.
        
        Args:
            config: RAG configuration
            checkpointer: Optional checkpointer (for internal use by create())
        """
        self.config = config
        store_factory = VECTOR_STORES.get(config.vector_store_type)
        ranker_factory = RANKERS.get(config.ranker_type)
        
        if not store_factory or not ranker_factory:
            raise ValueError("Invalid component type in config")
            
        vector_store = store_factory(config)
        ranker = ranker_factory()
        if(config.hyde):
            hyde_generator = HyDEGenerator()
        else:
            hyde_generator = None

        # Create the tool with the specific components
        rag_tool = create_rag_tool(vector_store, ranker, hyde_generator)
        
        llm = ChatOllama(model=config.llm_model, temperature=0)
        llm_with_tools = llm.bind_tools([rag_tool])

        workflow = create_agent_graph(llm_with_tools, rag_tool)
        
        # Compile agent with or without checkpointer
        self.pool = None
        self.checkpointer = checkpointer
        self.agent = workflow.compile(checkpointer=checkpointer)

        self.interrupted_ids = set()

    @classmethod
    async def create(cls, config: RAGConfig):
        """
        Async factory method to create RAGAgent with checkpointing support.
        
        Usage:
            agent = await RAGAgent.create(config)
            async for response in agent.chat(query, conversation_id):
                ...
            await agent.close()
        
        Args:
            config: RAG configuration
            
        Returns:
            Initialized RAGAgent with async checkpointer
        """
        # Create async pool and checkpointer
        pool = AsyncConnectionPool(conninfo=DB_URI, max_size=20, kwargs={"autocommit": True}, open=False)
        await pool.open()
        
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        
        # Create agent instance with checkpointer
        agent = cls(config, checkpointer=checkpointer)
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
        stream: bool = True
    ):
        """
        Stream responses from the agent using stream_mode="messages".
        
        Yields ChatResponse objects for:
        - AI message content (RESPONSE status, type: response.output_text.delta)
        - Tool calls (RESPONSE status, type: response.function_call_arguments.done)
        - Tool results (FUNCTION status, type: function)
        - Completion signal (COMPLETE status)
        """
        config = {"configurable": {"thread_id": conversation_id}}
        messages = [HumanMessage(content=query)]
        
        # Initialize state with new message only
        # Token counts will be maintained by the checkpointer across conversation
        initial_state = {
            "messages": messages
        }
        
        # Track cumulative token usage
        total_input_tokens = 0
        total_output_tokens = 0
        if stream:
            # Stream using messages mode - this streams LLM tokens as they're generated
            async for chunk in self.agent.astream(initial_state, config=config, stream_mode="messages"):

                if self.is_interrupted(conversation_id):
                    self.interrupted_ids.remove(conversation_id)
                    yield ChatResponse(
                        status=Status.CANCEL,
                        type="chat.cancel", 
                        content="", 
                        metadata=Metadata(
                            conversation_id=conversation_id,
                            input_tokens_used=total_input_tokens,
                            output_tokens_used=total_output_tokens)
                    )
                    return
                
                # chunk is a tuple: (message_chunk, metadata)
                # message_chunk contains individual tokens from the LLM
                msg, metadata = chunk
                
                # Handle AI message chunks (tokens from LLM)
                if isinstance(msg, AIMessage):
                    # Check for tool calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            yield ChatResponse(
                                status=Status.RESPONSE,
                                type="response.function_call_arguments.delta",
                                content=f"Calling {tool_call['name']} with args: {tool_call['args']}",
                                metadata=Metadata(
                                    conversation_id=conversation_id,
                                    input_tokens_used=total_input_tokens,
                                    output_tokens_used=total_output_tokens
                                )
                            )
                    
                    # Stream AI message content chunks (individual tokens)
                    if msg.content:
                        yield ChatResponse(
                            status=Status.RESPONSE,
                            type="response.output_text.delta",
                            content=msg.content,
                            metadata=Metadata(
                                conversation_id=conversation_id,
                                input_tokens_used=total_input_tokens,
                                output_tokens_used=total_output_tokens
                            )
                        )
                    
                    # Update token usage if available
                    if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                        total_input_tokens += msg.usage_metadata.get('input_tokens', 0)
                        total_output_tokens += msg.usage_metadata.get('output_tokens', 0)
                
                # Handle tool messages
                elif isinstance(msg, ToolMessage):
                    yield ChatResponse(
                        status=Status.FUNCTION,
                        type="function",
                        content=msg.content,
                        metadata=Metadata(
                            conversation_id=conversation_id,
                            input_tokens_used=total_input_tokens,
                            output_tokens_used=total_output_tokens
                        )
                    )
            
            # Yield completion signal
            yield ChatResponse(
                status=Status.COMPLETE,
                type="completion",
                content="",
                metadata=Metadata(
                    conversation_id=conversation_id,
                    input_tokens_used=total_input_tokens,
                    output_tokens_used=total_output_tokens
                )
            )
        else:
            final_state = await self.agent.ainvoke(initial_state, config=config)
            
            # Yield the final state as a single response
            yield final_state
            return

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
        config = {"configurable": {"thread_id": conversation_id}}
        state = await self.agent.aget_state(config=config)
        messages = state.values["messages"]

        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                history.append({"role": "tool", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                history.append({"role": "system", "content": msg.content})
            elif isinstance(msg, BaseMessage):
                history.append({"role": "Unknown", "content": msg.content})
    
        return history
    
    async def clear_session(self, conversation_id: str) -> bool:
        await self.checkpointer.adelete_thread(conversation_id)
        return True




def create_rag_tool(vector_store, ranker, hyde_generator: Optional[HyDEGenerator]):
    @tool
    def hybrid_RAG_retrieve(query: str):
        """
        Retrieve relevant context from the knowledge base using hybrid search and reranking.
        
        Args:
            query: The search query to find relevant information
            
        Returns:
            Tuple of (Serialized context, List of documents)
        """
        # Retrieve documents using Milvus hybrid search (dense + sparse)
        inputText = hyde_generator.generate(query) if hyde_generator else query
        
        vs = vector_store.get_vector_store()
        retrieved_docs = vs.similarity_search(inputText, k=30)

        # Rerank the retrieved documents
        # The ranker is a BGERanker instance
        k = 3
        reranked_docs = ranker.rerank(query, retrieved_docs, k)
        
        # Generate serialized output
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
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
    async for response in agent.chat(query, conversation_id=str(28), stream=False):
        print("Messages:")
        print(response["messages"])
        print()
        
        # Context might not exist if summarization hasn't triggered yet
        if "context" in response and "running_summary" in response["context"]:
            print("Summary:")
            print(response["context"]["running_summary"].summary)
            print()
        
        print(f"Input tokens: {response.get('input_tokens_used', 0)}")
        print(f"Output tokens: {response.get('output_tokens_used', 0)}")
    
    # Stream responses
    # async for response in agent.chat(query, conversation_id=str(20)):
    #     print(f"[{response.status.value}] {response.type}: {response.content}")
    #     if response.status == Status.COMPLETE:
    #         print(f"\nFinal token usage - Input: {response.metadata.input_tokens_used}, Output: {response.metadata.output_tokens_used}")
    
    # Clean up
    await agent.close()

if __name__ == "__main__":
    asyncio.run(main())
    