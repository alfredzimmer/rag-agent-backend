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

DB_URI = "postgresql://agent_master:*181nosdunK@localhost:5432/agent_data"

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

        self.interrupted = False

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
        pool = AsyncConnectionPool(conninfo=DB_URI, max_size=20, kwargs={"autocommit": True})
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
        conversation_id: str
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
        
        # Initialize state with token counts
        initial_state = {
            "messages": messages,
            "input_tokens_used": 0,
            "output_tokens_used": 0
        }
        
        # Track cumulative token usage
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Stream using messages mode - this streams LLM tokens as they're generated
        async for chunk in self.agent.astream(initial_state, config=config, stream_mode="messages"):

            if self.interrupted:
                self.interrupted = False
                break
            
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

    def interrupt(self):
        self.interrupted = True

            


# def call(self, query: Union[str, List[str]]):
#         # System message
#         SYSTEM_MESSAGE = """You are a helpful assistant with access to a specialized knowledge base. 
#         IMPORTANT: You MUST ALWAYS use the hybrid_RAG_retrieve tool FIRST before answering any question. 
#         Never rely solely on your general knowledge. Always check the knowledge base for relevant information."""

#         # Normalize input to list
#         is_batch = isinstance(query, list)
#         queries = query if is_batch else [query]

#         # Initialize chat histories for each query
#         all_messages = []
#         for q in queries:
#             all_messages.append([
#                 {"role": "system", "content": SYSTEM_MESSAGE},
#                 HumanMessage(content=q)
#             ])
        
#         # Store retrieved docs for each query
#         batch_retrieved_docs = [[] for _ in queries]
#         final_responses = [""] * len(queries)
        
#         try:
#             # Step 1: Initial batch call to LLM
#             responses = self.llm_with_tools.batch(all_messages)
            
#             # Track which indices need a second pass (tool execution)
#             indices_to_process = []
            
#             # Process initial responses
#             for i, response in enumerate(responses):
#                 if response.tool_calls:
#                     # Add the assistant's response with tool calls
#                     all_messages[i].append(response)
#                     indices_to_process.append(i)
#                 else:
#                     # No tool call, just get the response
#                     final_responses[i] = response.content

#             # Step 2: Execute tools for those that need it
#             if indices_to_process:
#                 # Collect all tool calls that need execution
#                 tool_tasks = []
#                 for i in indices_to_process:
#                     response = responses[i]
#                     for tool_call in response.tool_calls:
#                         tool_tasks.append((i, tool_call))
                
#                 # Execute tool calls in parallel
#                 with ThreadPoolExecutor() as executor:
#                     # Submit all tasks
#                     future_to_task = {
#                         executor.submit(self.rag_tool.invoke, task[1]["args"]): task 
#                         for task in tool_tasks
#                     }
                    
#                     # Process results as they complete
#                     for future in as_completed(future_to_task):
#                         i, tool_call = future_to_task[future]
#                         try:
#                             serialized_context, docs = future.result()
                            
#                             # Store docs
#                             batch_retrieved_docs[i].extend(docs)
                            
#                             # Add tool result to messages
#                             all_messages[i].append(ToolMessage(
#                                 content=serialized_context,
#                                 tool_call_id=tool_call["id"]
#                             ))
#                         except Exception as exc:
#                             print(f"Tool execution failed: {exc}")
#                             # Add error message to tool result so the LLM knows it failed
#                             all_messages[i].append(ToolMessage(
#                                 content=f"Error: {str(exc)}",
#                                 tool_call_id=tool_call["id"]
#                             ))

#                 # Prepare second batch pass
#                 second_pass_indices = indices_to_process
                
#                 # Step 3: Second batch call to LLM for those that used tools
#                 if second_pass_indices:
#                     second_pass_messages = [all_messages[i] for i in second_pass_indices]
#                     second_responses = self.llm_with_tools.batch(second_pass_messages)
                    
#                     for idx, response in zip(second_pass_indices, second_responses):
#                         final_responses[idx] = response.content
            
#         except Exception as e:
#             print(f"Error in agent_call: {e}")
#             if is_batch:
#                 return [], [f"Error: {str(e)}"] * len(queries)
#             return [], f"Error: {str(e)}"

#         # Return results formatted correctly
#         results = []
#         for i in range(len(queries)):
#             results.append((final_responses[i], [doc.page_content for doc in batch_retrieved_docs[i]]))

#         if is_batch:
#             return results
#         else:
#             return results[0]

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
        return serialized, reranked_docs

    return hybrid_RAG_retrieve


def get_session_history(conversation_id: str) -> List[Dict]:
    """Get the conversation history for a session.
    
    Args:
        conversation_id: The conversation ID
        
    Returns:
        List of message dictionaries with 'role' and 'content'
    """
    storage = get_storage()
    messages = storage.load_session(conversation_id)
    
    if not messages:
        return []
    
    history = []
    for msg in messages:
        if isinstance(msg, dict):
            history.append(msg)
        elif isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            history.append({"role": "tool", "content": msg.content})
    
    return history


def clear_session(conversation_id: str) -> bool:
    storage = get_storage()
    return storage.delete_session(conversation_id)


def list_sessions() -> List[str]:
    storage = get_storage()
    return storage.list_sessions() 


async def main():
    config = RAGConfig()
    
    # Use the async factory method to create agent with checkpointing
    agent = await RAGAgent.create(config)
    
    query = input("Enter your query: ")
    
    # Stream responses
    async for response in agent.chat(query, conversation_id=str(3)):
        print(f"[{response.status.value}] {response.type}: {response.content}")
        if response.status == Status.COMPLETE:
            print(f"\nFinal token usage - Input: {response.metadata.input_tokens_used}, Output: {response.metadata.output_tokens_used}")
    
    # Clean up
    await agent.close()

if __name__ == "__main__":
    asyncio.run(main())
    