from typing import Union, List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import uuid
from enum import Enum
from pydantic import BaseModel, Field
import asyncio

from .milvus import create_milvus_store
from .modules.reranker import BGERanker
from .modules.hyde import HyDEGenerator
from .session_storage import get_storage

@dataclass
class RAGConfig:
    vector_store_type: str = "milvus"  
    ranker_type: str = "bge"
    dense_embedding_model: str = "qwen3-embedding:8b"
    sparse_embedding_model: str = "splade" #[splade, bm25, bge]
    llm_model: str = "qwen3:8b"
    hyde: bool = True

class Status(Enum):
   CREATED = "created"
   RESPONSE = "response"
   USAGE = "usage"
   FUNCTION = "function"
   COMPLETE = "complete"
   CANCEL = "cancel"
   ERROR = "error"

class Metadata(BaseModel):
   session_id: str = Field(..., description="Session ID")
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
    def __init__(self, config: RAGConfig):
        self.config = config
        self.interrupted = False
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
        self.rag_tool = create_rag_tool(vector_store, ranker, hyde_generator)
        
        llm = ChatOllama(model=config.llm_model, temperature=0)
        self.llm_with_tools = llm.bind_tools([self.rag_tool])


    def call(self, query: Union[str, List[str]]):
        # System message
        SYSTEM_MESSAGE = """You are a helpful assistant with access to a specialized knowledge base. 
        IMPORTANT: You MUST ALWAYS use the hybrid_RAG_retrieve tool FIRST before answering any question. 
        Never rely solely on your general knowledge. Always check the knowledge base for relevant information."""

        # Normalize input to list
        is_batch = isinstance(query, list)
        queries = query if is_batch else [query]

        # Initialize chat histories for each query
        all_messages = []
        for q in queries:
            all_messages.append([
                {"role": "system", "content": SYSTEM_MESSAGE},
                HumanMessage(content=q)
            ])
        
        # Store retrieved docs for each query
        batch_retrieved_docs = [[] for _ in queries]
        final_responses = [""] * len(queries)
        
        try:
            # Step 1: Initial batch call to LLM
            responses = self.llm_with_tools.batch(all_messages)
            
            # Track which indices need a second pass (tool execution)
            indices_to_process = []
            
            # Process initial responses
            for i, response in enumerate(responses):
                if response.tool_calls:
                    # Add the assistant's response with tool calls
                    all_messages[i].append(response)
                    indices_to_process.append(i)
                else:
                    # No tool call, just get the response
                    final_responses[i] = response.content

            # Step 2: Execute tools for those that need it
            if indices_to_process:
                # Collect all tool calls that need execution
                tool_tasks = []
                for i in indices_to_process:
                    response = responses[i]
                    for tool_call in response.tool_calls:
                        tool_tasks.append((i, tool_call))
                
                # Execute tool calls in parallel
                with ThreadPoolExecutor() as executor:
                    # Submit all tasks
                    future_to_task = {
                        executor.submit(self.rag_tool.invoke, task[1]["args"]): task 
                        for task in tool_tasks
                    }
                    
                    # Process results as they complete
                    for future in as_completed(future_to_task):
                        i, tool_call = future_to_task[future]
                        try:
                            serialized_context, docs = future.result()
                            
                            # Store docs
                            batch_retrieved_docs[i].extend(docs)
                            
                            # Add tool result to messages
                            all_messages[i].append(ToolMessage(
                                content=serialized_context,
                                tool_call_id=tool_call["id"]
                            ))
                        except Exception as exc:
                            print(f"Tool execution failed: {exc}")
                            # Add error message to tool result so the LLM knows it failed
                            all_messages[i].append(ToolMessage(
                                content=f"Error: {str(exc)}",
                                tool_call_id=tool_call["id"]
                            ))

                # Prepare second batch pass
                second_pass_indices = indices_to_process
                
                # Step 3: Second batch call to LLM for those that used tools
                if second_pass_indices:
                    second_pass_messages = [all_messages[i] for i in second_pass_indices]
                    second_responses = self.llm_with_tools.batch(second_pass_messages)
                    
                    for idx, response in zip(second_pass_indices, second_responses):
                        final_responses[idx] = response.content
            
        except Exception as e:
            print(f"Error in agent_call: {e}")
            if is_batch:
                return [], [f"Error: {str(e)}"] * len(queries)
            return [], f"Error: {str(e)}"

        # Return results formatted correctly
        results = []
        for i in range(len(queries)):
            results.append((final_responses[i], [doc.page_content for doc in batch_retrieved_docs[i]]))

        if is_batch:
            return results
        else:
            return results[0]
    
    async def chat(
        self,
        query: str,
        session_id: Optional[str] = None
    ):

        storage = get_storage()
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        yield ChatResponse(status=Status.CREATED, type="chat.created", content="", metadata=Metadata(session_id=session_id, input_tokens_used=0, output_tokens_used=0))
        
        input_tokens_used = 0
        output_tokens_used = 0
                        
        SYSTEM_MESSAGE = """You are a helpful assistant with access to a specialized knowledge base. 
        IMPORTANT: You MUST ALWAYS use the hybrid_RAG_retrieve tool FIRST before answering any question. 
        Never rely solely on your general knowledge. Always check the knowledge base for relevant information."""
        
        messages = storage.load_session(session_id)
        
        # Add system message if this is the first message
        if len(messages) == 0:
            messages.append({"role": "system", "content": SYSTEM_MESSAGE})
        
        messages.append(HumanMessage(content=query))
        
        retrieved_docs = []
        
        if self.interrupted:
            yield ChatResponse(status=Status.CANCEL, type="chat.cancel", content="", metadata=Metadata(session_id=session_id, input_tokens_used=input_tokens_used, output_tokens_used=output_tokens_used))
            return
        try:
            response = self.llm_with_tools.invoke(messages)
            if response.usage_metadata:
                input_tokens_used += response.usage_metadata.get("input_tokens", 0)
                output_tokens_used += response.usage_metadata.get("output_tokens", 0)
            
            if response.tool_calls:
                messages.append(response)
                
                for tool_call in response.tool_calls:
                    print(f"[Calling {tool_call['name']}...]")
                    yield ChatResponse(status=Status.RESPONSE, type="response.function_call_arguments.done", content=f"{self.rag_tool.name}: {tool_call['args']}", metadata=Metadata(session_id=session_id, input_tokens_used=input_tokens_used, output_tokens_used=output_tokens_used))
                    
                    if self.interrupted:
                        yield ChatResponse(status=Status.CANCEL, type="chat.cancel", content="", metadata=Metadata(session_id=session_id, input_tokens_used=input_tokens_used, output_tokens_used=output_tokens_used))
                        return
                    serialized_context, docs = self.rag_tool.invoke(tool_call["args"])
                    retrieved_docs.extend(docs)
                    
                    messages.append(ToolMessage(
                        content=serialized_context,
                        tool_call_id=tool_call["id"]
                    ))

                    yield ChatResponse(status=Status.FUNCTION, type="function", content=serialized_context, metadata=Metadata(session_id=session_id, input_tokens_used=input_tokens_used, output_tokens_used=output_tokens_used))


            # Final response generation (streamed)
            if self.interrupted:
                yield ChatResponse(status=Status.CANCEL, type="chat.cancel", content="", metadata=Metadata(session_id=session_id, input_tokens_used=input_tokens_used, output_tokens_used=output_tokens_used))
                return
            final_response = ""
            for chunk in self.llm_with_tools.stream(messages):
                if chunk.usage_metadata:
                    input_tokens_used += chunk.usage_metadata.get("input_tokens", 0)
                    output_tokens_used += chunk.usage_metadata.get("output_tokens", 0)
                if chunk.content:
                    yield ChatResponse(status=Status.RESPONSE, type="response.output_text.delta", content=chunk.content, metadata=Metadata(session_id=session_id, input_tokens_used=input_tokens_used, output_tokens_used=output_tokens_used))
                    final_response += chunk.content
            
            messages.append(AIMessage(content=final_response))
            response_text = final_response
            yield ChatResponse(status=Status.RESPONSE, type="response.output_text.done", content=final_response, metadata=Metadata(session_id=session_id, input_tokens_used=input_tokens_used, output_tokens_used=output_tokens_used))
            
            # Update session storage
            storage.save_session(session_id, messages)

            return
            
        except Exception as e:
            print(f"Error in agent_chat: {e}")
            import traceback
            traceback.print_exc()
            yield ChatResponse(status=Status.ERROR, type="error", content=str(e), metadata=Metadata(session_id=session_id, input_tokens_used=input_tokens_used, output_tokens_used=output_tokens_used))
            return
            

            

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
        print(f"Waiting for Embedding: {inputText[:100]}...") # Log first 100 chars
        
        vs = vector_store.get_vector_store()
        retrieved_docs = vs.similarity_search(inputText, k=30)

        # Rerank the retrieved documents
        # The ranker is a BGERanker instance
        k = 3
        reranked_docs = ranker.rerank(query, retrieved_docs, k)

        print(f"Reranked {len(reranked_docs)} docs")
        print(f"Reranked docs: {reranked_docs}")
        
        # Generate serialized output
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in reranked_docs
        )
        return serialized, reranked_docs

    return hybrid_RAG_retrieve


def get_session_history(session_id: str) -> List[Dict]:
    """Get the conversation history for a session.
    
    Args:
        session_id: The session ID
        
    Returns:
        List of message dictionaries with 'role' and 'content'
    """
    storage = get_storage()
    messages = storage.load_session(session_id)
    
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


def clear_session(session_id: str) -> bool:
    storage = get_storage()
    return storage.delete_session(session_id)


def list_sessions() -> List[str]:
    storage = get_storage()
    return storage.list_sessions() 


async def main():
    config = RAGConfig()
    agent = RAGAgent(config)
    query = input("Enter your query: ")
    session_id = str(uuid.uuid4())
    async for value in agent.chat(query, session_id=session_id):
        if value.status == Status.RESPONSE:
            if value.type == "response.output_text.delta":
                print(value.content, end="")
            elif value.type == "response.output_text.done":
                print()
            elif value.type == "response.function_call_arguments.done":
                print(f"[Calling {value.content}...]")
        else:
            print(f"Received: {value}")
    clear_session(session_id)

if __name__ == "__main__":
    asyncio.run(main())
