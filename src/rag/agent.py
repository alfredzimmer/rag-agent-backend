from typing import Union, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from .milvus import create_milvus_store
from .modules.reranker import BGERanker
from .modules.hyde import HyDEGenerator

@dataclass
class RAGConfig:
    vector_store_type: str = "milvus"  
    ranker_type: str = "bge"
    dense_embedding_model: str = "qwen3-embedding:8b"
    sparse_embedding_model: str = "splade" #[splade, bm25, bge]
    llm_model: str = "qwen3:8b"
    hyde: bool = True

# Registry of component creators
VECTOR_STORES = {
    "milvus": create_milvus_store,
}

RANKERS = {
    "bge": BGERanker,
}

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

def build_agent(config: RAGConfig):
    # Get creator functions from registry
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
    
    # Return the runnable/chain and the tool (so we can execute it manually if needed)
    return llm_with_tools, rag_tool

def agent_call(query: Union[str, List[str]], config: RAGConfig = None):
    # Initialize the agent and tool using the factory
    if config is None:
        config = RAGConfig()
    llm_with_tools, rag_tool = build_agent(config)

    # System message
    SYSTEM_MESSAGE = """You are a helpful assistant with access to a knowledge base. 
    Use the hybrid_RAG_retrieve tool to search for relevant information when needed to answer user questions."""

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
        responses = llm_with_tools.batch(all_messages)
        
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
                    executor.submit(rag_tool.invoke, task[1]["args"]): task 
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
                second_responses = llm_with_tools.batch(second_pass_messages)
                
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


if __name__ == "__main__":
    # Initialize the agent
    config = RAGConfig(sparse_embedding_model="bm25", hyde=False)
    llm_with_tools, rag_tool = build_agent(config)

    # System message
    SYSTEM_MESSAGE = """You are a helpful assistant with access to a knowledge base. 
    Use the hybrid_RAG_retrieve tool to search for relevant information when needed to answer user questions."""

    # Initialize chat history
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

    print("RAG Agent started. Type 'exit' or 'quit' to end the conversation.\n")

    # Continuous conversation loop
    while True:
        user_input = input('\nYou: ')
        
        # Exit condition
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
                
        # Add user message
        messages.append(HumanMessage(content=user_input))
        
        print("Agent: ", end="", flush=True)
        
        try:
            # Get response from LLM
            response = llm_with_tools.invoke(messages)
            
            # Check if the model wants to use tools
            if response.tool_calls:
                # Add the assistant's response with tool calls
                messages.append(response)
                
                # Execute each tool call
                for tool_call in response.tool_calls:
                    print(f"\n[Calling {tool_call['name']}...]")
                    
                    # Execute the tool
                    tool_result = rag_tool.invoke(tool_call["args"])
                    
                    # Add tool result to messages
                    messages.append(ToolMessage(
                        content=tool_result[0], # tool returns (serialized, docs), we need serialized for message
                        tool_call_id=tool_call["id"]
                    ))
                
                # Get final response with tool results (streaming)
                print("Agent: ", end="", flush=True)
                final_response = ""
                for chunk in llm_with_tools.stream(messages):
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                        final_response += chunk.content
                
                print()  # New line
                messages.append(AIMessage(content=final_response))
            else:
                # No tool call, just stream the response
                full_response = ""
                for chunk in llm_with_tools.stream(messages):
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                        full_response += chunk.content
                
                print()  # New line
                messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nError: {e}")
            print("Please try again.")