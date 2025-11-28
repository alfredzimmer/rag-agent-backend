from typing import Union, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


from .milvus import vector_store
from .modules.reranker import rerank


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
    # The Milvus vector store handles hybrid search automatically
    retrieved_docs = vector_store.similarity_search(query, k=30)

    # Rerank the retrieved documents
    k = 2
    reranked_docs = list(rerank(query, retrieved_docs, k))
    
    # Generate serialized output
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in reranked_docs
    )
    return serialized, reranked_docs


def agent_call(query: Union[str, List[str]]):
    # Initialize the LLM with tool binding
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0,
    )

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools([hybrid_RAG_retrieve])

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
                    executor.submit(hybrid_RAG_retrieve.invoke, task[1]["args"]): task 
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
                second_responses = llm.batch(second_pass_messages)
                
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
    # Initialize the LLM with tool binding
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0,
    )

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools([hybrid_RAG_retrieve])

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
                    tool_result = hybrid_RAG_retrieve.invoke(tool_call["args"])
                    
                    # Add tool result to messages
                    messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"]
                    ))
                
                # Get final response with tool results (streaming)
                print("Agent: ", end="", flush=True)
                final_response = ""
                for chunk in llm.stream(messages):
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                        final_response += chunk.content
                
                print()  # New line
                messages.append(AIMessage(content=final_response))
            else:
                # No tool call, just stream the response
                full_response = ""
                for chunk in llm.stream(messages):
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                        full_response += chunk.content
                
                print()  # New line
                messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")