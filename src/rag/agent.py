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
    k = 3
    reranked_docs = rerank(query, retrieved_docs, k)

    # Generate serialized output
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in reranked_docs
    )
    return serialized, reranked_docs


def agent_call(query: str):
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
    
    # Add user message
    messages.append(HumanMessage(content=query))
    
    all_retrieved_docs = []
    final_response = ""
    
    try:
        # Get response from LLM
        response = llm_with_tools.invoke(messages)
        
        # Check if the model wants to use tools
        if response.tool_calls:
            # Add the assistant's response with tool calls
            messages.append(response)
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                
                # Execute the tool
                serialized_context, docs = hybrid_RAG_retrieve.invoke(tool_call["args"])
                all_retrieved_docs.extend(docs)
                
                # Add tool result to messages
                messages.append(ToolMessage(
                    content=serialized_context,
                    tool_call_id=tool_call["id"]
                ))
            
            # Get final response (non-streaming)
            final_response_msg = llm.invoke(messages)
            final_response = final_response_msg.content
        else:
            # No tool call, just get the response
            final_response = response.content
        
    except Exception as e:
        print(f"Error in agent_call: {e}")
        return [], f"Error: {str(e)}"

    return all_retrieved_docs, final_response


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