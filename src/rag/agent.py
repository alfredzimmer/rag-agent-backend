import ollama
from vectordb import qdrant_client
from modules.retriever import hybrid_retrieve
from modules.utils import scoredpoint_to_document
from modules.reranker import rerank

# @tool(response_format="content_and_artifact")
# def simple_RAG_retrieve(query: str):
#     """Retrieve top-2 chunks from the embedded Wiki article matching the query."""
#     retrieved_docs = vector_store.similarity_search(query, k=2)
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\nContent: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs

def hybrid_RAG_retrieve(query: str):
    """
    Retrieve with hybrid + reranking
    """
    # len(dense_res) = len(sparse_res) <= 50 by default
    dense_res, sparse_res = hybrid_retrieve(query, qdrant_client, 30)
    # turn list of ScoredPoint to list of Document
    dense_docs = [scoredpoint_to_document(p) for p in dense_res]
    sparse_docs = [scoredpoint_to_document(p) for p in sparse_res]

    # rerank
    k = 3
    reranked_docs = rerank(query, dense_docs + sparse_docs, k)

    # generate log
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in reranked_docs
    )

    print("Context:", serialized)
    return serialized

# Define the tool schema for Ollama
tools = [{
    'type': 'function',
    'function': {
        'name': 'hybrid_RAG_retrieve',
        'description': 'Retrieve relevant context from the knowledge base using hybrid search and reranking',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The search query to find relevant information',
                },
            },
            'required': ['query'],
        },
    },
}]

# Initialize conversation with system message
messages = [
    {
        'role': 'system',
        'content': 'You are a helpful assistant with access to a knowledge base. You must pull resources the hybrid_RAG_retrieve tool to search for relevant information when needed to answer user questions.'
    }
]

print("RAG Agent started. Type 'exit' or 'quit' to end the conversation.\n")

# Continuous conversation loop
while True:
    user_input = input('\nYou: ')
    
    # Exit condition
    if user_input.lower() in ['exit', 'quit', 'q']:
        print("Goodbye!")
        break
    
    # Add user message to history
    messages.append({'role': 'user', 'content': user_input})
    
    print("Agent: ", end="", flush=True)
    
    # Stream the response
    response = ollama.chat(
        model='qwen3:30b-a3b-instruct-2507-q8_0',
        messages=messages,
        tools=tools,
        stream=True,
        think=False,
    )
    
    # Collect the full response
    full_content = ""
    tool_calls = []
    
    for chunk in response:
        # Check if this chunk has content
        if chunk['message'].get('content'):
            content = chunk['message']['content']
            print(content, end="", flush=True)
            full_content += content
        
        # Check if this chunk has tool calls
        if chunk['message'].get('tool_calls'):
            tool_calls = chunk['message']['tool_calls']
    
    print()  # New line after response
    
    # Add assistant's response to history
    if tool_calls:
        # Model wants to use tools
        messages.append({
            'role': 'assistant',
            'content': full_content,
            'tool_calls': tool_calls
        })
        
        # Execute each tool call
        for tool_call in tool_calls:
            function_name = tool_call['function']['name']
            arguments = tool_call['function']['arguments']
            
            print(f"[Calling tool: {function_name} with query: {arguments['query']}]")
            
            # Execute the tool
            if function_name == 'hybrid_RAG_retrieve':
                tool_result = hybrid_RAG_retrieve(arguments['query'])
            
            # Add tool result to messages
            messages.append({
                'role': 'tool',
                'content': tool_result,
            })
        
        # Get final response with tool results (streaming)
        print("Agent: ", end="", flush=True)
        final_response = ollama.chat(
            model='qwen3:8b',
            messages=messages,
            stream=True,
        )
        
        final_content = ""
        for chunk in final_response:
            if chunk['message'].get('content'):
                content = chunk['message']['content']
                print(content, end="", flush=True)
                final_content += content
        
        print()  # New line
        messages.append({'role': 'assistant', 'content': final_content})
    else:
        # No tool call, just add the response
        messages.append({'role': 'assistant', 'content': full_content})