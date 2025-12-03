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