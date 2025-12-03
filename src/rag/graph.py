"""
LangGraph-based RAG agent implementation.

This module defines the graph structure for the RAG agent using LangGraph,
providing native support for interrupts, checkpointing, and streaming.
"""

from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, END, MessagesState, START
from langgraph.prebuilt import ToolNode
import operator
from langmem.short_term import SummarizationNode, RunningSummary
from langchain_core.messages.utils import count_tokens_approximately
from langchain_ollama import ChatOllama
import datetime
import json

def log_debug(section: str, content: str):
    """Helper to log debug info with timestamp and formatting."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("debug_graph.log", "a") as f:
        f.write(f"[{timestamp}] [{section}]\n")
        f.write(f"{content}\n")
        f.write("-" * 80 + "\n")


summarization_model = ChatOllama(model="qwen3:8b", temperature=0, num_predict=1024)

class State(MessagesState):
    context: dict[str, RunningSummary]  
    input_tokens_used: Annotated[int, operator.add]
    output_tokens_used: Annotated[int, operator.add]

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]

def create_agent_graph(llm_with_tools, rag_tool):
    """
    Create the LangGraph agent graph.
    
    Args:
        llm_with_tools: LLM instance with tools bound
        rag_tool: The RAG retrieval tool
        
    Returns:
        Compiled graph ready for invocation
    """
    
    # Define the agent node
    def agent_node(state: LLMInputState) -> State:
        """
        Agent node: LLM decides what to do next.
        Note: This uses invoke() for non-streaming. For streaming, we handle it in the agent.chat() method.
        """
        messages = [
            SystemMessage(
                content="You are a helpful assistant with access to a specialized knowledge base. "
                "IMPORTANT: You should use the hybrid_RAG_retrieve tool before answering any technical question. "
                "Never rely solely on your general knowledge. Always check the knowledge base for relevant information."
            )
        ] + state["summarized_messages"]

        print(f"Summarized messages: {state['summarized_messages']}")
        if 'context' in state:
            print(f"Context: {state['context']}")


        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "context": state.get("context", {}),
            "input_tokens_used": response.usage_metadata.get("input_tokens", 0),
            "output_tokens_used": response.usage_metadata.get("output_tokens", 0)
        }
    
    # Define the tool node using LangGraph's built-in ToolNode
    tool_node = ToolNode([rag_tool])

    summarization_node = SummarizationNode(  
        token_counter=count_tokens_approximately,
        model=summarization_model,
        max_tokens=4096,  # Increased to fit more messages in summarization context
        max_tokens_before_summary=2048,  # Trigger summary when conversation exceeds 2048 tokens
        max_summary_tokens=1024,  # Increased summary size for better context retention
    )

    
    # Define the router function
    def should_continue(state: State) -> Literal["tools", "end"]:
        """
        Router: Decide whether to continue to tools or end.
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # Otherwise, end
        return "end"
    
    # Build the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("summarize", summarization_node)  
    
    # Set entry point
    workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "summarize")
    
    # Compile the graph
    # Note: We'll add checkpointer and interrupt configuration when compiling in RAGAgent
    return workflow
