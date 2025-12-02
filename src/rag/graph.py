"""
LangGraph-based RAG agent implementation.

This module defines the graph structure for the RAG agent using LangGraph,
providing native support for interrupts, checkpointing, and streaming.
"""

from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator


class AgentState(TypedDict):
    """State passed between nodes in the graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    input_tokens_used: int
    output_tokens_used: int


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
    def agent_node(state: AgentState) -> AgentState:
        """
        Agent node: LLM decides what to do next.
        """
        messages = [
            SystemMessage(
                content="You are a helpful assistant with access to a specialized knowledge base. "
                        "IMPORTANT: You MUST ALWAYS use the hybrid_RAG_retrieve tool FIRST before answering any question. "
                        "Never rely solely on your general knowledge. Always check the knowledge base for relevant information."
            )
        ] + state["messages"]


        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "input_tokens_used": state["input_tokens_used"] + response.usage_metadata.get("input_tokens", 0),
            "output_tokens_used": state["output_tokens_used"] + response.usage_metadata.get("output_tokens", 0)
        }
    
    # Define the tool node using LangGraph's built-in ToolNode
    tool_node = ToolNode([rag_tool])
    
    # Define the router function
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
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
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
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
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    # Note: We'll add checkpointer and interrupt configuration when compiling in RAGAgent
    return workflow
