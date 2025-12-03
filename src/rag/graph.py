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


summarization_model = ChatOllama(model="qwen3:8b", temperature=0, num_predict=256)

class State(MessagesState):
    context: dict[str, RunningSummary]  
    input_tokens_used: int
    output_tokens_used: int

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]
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
    def agent_node(state: LLMInputState) -> State:
        """
        Agent node: LLM decides what to do next.
        Note: This uses invoke() for non-streaming. For streaming, we handle it in the agent.chat() method.
        """
        messages = [
            SystemMessage(
                content="You are a helpful assistant with access to a specialized knowledge base. "
                "IMPORTANT: You MUST use the hybrid_RAG_retrieve tool before answering any technical question. "
                "Never rely solely on your general knowledge. Always check the knowledge base for relevant information."
            )
        ] + state["summarized_messages"]

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "input_tokens_used": state["input_tokens_used"] + response.usage_metadata.get("input_tokens", 0),
            "output_tokens_used": state["output_tokens_used"] + response.usage_metadata.get("output_tokens", 0)
        }
    
    # Define the tool node using LangGraph's built-in ToolNode
    tool_node = ToolNode([rag_tool])

    summarization_node = SummarizationNode(  
        token_counter=count_tokens_approximately,
        model=summarization_model,
        max_tokens=1024,
        max_tokens_before_summary=1024,
        max_summary_tokens=256,
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
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    # Note: We'll add checkpointer and interrupt configuration when compiling in RAGAgent
    return workflow
