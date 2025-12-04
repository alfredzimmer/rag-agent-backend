"""
LangGraph-based RAG agent implementation.

This module defines the graph structure for the RAG agent using LangGraph,
providing native support for interrupts, checkpointing, and streaming.
"""

from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, END, MessagesState, START
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
import operator
from langmem.short_term import SummarizationNode, RunningSummary
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
import datetime
import asyncio

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

def create_agent_graph(llm_with_tools, rag_tool, memory_manager, *, debug: bool = False):
    # Define the agent node ##################################################
    async def agent_node(state: LLMInputState, config: RunnableConfig) -> State:
        """
        Agent node: LLM decides what to do next.
        Note: This uses invoke() for non-streaming. For streaming, we handle it in the agent.chat() method.
        """

        last_message = state["summarized_messages"][-1].content if state["summarized_messages"] else ""

        memories = await memory_manager.asearch(
            query=last_message,
            config=config,
            limit=5
        )

        memory_context = ""
        if memories:
            formatted = "\n".join([f"- {m.value}" for m in memories])
            memory_context = f"\n\nRELEVANT USER FACTS/MEMORIES:\n{formatted}"

        system_context =(
            "You are a helpful assistant with access to a specialized knowledge base. "
            "IMPORTANT: You should use the hybrid_RAG_retrieve tool before answering any technical question. "
            "Never rely solely on your general knowledge. Always check the knowledge base for relevant information."
            f"{memory_context}"
        )

        messages = [SystemMessage(content=system_context)] + state["summarized_messages"]

        
        if debug:
            # Debug logging
            log_content = f"Summarized Messages: {state['summarized_messages']}\n"
            if 'context' in state:
                log_content += f"Context: {state['context']}"
            log_debug("AGENT_NODE_INPUT", log_content)

            response = llm_with_tools.invoke(messages)

            # Log response
            log_debug("AGENT_RESPONSE", f"Response: {response}")
            log_debug("SYSTEM_CONTEXT", f"System context: {system_context}")

        response = await llm_with_tools.ainvoke(messages)

        return {
            "messages": [response],
            "context": state.get("context", {}),
            "input_tokens_used": response.usage_metadata.get("input_tokens", 0),
            "output_tokens_used": response.usage_metadata.get("output_tokens", 0)
        }

    async def save_memory_node(state: State, config: RunnableConfig):
        """
        Save memory node: Save the memory to the memory manager.
        """
        async def _background_save():
            try:
                await memory_manager.ainvoke({"messages": state["messages"]}, config=config)
                print(f"Memory saved successfully for user {config.get('configurable', {}).get('user_id', 'unknown')}") if debug else None
            except Exception as e:
                print(f"Error saving memory: {e}")
        asyncio.create_task(_background_save())
        return {}
    
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
    def should_continue(state: State) -> Literal["tools", "save_memory"]:
        """
        Router: Decide whether to continue to tools or end.
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # Otherwise, end
        return "save_memory"
    
    # Build the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("summarize", summarization_node)  
    workflow.add_node("save_memory", save_memory_node)

    # Set entry point
    workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "save_memory": "save_memory"
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "summarize")
    workflow.add_edge("save_memory", END)
    
    return workflow
