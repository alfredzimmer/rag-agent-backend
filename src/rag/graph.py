"""
LangGraph-based RAG agent implementation.

This module defines the graph structure for the RAG agent using LangGraph,
providing native support for interrupts, checkpointing, and streaming.
"""

from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage, AnyMessage, HumanMessage
from langgraph.graph import StateGraph, END, MessagesState, START
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
import operator
from langmem.short_term import SummarizationNode, RunningSummary
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
import datetime
import asyncio
import textwrap
from .modules.custom_summarization import SummarizationNode

def log_debug(section: str, content: str):
    """Helper to log debug info with timestamp and formatting."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("debug_graph.log", "a") as f:
        f.write(f"[{timestamp}] [{section}]\n")
        # Wrap content if it's too long, but preserve existing newlines
        wrapper = textwrap.TextWrapper(width=100, break_long_words=False, replace_whitespace=False)
        formatted_content = "\n".join(wrapper.fill(line) for line in content.splitlines())
        f.write(f"{formatted_content}\n")
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
            messages = state['summarized_messages']
            log_content = f"Summarized Messages ({len(messages)} total):\n"
            for i, msg in enumerate(messages, 1):
                log_content += f"  Message {i}: {msg}\n\n"
            if 'context' in state:
                log_content += f"\nContext: {state['context']}"
            log_debug("AGENT_NODE_INPUT", log_content)

        response = await llm_with_tools.ainvoke(messages)

        if debug:
            log_debug("AGENT_RESPONSE", f"Response: {response}")
            log_debug("SYSTEM_CONTEXT", f"System context: {system_context}")
            log_debug("Input Tokens Used", f"Input tokens used: {response.usage_metadata.get('input_tokens', 0)}")
            log_debug("Output Tokens Used", f"Output tokens used: {response.usage_metadata.get('output_tokens', 0)}")

        # Check if this is the final response (no tool calls)
        # If so, clean up ToolMessages before returning
        has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls
        
        # if not has_tool_calls:
        #     # Agent is done - clean up all ToolMessages from state before returning
        #     # This prevents accumulation of tool call data across turns
        #     from langchain_core.messages import RemoveMessage
            
        #     # Find all ToolMessages to remove
        #     tool_messages_to_remove = [
        #         RemoveMessage(id=msg.id) 
        #         for msg in state.get("summarized_messages", []) 
        #         if isinstance(msg, ToolMessage)
        #     ]
            
        #     # Return removal commands + new response
        #     return {
        #         "messages": tool_messages_to_remove + [response],
        #         "context": state.get("context", {}),
        #         "input_tokens_used": response.usage_metadata.get("input_tokens", 0),
        #         "output_tokens_used": response.usage_metadata.get("output_tokens", 0)
        #     }
        
        # Has tool calls - return normally (ToolMessages will be added by tool node)
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
        model=summarization_model,
        max_tokens=2000,
        max_tokens_before_summary=2000,
        keep_last_n_messages=6,
        max_summary_tokens=1024,
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
