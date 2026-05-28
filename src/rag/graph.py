"""
LangGraph-based RAG agent implementation.

This module defines the graph structure for the RAG agent using LangGraph,
providing native support for interrupts, checkpointing, and streaming.
"""

from typing import TypedDict, Annotated, Sequence, Literal, Optional
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage, AnyMessage, HumanMessage
from langgraph.graph import StateGraph, END, MessagesState, START
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
import operator
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
import datetime
import asyncio
import json
import textwrap
from .modules.custom_summarization import SummarizationNode, RunningSummary

class State(MessagesState):
    context: dict[str, RunningSummary]
    rating: float
    title: Optional[str]
    input_tokens_used: Annotated[int, operator.add]
    output_tokens_used: Annotated[int, operator.add]

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]

def create_agent_graph(
    llm_with_tools,
    rag_tool,
    memory_manager=None,
    summarization_llm=None,
    training_llm=None,
    title_llm=None,
    *,
    debug: bool = False
):

    # Define the agent node
    async def agent_node(state: dict, config: RunnableConfig):
        """
        Agent node: LLM decides what to do next.
        Note: This uses invoke() for non-streaming. For streaming, we handle it in the agent.chat() method.
        """

        active_messages = state.get("summarized_messages", state.get("messages", []))
        print(f"[AGENT NODE DEBUG] Running agent node. active_messages count: {len(active_messages)}")
        for idx, m in enumerate(active_messages):
            print(f"  [{idx}] Type: {type(m).__name__}, ID: {getattr(m, 'id', None)}, Content: {str(m.content)[:100]}")
        last_message = active_messages[-1].content if active_messages else ""

        memories = []
        if memory_manager:
            memories = await memory_manager.asearch(
                query=last_message,
                config=config,
                limit=5
            )

        memory_context = ""
        if memories:
            formatted = "\n".join([f"- {m.value}" for m in memories if m.score is not None and m.score > 0.5])
            memory_context = f"\n\nRELEVANT USER FACTS/MEMORIES:\n{formatted}"
            
            # Debug: log all memories with scores
            if debug:
                all_memories_debug = "\n".join([f"Score {m.score:.3f}: {m.value}" for m in memories])
                log_debug("MEMORIES_WITH_SCORES", all_memories_debug if memories else "No memories found")

        latest_user_index = next(
            (
                idx
                for idx in range(len(active_messages) - 1, -1, -1)
                if isinstance(active_messages[idx], HumanMessage)
            ),
            None,
        )
        latest_turn_has_tool_result = (
            latest_user_index is not None
            and any(isinstance(msg, ToolMessage) for msg in active_messages[latest_user_index + 1:])
        )
        retrieval_instruction = (
            "The latest user request already has retrieved context in the messages. "
            "Do not call hybrid_RAG_retrieve again for this same request; answer using that tool result. "
            if latest_turn_has_tool_result
            else "Use the hybrid_RAG_retrieve tool before answering the latest user request. "
        )

        system_context = (
            "You are a helpful assistant with access to a specialized knowledge base and user memories. "
            "IMPORTANT INSTRUCTIONS:\n"
            f"1. {retrieval_instruction}"
            "Never rely solely on your general knowledge for technical content.\n"
            "2. For personal questions about the user: Check the 'RELEVANT USER FACTS/MEMORIES' section below FIRST. "
            "If the answer is in the memories, use that information directly. "
            f"{memory_context}"
        )

        messages = [SystemMessage(content=system_context)] + active_messages

        response = None
        async for chunk in llm_with_tools.astream(messages, config=config):
            if response is None:
                response = chunk
            else:
                response += chunk

        if debug:
            # Debug logging
            log_content = f"Active Messages ({len(active_messages)} total):\n"
            for i, msg in enumerate(active_messages, 1):
                log_content += f"  Message {i}: {msg}\n\n"
            if 'context' in state:
                log_content += f"\nContext: {state['context']}"
            log_debug("AGENT_NODE_INPUT", log_content)
            log_debug("AGENT_RESPONSE", f"Response: {response}")
            log_debug("TOKEN_USAGE", f"Input tokens: {response.usage_metadata.get('input_tokens', 0)}, Output tokens: {response.usage_metadata.get('output_tokens', 0)}")

        # Check if this is the final response (no tool calls)
        # If so, clean up ToolMessages before returning
        has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls
        
        if not has_tool_calls:
            # Agent is done - clean up all ToolMessages and AIMessages with tool_calls from state before returning
            # This prevents accumulation of tool call data and keeps conversation history valid across turns
            from langchain_core.messages import RemoveMessage
            
            # Find all ToolMessages and AIMessages with tool_calls to remove
            messages_to_remove = []
            for msg in active_messages:
                if isinstance(msg, ToolMessage) and msg.id:
                    messages_to_remove.append(RemoveMessage(id=msg.id))
                elif isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls and msg.id:
                    messages_to_remove.append(RemoveMessage(id=msg.id))
            
            # Return removal commands + new response
            return {
                "messages": messages_to_remove + [response],
                "context": state.get("context", {}),
                "input_tokens_used": response.usage_metadata.get("input_tokens", 0),
                "output_tokens_used": response.usage_metadata.get("output_tokens", 0)
            }
        
        # Has tool calls - return normally (ToolMessages will be added by tool node)
        return {
            "messages": [response],
            "input_tokens_used": response.usage_metadata.get("input_tokens", 0),
            "output_tokens_used": response.usage_metadata.get("output_tokens", 0)
        }

    async def save_memory_node(state: State, config: RunnableConfig):
        """
        Save memory node: Save the memory to the memory manager.
        """
        if not memory_manager:
            return {}
            
        async def _background_save():
            try:
                await memory_manager.ainvoke({"messages": state["messages"]}, config=config)
                print(f"Memory saved successfully for user {config.get('configurable', {}).get('user_id', 'unknown')}") if debug else None
            except Exception as e:
                print(f"Error saving memory: {e}")
        asyncio.create_task(_background_save())
        return {}

    async def title_node(state: State, config: RunnableConfig):
        """
        Title generation node: Generate a title for the conversation.
        """
        messages = state["messages"]
        first_human = first_ai = None
        
        first_human_content = ""
        first_ai_content = ""
        
        for msg in messages:
            content = msg.content
            if isinstance(content, str) and "Please use the hybrid_RAG_retrieve tool" in content:
                content = content.replace("Please use the hybrid_RAG_retrieve tool to answer if needed. If retrieval yields no relevant results, DO NOT hallucinate. ", "", 1)
            
            if isinstance(msg, HumanMessage) and first_human is None:
                first_human = msg
                first_human_content = content
            if isinstance(msg, AIMessage) and first_ai is None:
                first_ai = msg
                first_ai_content = content

        if len(messages) <= 4 and first_human and first_ai:
            try:
                title_prompt = f"""Generate a distinct, 3-5 word title for this conversation based on the conversation history:
                user: {first_human_content}
                assistant: {first_ai_content}
                """
                title_res = await title_llm.ainvoke(title_prompt)
                title = title_res.content
                if debug:
                    log_debug("TITLE_GENERATION", f"Generated Title: {title}")
                return {"title": title}
            except Exception as e:
                if debug:
                    log_debug("TITLE_GENERATION", f"Error generating title: {e}")
                return {}
        return {}

    async def evaluator_node(state: State, config: RunnableConfig):
        """
        Evaluator node: Evaluate the input and output quality.
        Checks if the response contains valuable information for training and involves technical content.
        Sets evaluation=True if threshold is reached.
        """
        messages = state["messages"]

        last_human = last_ai = None
        
        last_human_content = ""
        last_ai_content = ""

        for msg in reversed(messages):
            content = msg.content
            if isinstance(content, str) and "Please use the hybrid_RAG_retrieve tool" in content:
                content = content.replace("Please use the hybrid_RAG_retrieve tool to answer if needed. If retrieval yields no relevant results, DO NOT hallucinate. ", "", 1)
            
            if isinstance(msg, HumanMessage) and last_human is None:
                last_human = msg
                last_human_content = content
            if isinstance(msg, AIMessage) and last_ai is None:
                last_ai = msg
                last_ai_content = content

        # Use last pair for evaluation
        user_input = last_human_content if last_human else ""
        ai_output = last_ai_content if last_ai else ""

        if not user_input or not ai_output:
            if debug:
                log_debug("EVALUATOR_NODE", "Skipping evaluation: missing input or output")
            return {
                "rating": 0.0
            }
        
        # Create evaluation prompt
        evaluation_prompt = f"""You are an expert evaluator for training data quality. Evaluate the following input-output pair.
Input (User Query):
{user_input}
Output (AI Response):
{ai_output}
Evaluate this pair on two criteria:
1. **Valuable for Training**: Does this pair contain valuable information that would be useful for training a model? Consider if it demonstrates clear reasoning, provides technical knowledge, or shows good question-answer patterns.
2. **Technical Content**: Does the response involve technical content, concepts, procedures, or domain-specific knowledge?

For each criterion, provide a score from 0.0 to 1.0, where:
- 0.0-0.3: Poor quality, not valuable
- 0.4-0.6: Moderate quality, somewhat valuable
- 0.7-0.9: Good quality, valuable
- 1.0: Excellent quality, highly valuable

Format all responses as JSON object with the following keys:
"VALUABLE_FOR_TRAINING": [score as float],
"TECHNICAL_CONTENT": [score as float],
"EXPLANATION": [brief explanation of your scores as string]
"""
        try:
            # Use training LLM to evaluate
            eval_response = await training_llm.ainvoke(evaluation_prompt)
            eval_text = eval_response.content if hasattr(eval_response, 'content') else str(eval_response)
                    
            eval_json = json.loads(eval_text)
           
            valuable_score = float(eval_json.get("VALUABLE_FOR_TRAINING", 0.0))
            technical_score = float(eval_json.get("TECHNICAL_CONTENT", 0.0))
            
            overall_score = (valuable_score + technical_score) / 2.0

            if debug:
                    log_debug("EVALUATOR_NODE", f"Evaluation response: {eval_text}\nOverall score: {overall_score}")
            
            result = {"rating": overall_score}
            return result
            
        except Exception as e:
            if debug:
                log_debug("EVALUATOR_NODE", f"Error during evaluation: {e}")
            return {
                "rating": 0.0
            }
    
    # Define the tool node using LangGraph's built-in ToolNode
    tool_node = ToolNode([rag_tool])

    summarization_node = None
    if summarization_llm:
        summarization_node = SummarizationNode(
            model=summarization_llm,
            max_tokens=3000,
            max_tokens_before_summary=3000,
            keep_last_n_messages=6,
            max_summary_tokens=512,
            input_messages_key="messages",
            output_messages_key="messages",
        )

    # Check for optional component availability
    use_summarize = summarization_llm is not None
    use_evaluator = training_llm is not None
    use_save_memory = memory_manager is not None
    use_title = title_llm is not None

    # Helper to get next node in chain dynamically
    post_agent_nodes = []
    if use_title:
        post_agent_nodes.append("title")
    if use_evaluator:
        post_agent_nodes.append("evaluator")
    if use_save_memory:
        post_agent_nodes.append("save_memory")
        
    def get_next_step(current_step: str) -> str:
        if current_step == "agent":
            return post_agent_nodes[0] if post_agent_nodes else END
            
        try:
            idx = post_agent_nodes.index(current_step)
            if idx + 1 < len(post_agent_nodes):
                return post_agent_nodes[idx + 1]
            return END
        except ValueError:
            return END

    # Define the router function
    def should_continue(state: State) -> Literal["tools", "next"]:
        """
        Router: Decide whether to continue to tools or move to the next phase.
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls (only AIMessage can have tool_calls), continue to tools
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        return "next"
    
    # Build the graph
    workflow = StateGraph(State)
    
    # Add Core Nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Entry point and Loop logic
    if use_summarize:
        workflow.add_node("summarize", summarization_node)
        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", "agent")
        workflow.add_edge("tools", "summarize") # loop back through summarize
    else:
        workflow.add_edge(START, "agent")
        workflow.add_edge("tools", "agent") # loop back directly to agent
    
    # Post-Agent branching
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "next": get_next_step("agent")
        }
    )
    
    # Optional nodes in the final chain
    if use_title:
        workflow.add_node("title", title_node)
        workflow.add_edge("title", get_next_step("title"))
        
    if use_evaluator:
        workflow.add_node("evaluator", evaluator_node)
        workflow.add_edge("evaluator", get_next_step("evaluator"))
    
    if use_save_memory:
        workflow.add_node("save_memory", save_memory_node)
        workflow.add_edge("save_memory", get_next_step("save_memory"))
    
    return workflow

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
