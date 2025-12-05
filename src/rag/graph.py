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
import datetime
import asyncio
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
    rating: float
    input_tokens_used: Annotated[int, operator.add]
    output_tokens_used: Annotated[int, operator.add]

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]

def create_agent_graph(llm_with_tools, rag_tool, memory_manager, training_llm=None, *, debug: bool = False):
    # Define the agent node ##################################################
    async def agent_node(state: LLMInputState, config: RunnableConfig):
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
            formatted = "\n".join([f"- {m.value}" for m in memories if m.score > 0.6])
            memory_context = f"\n\nRELEVANT USER FACTS/MEMORIES:\n{formatted}"

        system_context =(
            "You are a helpful assistant with access to a specialized knowledge base. "
            "IMPORTANT: You should use the hybrid_RAG_retrieve tool before answering any technical question. "
            "Never rely solely on your general knowledge. Always check the knowledge base for relevant information."
            f"{memory_context}"
        )

        messages = [SystemMessage(content=system_context)] + state["summarized_messages"]

        response = await llm_with_tools.ainvoke(messages)

        if debug:
            # Debug logging
            log_content = f"Summarized Messages: {state['summarized_messages']}\n"
            if 'context' in state:
                log_content += f"Context: {state['context']}"
            log_debug("AGENT_NODE_INPUT", log_content)
            log_debug("MEMORIES", f"Memories: {formatted if memories else 'No memories found'}")
            log_debug("AGENT_RESPONSE", f"Response: {response}")
            log_debug("TOKEN_USAGE", f"Input tokens: {response.usage_metadata.get('input_tokens', 0)}, Output tokens: {response.usage_metadata.get('output_tokens', 0)}")

        return {
            "messages": [response],
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

    async def evaluator_node(state: State, config: RunnableConfig):
        """
        Evaluator node: Evaluate the input and output quality.
        Checks if the response contains valuable information for training and involves technical content.
        Sets evaluation=True if threshold is reached.
        """
        if not training_llm:
            return {
                "rating": 0.0
            }
        
        messages = state["messages"]
        
        user_input = ""
        ai_output = ""
        
        # Find the last human message (input) and last AI message (output)
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not ai_output:
                ai_output = msg.content if isinstance(msg.content, str) else str(msg.content)
            elif isinstance(msg, HumanMessage) and not user_input:
                user_input = msg.content if isinstance(msg.content, str) else str(msg.content)
        
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
            
            if debug:
                log_debug("EVALUATOR_NODE", f"Evaluation response: {eval_text}")
            
            eval_json = json.loads(eval_text)
           
            valuable_score = float(eval_json.get("VALUABLE_FOR_TRAINING", 0.0))
            technical_score = float(eval_json.get("TECHNICAL_CONTENT", 0.0))
            explanation = eval_json.get("EXPLANATION", "")
            
            overall_score = (valuable_score + technical_score) / 2.0
            
            if debug:
                log_debug("EVALUATOR_NODE", 
                    f"Valuable Score: {valuable_score}, Technical Score: {technical_score}, "
                    f"Overall Score: {overall_score}"
                    f"Explanation: {explanation}")
            
            return {
                "rating": overall_score
            }
            
        except Exception as e:
            if debug:
                log_debug("EVALUATOR_NODE", f"Error during evaluation: {e}")
            return {
                "rating": 0.0
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
    def should_continue(state: State) -> Literal["tools", "evaluator"]:
        """
        Router: Decide whether to continue to tools or go to evaluator.
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls (only AIMessage can have tool_calls), continue to tools
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        return "evaluator"
    
    # Build the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("summarize", summarization_node)
    workflow.add_node("evaluator", evaluator_node)
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
            "evaluator": "evaluator"
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "summarize")
    # Add edge from evaluator to save_memory (always run evaluator before completion)
    workflow.add_edge("evaluator", "save_memory")
    workflow.add_edge("save_memory", END)
    
    return workflow
