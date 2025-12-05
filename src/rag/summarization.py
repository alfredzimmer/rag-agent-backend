"""
Custom summarization node for LangGraph.

This provides more control over summarization behavior compared to the default SummarizationNode.
"""

from typing import Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_ollama import ChatOllama


class CustomSummarizationNode:
    """
    Custom summarization node that gives you control over:
    - How many recent messages to keep unsummarized
    - When to trigger summarization
    - Summary format and content
    """
    
    def __init__(
        self,
        model: ChatOllama,
        max_tokens: int = 4000,
        keep_last_n_messages: int = 11,  # Always keep last N messages unsummarized
        summary_max_tokens: int = 1024,
    ):
        """
        Initialize custom summarization node.
        
        Args:
            model: LLM to use for summarization
            max_tokens: Trigger summarization when total exceeds this
            keep_last_n_messages: Always keep this many recent messages unsummarized
            summary_max_tokens: Maximum tokens for the summary
        """
        self.model = model
        self.max_tokens = max_tokens
        self.keep_last_n_messages = keep_last_n_messages
        self.summary_max_tokens = summary_max_tokens
    
    async def __call__(self, state: dict) -> dict:
        """
        Process messages and summarize if needed.
        
        Returns state with 'summarized_messages' key.
        """
        messages = state.get("messages", [])
        
        # Count total tokens
        total_tokens = count_tokens_approximately(messages)
        
        # If under threshold, return all messages as-is
        if total_tokens <= self.max_tokens:
            return {"summarized_messages": messages}
        
        # Need to summarize
        # Always keep the last N messages unsummarized
        recent_messages = messages[-self.keep_last_n_messages:]
        messages_to_summarize = messages[:-self.keep_last_n_messages]
        
        if not messages_to_summarize:
            # If we don't have enough messages to summarize, just return recent ones
            return {"summarized_messages": recent_messages}
        
        # Create summary of older messages
        summary_text = await self._create_summary(messages_to_summarize)
        summary_message = SystemMessage(content=f"Previous conversation summary:\n{summary_text}")
        
        # Return summary + recent unsummarized messages
        return {"summarized_messages": [summary_message] + recent_messages}
    
    async def _create_summary(self, messages: list[BaseMessage]) -> str:
        """Create a summary of the given messages."""
        
        # Format messages for summarization
        conversation_text = self._format_messages_for_summary(messages)
        
        # Create summarization prompt
        summary_prompt = f"""Summarize the following conversation concisely, preserving key facts, decisions, and context.
            Focus on:
            - Main topics discussed
            - Important questions asked and answers given
            - Key facts or information shared
            - Any decisions or conclusions reached

            Keep the summary under {self.summary_max_tokens} tokens.

            Conversation:
            {conversation_text}

            Summary:"""
        
        # Generate summary
        response = await self.model.ainvoke([HumanMessage(content=summary_prompt)])
        return response.content
    
    def _format_messages_for_summary(self, messages: list[BaseMessage]) -> str:
        """Format messages into readable text for summarization."""
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                # Skip tool calls in summary
                if msg.content:
                    formatted.append(f"Assistant: {msg.content}")
            elif isinstance(msg, SystemMessage):
                formatted.append(f"System: {msg.content}")
        
        return "\n\n".join(formatted)


def create_custom_summarization_node(
    model: ChatOllama,
    max_tokens: int = 4000,
    keep_last_n_messages: int = 6,
    summary_max_tokens: int = 1024,
):
    """
    Factory function to create a custom summarization node.
    
    Args:
        model: LLM to use for summarization
        max_tokens: Trigger summarization when total exceeds this
        keep_last_n_messages: Always keep this many recent messages unsummarized
        summary_max_tokens: Maximum tokens for the summary
    
    Returns:
        Callable that can be used as a LangGraph node
    """
    node = CustomSummarizationNode(
        model=model,
        max_tokens=max_tokens,
        keep_last_n_messages=keep_last_n_messages,
        summary_max_tokens=summary_max_tokens,
    )
    # Return the __call__ method as a proper callable
    return node.__call__