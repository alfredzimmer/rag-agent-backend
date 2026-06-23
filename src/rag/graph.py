"""
LangGraph-based RAG agent implementation.

This module defines the graph structure for the RAG agent using LangGraph,
providing native support for interrupts, checkpointing, and streaming.
"""

from typing import Annotated, Optional
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, MessagesState, START
import operator
from langchain_core.runnables import RunnableConfig
import asyncio
import json
import logging
from .modules.custom_summarization import SummarizationNode, RunningSummary

logger = logging.getLogger(__name__)

class State(MessagesState):

    context: dict[str, RunningSummary]
    rating: float
    title: Optional[str]
    input_tokens_used: Annotated[int, operator.add]
    output_tokens_used: Annotated[int, operator.add]
    tasks: list[str]
    worker_results: list[dict]

def create_agent_graph(
    llm,
    rag_tool,
    exa_tool=None,
    summarization_llm=None,
    title_llm=None,
    *,
    max_context_tokens: int = 3000,
):

    async def orchestrator_node(state: dict, config: RunnableConfig):
        """
        Orchestrator node: Decides what sub-queries/tasks are needed.
        """
        active_messages = state.get("summarized_messages", state.get("messages", []))
        last_user_message = ""
        for msg in reversed(active_messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break

        if not last_user_message:
            return {"tasks": []}

        # Clean instruction from user query if present
        clean_query = last_user_message
        if "Please use the hybrid_RAG_retrieve tool" in clean_query:
            clean_query = clean_query.replace("Please use the hybrid_RAG_retrieve tool to answer if needed. If retrieval yields no relevant results, DO NOT hallucinate. ", "", 1)

        orchestrator_prompt = f"""You are the coordinator/orchestrator of a multi-agent system.
Your job is to decompose the user's main query into 1 to 3 distinct, specific sub-queries or search tasks that can be executed in parallel to retrieve the necessary facts from a vector database.

Analyze the user's query and generate the sub-queries.
- If the query is simple, generate just 1 sub-query.
- If it is multi-faceted, generate 2 or 3 distinct sub-queries targeting different parts of the request.
- Return ONLY a JSON array of strings containing the sub-queries. Do NOT include any preamble, formatting, or extra text.

User Query: {clean_query}

JSON Response:"""

        try:
            # Generate tasks with base llm (using ainvoke since orchestrator does not stream output to user)
            response = await llm.ainvoke(orchestrator_prompt, config=config)
            content = response.content.strip()

            # Robust parsing of JSON array
            if "[" in content and "]" in content:
                content = content[content.find("["):content.rfind("]")+1]
            tasks = json.loads(content)
            if not isinstance(tasks, list):
                tasks = [clean_query]
        except Exception:
            logger.exception("Failed to decompose query; using the original query")
            tasks = [clean_query]

        logger.debug(
            "Query decomposition completed",
            extra={"original_query": clean_query, "tasks": tasks},
        )

        return {"tasks": tasks}

    async def worker_node(state: dict, config: RunnableConfig):
        """
        Worker node: Concurrently runs RAG retrieval for each sub-query.
        """
        tasks = state.get("tasks", [])
        if not tasks:
            # Fallback if no tasks generated
            active_messages = state.get("summarized_messages", state.get("messages", []))
            last_user_message = ""
            for msg in reversed(active_messages):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg.content
                    break
            tasks = [last_user_message] if last_user_message else [""]

        async def run_single_worker(sub_query: str):
            if not sub_query.strip():
                return {"sub_query": "", "context": "No sub-query specified."}

            # Read enable_exa from config
            enable_exa = config.get("configurable", {}).get("enable_exa", False)

            async def run_milvus():
                try:
                    return await asyncio.to_thread(
                        rag_tool.invoke,
                        {"query": sub_query},
                        config,
                    )
                except Exception as e:
                    return f"Error retrieving from Milvus: {e}"

            async def run_exa():
                if not exa_tool or not enable_exa:
                    return ""
                try:
                    return await asyncio.to_thread(exa_tool.invoke, {"query": sub_query})
                except Exception as e:
                    return f"Error retrieving from Exa: {e}"

            # Run Milvus and Exa search in parallel!
            milvus_res, exa_res = await asyncio.gather(run_milvus(), run_exa())

            # Combine context blocks and explicitly tag the sources
            combined_context = f"[Source: Milvus (Local Knowledge Base)]\n{milvus_res}\n\n"
            if exa_res:
                combined_context += f"[Source: Exa Web Search (Internet)]\n{exa_res}"

            return {
                "sub_query": sub_query,
                "context": combined_context
            }

        # Run all workers in parallel!
        worker_results = await asyncio.gather(*(run_single_worker(t) for t in tasks))

        logger.debug("Retrieval workers completed")

        return {"worker_results": worker_results}

    async def synthesizer_node(state: dict, config: RunnableConfig):
        """
        Synthesizer node: Combines raw retrieved contexts and generates/streams the final unified answer.
        """
        active_messages = state.get("summarized_messages", state.get("messages", []))
        last_user_message = ""
        for msg in reversed(active_messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break

        # Clean user query
        clean_query = last_user_message
        if "Please use the hybrid_RAG_retrieve tool" in clean_query:
            clean_query = clean_query.replace("Please use the hybrid_RAG_retrieve tool to answer if needed. If retrieval yields no relevant results, DO NOT hallucinate. ", "", 1)

        worker_results = state.get("worker_results", [])

        # Helper to parse chunks from serialized context string
        def parse_chunks(context_str, aspect_name):
            chunks = []
            current_source = "unknown"
            lines = context_str.split("\n")
            current_chunk = []

            for line in lines:
                if line.startswith("[Source: Milvus"):
                    current_source = "Milvus (Local Knowledge Base)"
                    continue
                elif line.startswith("[Source: Exa"):
                    current_source = "Exa Web Search (Internet)"
                    continue

                # Check if we are starting a new chunk
                if line.startswith("Source: {"):
                    if current_chunk:
                        chunks.append({
                            "source": current_source,
                            "aspect": aspect_name,
                            "text": "\n".join(current_chunk).strip()
                        })
                        current_chunk = []
                current_chunk.append(line)

            if current_chunk:
                chunks.append({
                    "source": current_source,
                    "aspect": aspect_name,
                    "text": "\n".join(current_chunk).strip()
                })
            return chunks

        # Group chunks by aspect
        chunks_by_aspect = {}
        for res in worker_results:
            aspect = res.get("sub_query", "")
            chunks = parse_chunks(res.get("context", ""), aspect)
            chunks_by_aspect[aspect] = chunks

        # Round-robin select chunks across aspects
        selected_chunks = []
        max_chunks_len = max(len(lst) for lst in chunks_by_aspect.values()) if chunks_by_aspect else 0
        for i in range(max_chunks_len):
            for aspect, chunks in chunks_by_aspect.items():
                if i < len(chunks):
                    selected_chunks.append(chunks[i])

        # Budget chunks by token count
        selected_by_aspect = {}
        current_tokens = 0
        for chunk in selected_chunks:
            chunk_repr = f"[{chunk['source']}]\n{chunk['text']}\n\n"
            chunk_tokens = count_tokens_approximately([HumanMessage(content=chunk_repr)])

            if current_tokens + chunk_tokens > max_context_tokens:
                break

            current_tokens += chunk_tokens
            aspect = chunk['aspect']
            if aspect not in selected_by_aspect:
                selected_by_aspect[aspect] = []
            selected_by_aspect[aspect].append(chunk)

        # Re-build final retrieved_contexts
        retrieved_contexts = ""
        for idx, (aspect, chunks) in enumerate(selected_by_aspect.items(), 1):
            retrieved_contexts += f"--- Retrieved Context for Aspect {idx}: {aspect} ---\n"
            milvus_chunks = [c for c in chunks if "Milvus" in c["source"]]
            exa_chunks = [c for c in chunks if "Exa" in c["source"]]

            if milvus_chunks:
                retrieved_contexts += "[Source: Milvus (Local Knowledge Base)]\n"
                retrieved_contexts += "\n\n".join(c["text"] for c in milvus_chunks) + "\n\n"
            if exa_chunks:
                retrieved_contexts += "[Source: Exa Web Search (Internet)]\n"
                retrieved_contexts += "\n\n".join(c["text"] for c in exa_chunks) + "\n\n"

        synthesizer_system = (
            "You are a synthesizer agent in a multi-agent RAG system. "
            "Your goal is to provide a single, comprehensive, and cohesive response to the user's query.\n"
            "You are provided with raw retrieved context blocks representing different aspects of the query, "
            "which contain sources labeled as '[Source: Milvus (Local Knowledge Base)]' and '[Source: Exa Web Search (Internet)]'.\n"
            "If there is any conflict between the information retrieved from Milvus and the web search results from Exa, "
            "you MUST prioritize the Milvus retrieval results as the ground truth. "
            "Be professional, structured, and avoid repeating the same facts multiple times."
        )

        synthesizer_user = f"""User original query: {clean_query}
Retrieved Context Blocks:
{retrieved_contexts}

Generate a clear, polished, and comprehensive response answering the user query based on the retrieved context above."""

        messages = [
            SystemMessage(content=synthesizer_system),
            HumanMessage(content=synthesizer_user)
        ]

        response = None
        # We stream using astream to allow the client to catch live token streams
        async for chunk in llm.astream(messages, config=config):
            if response is None:
                response = chunk
            else:
                response += chunk

        logger.debug("Synthesis completed")

        # In langgraph State update, we return the synthesized AIMessage response to add to `messages`
        return {
            "messages": [response],
            "input_tokens_used": response.usage_metadata.get("input_tokens", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0,
            "output_tokens_used": response.usage_metadata.get("output_tokens", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0,
        }

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
                logger.debug("Generated conversation title")
                return {"title": title}
            except Exception:
                logger.exception("Failed to generate conversation title")
                return {}
        return {}

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
    use_title = title_llm is not None

    # Helper to get next node in chain dynamically
    post_agent_nodes = []
    if use_title:
        post_agent_nodes.append("title")

    def get_next_step(current_step: str) -> str:
        if current_step == "synthesizer" or current_step == "agent":
            return post_agent_nodes[0] if post_agent_nodes else END

        try:
            idx = post_agent_nodes.index(current_step)
            if idx + 1 < len(post_agent_nodes):
                return post_agent_nodes[idx + 1]
            return END
        except ValueError:
            return END

    # Build the graph
    workflow = StateGraph(State)

    # Add Core Nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Entry point
    if use_summarize:
        workflow.add_node("summarize", summarization_node)
        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", "orchestrator")
    else:
        workflow.add_edge(START, "orchestrator")

    # Workflow chain
    workflow.add_edge("orchestrator", "worker")
    workflow.add_edge("worker", "synthesizer")

    # Synthesizer routes to post-agent chain
    workflow.add_edge("synthesizer", get_next_step("synthesizer"))

    # Optional nodes in the final chain
    if use_title:
        workflow.add_node("title", title_node)
        workflow.add_edge("title", get_next_step("title"))

    return workflow
