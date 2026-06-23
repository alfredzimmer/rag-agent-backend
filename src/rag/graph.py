"""
LangGraph-based RAG agent implementation.

This module defines the graph structure for the RAG agent using LangGraph,
providing native support for interrupts, checkpointing, and streaming.
"""

from typing import TypedDict, Annotated, Optional
from langchain_core.messages import AIMessage, SystemMessage, AnyMessage, HumanMessage
from langgraph.graph import StateGraph, END, MessagesState, START
import operator
from langchain_core.runnables import RunnableConfig
import datetime
import asyncio
import json
import textwrap
from .modules.custom_summarization import SummarizationNode, RunningSummary

# Strong references to running background tasks to prevent garbage collection
BACKGROUND_TASKS: set[asyncio.Task] = set()

class State(MessagesState):

    context: dict[str, RunningSummary]
    rating: float
    title: Optional[str]
    input_tokens_used: Annotated[int, operator.add]
    output_tokens_used: Annotated[int, operator.add]
    tasks: list[str]
    worker_results: list[dict]

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]

def create_agent_graph(
    llm,
    llm_with_tools,
    rag_tool,
    exa_tool=None,
    memory_manager=None,
    summarization_llm=None,
    training_llm=None,
    title_llm=None,
    *,
    max_context_tokens: int = 3000,
    debug: bool = False
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
        except Exception as e:
            if debug:
                print(f"[Orchestrator Error] Failed to parse tasks: {e}. Falling back to main query.")
            tasks = [clean_query]

        if debug:
            log_debug("ORCHESTRATOR_NODE", f"Original: {clean_query}\nDecomposed Tasks: {tasks}")

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
                    return await asyncio.to_thread(rag_tool.invoke, {"query": sub_query})
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

        if debug:
            log_content = ""
            for idx, res in enumerate(worker_results, 1):
                log_content += f"Worker {idx} Sub-query: {res['sub_query']}\nContext Length: {len(res['context'])} chars\n\n"
            log_debug("WORKER_NODE", log_content)

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

        memories = []
        if memory_manager:
            memories = await memory_manager.asearch(
                query=clean_query,
                config=config,
                limit=5
            )

        memory_context = ""
        if memories:
            formatted = "\n".join([f"- {m.value}" for m in memories if m.score is not None and m.score > 0.5])
            memory_context = f"\nRELEVANT USER FACTS/MEMORIES:\n{formatted}\n"

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
{memory_context}
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

        if debug:
            log_debug("SYNTHESIZER_NODE", f"Synthesized Response: {response}")

        # In langgraph State update, we return the synthesized AIMessage response to add to `messages`
        return {
            "messages": [response],
            "input_tokens_used": response.usage_metadata.get("input_tokens", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0,
            "output_tokens_used": response.usage_metadata.get("output_tokens", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0,
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
        task = asyncio.create_task(_background_save())
        BACKGROUND_TASKS.add(task)
        task.add_done_callback(BACKGROUND_TASKS.discard)
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

            # Run DeepEval evaluation
            deepeval_score = 0.0
            try:
                import sys
                import langchain_core.messages as core_messages
                import langchain_core.documents as core_documents

                # Mock langchain.schema to solve deepeval's legacy import issues
                if 'langchain.schema' not in sys.modules:
                    class MockSchema:
                        pass
                    mock_schema = MockSchema()
                    mock_schema.Document = core_documents.Document
                    mock_schema.HumanMessage = core_messages.HumanMessage
                    mock_schema.AIMessage = core_messages.AIMessage
                    mock_schema.SystemMessage = core_messages.SystemMessage
                    mock_schema.BaseMessage = core_messages.BaseMessage
                    sys.modules['langchain.schema'] = mock_schema

                from deepeval.metrics import AnswerRelevancyMetric
                from deepeval.test_case import LLMTestCase
                from deepeval.models import DeepEvalBaseLLM

                import requests

                class OllamaDeepEval(DeepEvalBaseLLM):
                    def __init__(self, model_name, base_url="http://127.0.0.1:11434"):
                        self.model_name = model_name
                        self.base_url = base_url
                        super().__init__(model_name)

                    def load_model(self):
                        return None

                    def get_model_name(self) -> str:
                        return self.model_name

                    def generate(self, prompt: str) -> str:
                        try:
                            payload = {
                                "model": self.model_name,
                                "prompt": prompt,
                                "stream": False,
                                "options": {
                                    "temperature": 0.0
                                }
                            }
                            res = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
                            res.raise_for_status()
                            return res.json().get("response", "")
                        except Exception as e:
                            return f"Error during generation: {e}"

                    async def a_generate(self, prompt: str) -> str:
                        return self.generate(prompt)

                eval_wrapper = OllamaDeepEval(training_llm.model)
                relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=eval_wrapper)
                test_case = LLMTestCase(
                    input=user_input,
                    actual_output=ai_output
                )

                # deepeval metrics have sync measure method. Let's run in executor thread.
                await asyncio.to_thread(relevancy_metric.measure, test_case)
                deepeval_score = relevancy_metric.score

                if debug:
                    log_debug("EVALUATOR_NODE_DEEPEVAL", f"DeepEval Answer Relevancy Score: {deepeval_score}\nReason: {getattr(relevancy_metric, 'reason', 'None')}")
            except Exception as de_err:
                if debug:
                    log_debug("EVALUATOR_NODE_DEEPEVAL_ERROR", f"DeepEval self-evaluation failed: {de_err}")

            # Combine the two metrics (custom training value + deepeval relevancy)
            if deepeval_score > 0.0:
                overall_score = (overall_score + deepeval_score) / 2.0

            result = {"rating": overall_score}
            return result

        except Exception as e:
            if debug:
                log_debug("EVALUATOR_NODE", f"Error during evaluation: {e}")
            return {
                "rating": 0.0
            }

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
