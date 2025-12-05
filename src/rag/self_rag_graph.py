"""
Self-RAG LangGraph implementation.

This module implements a streamlined RAG strategy with document grading
to filter irrelevant retrievals before generation.

Simplified from full Self-RAG to avoid duplicate streaming issues.
"""

from typing import TypedDict, Annotated, List
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END, MessagesState, START
from langmem.short_term import SummarizationNode, RunningSummary
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import operator
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


# =============================================================================
# Pydantic Models for Structured LLM Output
# =============================================================================

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# =============================================================================
# State Definitions
# =============================================================================

class SelfRAGState(MessagesState):
    """State for Self-RAG graph."""
    question: str  # The current question being processed
    documents: List[Document]  # Retrieved and filtered documents
    generation: str  # The generated answer
    context: dict[str, RunningSummary]  # For summarization
    rating: float  # Evaluation rating
    input_tokens_used: Annotated[int, operator.add]
    output_tokens_used: Annotated[int, operator.add]


class LLMInputState(TypedDict):
    """Input state for LLM nodes after summarization."""
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]


# =============================================================================
# Graph Factory
# =============================================================================

def create_self_rag_graph(
    llm,
    retriever,
    memory_manager,
    training_llm=None,
    *,
    debug: bool = False,
    grade_documents: bool = True,  # Set to False to skip document grading for faster response
):
    """
    Create a streamlined RAG graph with optional document grading.
    
    This is a simplified version that:
    - Generates only ONCE (no regeneration loops = no duplicate streaming)
    - Optionally grades documents in PARALLEL for speed
    - Maintains quality through document filtering rather than output grading
    
    Args:
        llm: The language model for generation
        retriever: The retriever (e.g., vector_store.as_retriever())
        memory_manager: Memory manager for storing user memories
        training_llm: Optional LLM for evaluation (training mode)
        debug: Enable debug logging
        grade_documents: Whether to grade documents for relevance (disable for faster response)
    
    Returns:
        StateGraph workflow (not compiled)
    """
    
    # Create structured LLM grader for documents
    structured_llm_doc_grader = llm.with_structured_output(GradeDocuments)
    
    # =========================================================================
    # Prompts
    # =========================================================================
    
    # Document relevance grader prompt
    doc_grade_system = """You are a grader assessing relevance of a retrieved document to a user question.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
    
    doc_grade_prompt = ChatPromptTemplate.from_messages([
        ("system", doc_grade_system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    retrieval_grader = doc_grade_prompt | structured_llm_doc_grader
    
    # RAG generation prompt
    rag_system = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise."""
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", rag_system),
        ("human", "Question: {question} \n\nContext: {context}"),
    ])
    
    rag_chain = rag_prompt | llm | StrOutputParser()
    
    # =========================================================================
    # Summarization Node
    # =========================================================================
    
    summarization_node = SummarizationNode(
        token_counter=count_tokens_approximately,
        model=summarization_model,
        max_tokens=4096,
        max_tokens_before_summary=2048,
        max_summary_tokens=1024,
    )
    
    # =========================================================================
    # Node Functions
    # =========================================================================
    
    async def extract_question_node(state: LLMInputState, config: RunnableConfig):
        """
        Extract the question from summarized messages and prepare for retrieval.
        """
        messages = state["summarized_messages"]
        
        # Get the last human message as the question
        question = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                question = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
        
        if debug:
            log_debug("EXTRACT_QUESTION", f"Question: {question}")
        
        return {
            "question": question,
            "documents": [],
            "generation": "",
        }
    
    async def retrieve_node(state: SelfRAGState, config: RunnableConfig):
        """
        Retrieve documents based on the current question.
        """
        question = state["question"]
        
        # Use the retriever to get documents
        documents = await retriever.ainvoke(question)
        
        if debug:
            log_debug("RETRIEVE", f"Question: {question} | Retrieved {len(documents)} documents")
        
        return {"documents": documents}
    
    async def grade_documents_node(state: SelfRAGState, config: RunnableConfig):
        """
        Grade retrieved documents for relevance to the question IN PARALLEL.
        Filter out irrelevant documents.
        """
        if not grade_documents:
            # Skip grading - use all documents
            if debug:
                log_debug("GRADE_DOCUMENTS", "Skipping document grading (disabled)")
            return {}
        
        question = state["question"]
        documents = state["documents"]
        
        if not documents:
            return {"documents": []}
        
        async def grade_single_doc(doc: Document) -> tuple[Document, bool]:
            """Grade a single document, returns (doc, is_relevant)."""
            try:
                score = await retrieval_grader.ainvoke({
                    "question": question,
                    "document": doc.page_content
                })
                is_relevant = score.binary_score.lower() == "yes"
                return (doc, is_relevant)
            except Exception as e:
                if debug:
                    log_debug("GRADE_DOCUMENTS", f"Error grading document: {e}")
                # On error, include the document to be safe
                return (doc, True)
        
        # Grade all documents in PARALLEL
        results = await asyncio.gather(*[grade_single_doc(doc) for doc in documents])
        
        # Filter to only relevant documents
        filtered_docs = [doc for doc, is_relevant in results if is_relevant]
        relevant_count = len(filtered_docs)
        irrelevant_count = len(documents) - relevant_count
        
        if debug:
            log_debug("GRADE_DOCUMENTS", 
                     f"Graded {len(documents)} docs in parallel: {relevant_count} relevant, {irrelevant_count} irrelevant"
                     f"Documents: {documents}")
        
        return {"documents": filtered_docs}
    
    async def generate_node(state: SelfRAGState, config: RunnableConfig):
        """
        Generate an answer based on documents.
        This node runs ONCE and streams directly to the user.
        The AI message is added here since there's no regeneration loop.
        """
        question = state["question"]
        documents = state["documents"]
        
        # Format documents for context
        if documents:
            context = "\n\n".join(doc.page_content for doc in documents)
        else:
            context = "No relevant documents found."
        
        # Generate answer (this will stream to user)
        generation = await rag_chain.ainvoke({
            "context": context,
            "question": question
        })
        
        if debug:
            log_debug("GENERATE", f"Question: {question}\nGeneration: {generation[:200]}...")
        
        # Add AI message to conversation
        ai_message = AIMessage(content=generation)
        
        return {
            "generation": generation,
            "messages": [ai_message]
        }
    
    async def save_memory_node(state: SelfRAGState, config: RunnableConfig):
        """
        Save memory in the background.
        """
        async def _background_save():
            try:
                await memory_manager.ainvoke({"messages": state["messages"]}, config=config)
                if debug:
                    user_id = config.get('configurable', {}).get('user_id', 'unknown')
                    print(f"Memory saved successfully for user {user_id}")
            except Exception as e:
                print(f"Error saving memory: {e}")
        
        asyncio.create_task(_background_save())
        return {}
    
    async def evaluator_node(state: SelfRAGState, config: RunnableConfig):
        """
        Evaluate the input and output quality for training purposes.
        Runs in background to not block the response.
        """
        if not training_llm:
            return {"rating": 0.0}
        
        question = state["question"]
        generation = state["generation"]
        
        if not question or not generation:
            if debug:
                log_debug("EVALUATOR", "Skipping evaluation: missing input or output")
            return {"rating": 0.0}
        
        evaluation_prompt = f"""You are an expert evaluator for training data quality. Evaluate the following input-output pair.
Input (User Query):
{question}
Output (AI Response):
{generation}
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
            eval_response = await training_llm.ainvoke(evaluation_prompt)
            eval_text = eval_response.content if hasattr(eval_response, 'content') else str(eval_response)
            
            eval_json = json.loads(eval_text)
            valuable_score = float(eval_json.get("VALUABLE_FOR_TRAINING", 0.0))
            technical_score = float(eval_json.get("TECHNICAL_CONTENT", 0.0))
            overall_score = (valuable_score + technical_score) / 2.0
            
            if debug:
                log_debug("EVALUATOR", f"Score: {overall_score} (valuable: {valuable_score}, technical: {technical_score})")
            
            return {"rating": overall_score}
            
        except Exception as e:
            if debug:
                log_debug("EVALUATOR", f"Error during evaluation: {e}")
            return {"rating": 0.0}
    
    # =========================================================================
    # Build the Graph (Simplified - No Regeneration Loops)
    # =========================================================================
    
    workflow = StateGraph(SelfRAGState)
    
    # Add nodes
    workflow.add_node("summarize", summarization_node)
    workflow.add_node("extract_question", extract_question_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("save_memory", save_memory_node)
    
    # summarize → extract_question → retrieve → grade_documents → generate → evaluator → save_memory → END
    workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "extract_question")
    workflow.add_edge("extract_question", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", "evaluator")
    workflow.add_edge("evaluator", "save_memory")
    workflow.add_edge("save_memory", END)
    
    return workflow
