from langchain_core.tools import tool
from vectordb import vector_store, qdrant_client
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rag.modules.retriever import hybrid_retrieve
from rag.modules.utils import scoredpoint_to_document
from rag.modules.reranker import rerank

model = ChatOpenAI(model="gpt-5-mini")

@tool(response_format="content_and_artifact")
def simple_RAG_retrieve(query: str):
    """Retrieve top-2 chunks from the embedded Wiki article matching the query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


"""
Retrieve with hybrid + reranking
"""
@tool(response_format="content_and_artifact")
def hybrid_RAG_retrieve(query: str):
    # len(dense_res) = len(sparse_res) <= 50 by default
    dense_res, sparse_res = hybrid_retrieve(query, qdrant_client)
    # turn list of ScoredPoint to list of Document
    dense_docs = [scoredpoint_to_document(p) for p in dense_res]
    sparse_docs = [scoredpoint_to_document(p) for p in sparse_res]

    # rerank
    k = 5
    reranked_docs = rerank(query, dense_docs + sparse_docs, k)

    # genrate log
    # assumes that doc must have metadata and page_content 
    # when the associated PointStruct was upserted into Qdrant
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in reranked_docs
    )
    return serialized, reranked_docs



tools = [simple_RAG_retrieve]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You have access to a tool that retrieves context from a book. "
            "Use the tool to help answer user queries.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

query = input('Ask you question: ')

for event in agent_executor.stream(
    {"input": query}
):
    event["messages"][-1].pretty_print()
