from langchain_core.tools import tool
from vectordb import qdrant_client
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from modules.retriever import hybrid_retrieve
from modules.utils import scoredpoint_to_document
from modules.reranker import rerank

model = ChatOpenAI(model="gpt-5-mini")

# @tool(response_format="content_and_artifact")
# def simple_RAG_retrieve(query: str):
#     """Retrieve top-2 chunks from the embedded Wiki article matching the query."""
#     retrieved_docs = vector_store.similarity_search(query, k=2)
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\nContent: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs



@tool(response_format="content_and_artifact")
def hybrid_RAG_retrieve(query: str):
    """
    Retrieve with hybrid + reranking
    """
    # len(dense_res) = len(sparse_res) <= 50 by default
    dense_res, sparse_res = hybrid_retrieve(query, qdrant_client, 10)
    # turn list of ScoredPoint to list of Document
    dense_docs = [scoredpoint_to_document(p) for p in dense_res]
    sparse_docs = [scoredpoint_to_document(p) for p in sparse_res]

    # rerank
    k = 3
    reranked_docs = rerank(query, dense_docs + sparse_docs, k)

    # genrate log
    # assumes that doc must have metadata and page_content 
    # when the associated PointStruct was upserted into Qdrant
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in reranked_docs
    )
    return serialized, reranked_docs



tools = [hybrid_RAG_retrieve]
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)

agent = create_agent(model, tools, system_prompt=prompt)

query = input('Ask you question: ')

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
