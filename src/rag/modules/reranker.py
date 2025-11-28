"""Reranking utilities backed by BGE."""

from langchain_core.documents import Document
from FlagEmbedding import FlagReranker
from threading import Lock

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
reranker_lock = Lock()

def to_qa_pair(query: str, documents: list[Document]) -> list[list[str]]:
    qa_pair: list[list[str]] = []

    for idx, doc in enumerate(documents):
        qa_pair.append([query, doc.page_content])

    return qa_pair


def rerank(query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
    """Return the BGE reranked documents for *query*."""
    if not documents:
        return []

    qa_pair = to_qa_pair(query, documents)
    with reranker_lock:
        scores = reranker.compute_score(qa_pair, normalize=True)

    scored_docs: list[Document] = []
    for doc, score in zip(documents, scores):
        metadata = dict(doc.metadata)
        metadata["rerank_score"] = float(score)
        scored_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )

    ordered_docs = sorted(
        scored_docs,
        key=lambda doc: doc.metadata.get("rerank_score", 0.0),
        reverse=True,
    )

    if top_k is not None:
        ordered_docs = ordered_docs[:top_k]

    return ordered_docs
