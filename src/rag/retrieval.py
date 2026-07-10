"""Two-stage retrieval for the RAG agent.

The agent originally searched Milvus once and fed the top_k dense hits straight
to the model. That single-pass search is the documented recall bottleneck: near
-duplicate chunks crowd distinct evidence out of the top_k, and nothing reorders
hits by actual relevance. This module keeps that exact behaviour as the default
("none") and adds two opt-in backends that need no new dependency and no second
model resident on the GPU:

- "mmr": over-fetch ``fetch_k`` candidates, then Maximal Marginal Relevance
  selects a diverse top_k so duplicate passages stop displacing distinct ones.
- "llm": over-fetch ``fetch_k`` candidates, then a listwise rerank with the
  already-resident chat model reorders them before the top_k is kept.

Every backend degrades gracefully: any failure falls back to plain dense order,
so a misconfigured or overloaded reranker can never make retrieval worse than
the "none" baseline.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional, Protocol

from langchain_core.documents import Document

from .config import RAGConfig

logger = logging.getLogger(__name__)

BACKENDS = ("none", "mmr", "llm")

# Candidate passages are truncated to this many characters before being shown to
# the reranking model; enough to judge relevance without blowing up the prompt.
SNIPPET_CHARS = 400

RERANK_PROMPT = """You rank retrieved passages by how well each one helps answer a question.

Question: {query}

Passages:
{listing}

Return the passage numbers ordered from most to least relevant (most relevant
first) as a comma-separated list, e.g. "3, 1, 5". List only the {k} most
relevant numbers and output nothing but the numbers."""


class _VectorStore(Protocol):
    def similarity_search(self, query: str, k: int) -> list[Document]: ...

    def max_marginal_relevance_search(
        self, query: str, k: int, fetch_k: int, lambda_mult: float
    ) -> list[Document]: ...


class _ChatModel(Protocol):
    def invoke(self, prompt: str): ...


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Collapse whitespace and clip a passage to a rerank-friendly length."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "…"


def _parse_order(text: str, n: int) -> list[int]:
    """Turn a model's "3, 1, 5" reply into a complete 0-based permutation.

    Passages are labelled 1..n in the prompt. We take the integers in the reply
    that fall in range (deduplicated, order preserved), then append any labels
    the model omitted so no candidate is ever silently dropped. A garbled reply
    therefore degrades to the original dense order rather than losing documents.
    """
    order: list[int] = []
    seen: set[int] = set()
    for match in re.findall(r"\d+", text):
        idx = int(match) - 1
        if 0 <= idx < n and idx not in seen:
            order.append(idx)
            seen.add(idx)
    for idx in range(n):
        if idx not in seen:
            order.append(idx)
            seen.add(idx)
    return order


class Retriever:
    """Resolve a query to an ordered list of passages for the prompt."""

    def __init__(
        self,
        vector_store: _VectorStore,
        config: RAGConfig,
        *,
        llm_factory: Optional[Callable[[], _ChatModel]] = None,
    ) -> None:
        self.vector_store = vector_store
        self.config = config
        # Called lazily and only for the "llm" backend, so the default path never
        # constructs a chat client. The agent passes its cached non-reasoning LLM.
        self._llm_factory = llm_factory
        self._llm: Optional[_ChatModel] = None

        backend = (config.rerank_backend or "none").strip().lower()
        if backend not in BACKENDS:
            logger.warning(
                "Unknown RAG_RERANK_BACKEND %r; falling back to 'none' (valid: %s)",
                config.rerank_backend,
                ", ".join(BACKENDS),
            )
            backend = "none"
        self.backend = backend

    @property
    def candidate_k(self) -> int:
        """Size of the over-fetched candidate pool (never below top_k)."""
        return max(self.config.fetch_k, self.config.top_k)

    def retrieve(self, query: str) -> list[Document]:
        top_k = self.config.top_k
        if self.backend == "mmr":
            return self._mmr(query, top_k)
        if self.backend == "llm":
            return self._llm_rerank(query, top_k)
        # "none": exactly the original single-pass dense search.
        return self.vector_store.similarity_search(query, top_k)

    def _mmr(self, query: str, top_k: int) -> list[Document]:
        try:
            return self.vector_store.max_marginal_relevance_search(
                query,
                k=top_k,
                fetch_k=self.candidate_k,
                lambda_mult=self.config.mmr_lambda,
            )
        except Exception:
            logger.exception("MMR search failed; falling back to dense search")
            return self.vector_store.similarity_search(query, top_k)

    def _llm_rerank(self, query: str, top_k: int) -> list[Document]:
        docs = self.vector_store.similarity_search(query, self.candidate_k)
        if len(docs) <= top_k:
            return docs
        try:
            order = self._rank_with_llm(query, docs, top_k)
        except Exception:
            logger.exception("LLM rerank failed; falling back to dense order")
            return docs[:top_k]
        return [docs[i] for i in order[:top_k]]

    def _rank_with_llm(self, query: str, docs: list[Document], top_k: int) -> list[int]:
        llm = self._get_llm()
        listing = "\n".join(
            f"[{i}] {_snippet(doc.page_content)}" for i, doc in enumerate(docs, start=1)
        )
        prompt = RERANK_PROMPT.format(query=query, listing=listing, k=top_k)
        response = llm.invoke(prompt)
        text = getattr(response, "content", response)
        return _parse_order(str(text), len(docs))

    def _get_llm(self) -> _ChatModel:
        if self._llm is None:
            if self._llm_factory is None:
                raise RuntimeError("the 'llm' rerank backend requires an llm_factory")
            self._llm = self._llm_factory()
        return self._llm
