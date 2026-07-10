from __future__ import annotations

import unittest

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag.config import RAGConfig
from rag.retrieval import Retriever, _parse_order


def _docs(n: int) -> list[Document]:
    return [Document(page_content=f"passage {i}", metadata={"i": i}) for i in range(n)]


class FakeStore:
    """Records how it was queried so tests can assert the retrieval strategy."""

    def __init__(self, docs: list[Document], *, mmr_error: bool = False) -> None:
        self.docs = docs
        self.mmr_error = mmr_error
        self.calls: list[tuple] = []

    def similarity_search(self, query: str, k: int) -> list[Document]:
        self.calls.append(("dense", k))
        return self.docs[:k]

    def max_marginal_relevance_search(
        self, query: str, k: int, fetch_k: int, lambda_mult: float
    ) -> list[Document]:
        self.calls.append(("mmr", k, fetch_k, lambda_mult))
        if self.mmr_error:
            raise RuntimeError("mmr not supported")
        # Pretend MMR drops odd-indexed near-duplicates.
        return [d for i, d in enumerate(self.docs) if i % 2 == 0][:k]


class FakeLLM:
    def __init__(self, output: str | Exception) -> None:
        self.output = output
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> AIMessage:
        self.prompts.append(prompt)
        if isinstance(self.output, Exception):
            raise self.output
        return AIMessage(content=self.output)


def _config(**overrides) -> RAGConfig:
    base = dict(top_k=3, fetch_k=10, rerank_backend="none", mmr_lambda=0.5)
    base.update(overrides)
    return RAGConfig(**base)


class NoneBackendTests(unittest.TestCase):
    def test_default_backend_is_single_pass_dense_at_top_k(self) -> None:
        store = FakeStore(_docs(20))
        retriever = Retriever(store, _config())

        result = retriever.retrieve("q")

        # Exactly the original behaviour: one dense search, no over-fetch.
        self.assertEqual(store.calls, [("dense", 3)])
        self.assertEqual([d.metadata["i"] for d in result], [0, 1, 2])

    def test_unknown_backend_falls_back_to_none(self) -> None:
        store = FakeStore(_docs(20))
        retriever = Retriever(store, _config(rerank_backend="cohere"))

        retriever.retrieve("q")

        self.assertEqual(retriever.backend, "none")
        self.assertEqual(store.calls, [("dense", 3)])


class MMRBackendTests(unittest.TestCase):
    def test_mmr_over_fetches_and_selects_top_k(self) -> None:
        store = FakeStore(_docs(20))
        retriever = Retriever(store, _config(rerank_backend="mmr"))

        result = retriever.retrieve("q")

        self.assertEqual(store.calls, [("mmr", 3, 10, 0.5)])
        # Odd near-duplicates removed by the fake MMR.
        self.assertEqual([d.metadata["i"] for d in result], [0, 2, 4])

    def test_candidate_pool_never_smaller_than_top_k(self) -> None:
        store = FakeStore(_docs(20))
        retriever = Retriever(store, _config(rerank_backend="mmr", top_k=8, fetch_k=2))

        retriever.retrieve("q")

        # fetch_k=2 < top_k=8, so the pool is clamped up to top_k.
        self.assertEqual(store.calls, [("mmr", 8, 8, 0.5)])

    def test_mmr_failure_falls_back_to_dense(self) -> None:
        store = FakeStore(_docs(20), mmr_error=True)
        retriever = Retriever(store, _config(rerank_backend="mmr"))

        result = retriever.retrieve("q")

        self.assertEqual(store.calls[0][0], "mmr")
        self.assertEqual(store.calls[1], ("dense", 3))
        self.assertEqual([d.metadata["i"] for d in result], [0, 1, 2])


class LLMBackendTests(unittest.TestCase):
    def test_llm_rerank_reorders_candidates_and_trims(self) -> None:
        store = FakeStore(_docs(20))
        llm = FakeLLM("passages ranked: 5, 2, 9")
        retriever = Retriever(
            store, _config(rerank_backend="llm"), llm_factory=lambda: llm
        )

        result = retriever.retrieve("q")

        # Over-fetch the candidate pool, then keep the model's top 3 (1-based).
        self.assertEqual(store.calls, [("dense", 10)])
        self.assertEqual([d.metadata["i"] for d in result], [4, 1, 8])

    def test_partial_ranking_keeps_full_top_k_without_dropping_docs(self) -> None:
        store = FakeStore(_docs(20))
        llm = FakeLLM("only one good: 4")  # model names a single passage
        retriever = Retriever(
            store, _config(rerank_backend="llm"), llm_factory=lambda: llm
        )

        result = retriever.retrieve("q")

        # Named passage first, then dense order fills the rest — still top_k docs.
        self.assertEqual(len(result), 3)
        self.assertEqual([d.metadata["i"] for d in result], [3, 0, 1])

    def test_llm_error_falls_back_to_dense_order(self) -> None:
        store = FakeStore(_docs(20))
        llm = FakeLLM(RuntimeError("model offline"))
        retriever = Retriever(
            store, _config(rerank_backend="llm"), llm_factory=lambda: llm
        )

        result = retriever.retrieve("q")

        self.assertEqual([d.metadata["i"] for d in result], [0, 1, 2])

    def test_no_over_fetch_when_pool_not_larger_than_top_k(self) -> None:
        # Only 2 candidates exist but top_k is 3: nothing to rerank.
        store = FakeStore(_docs(2))
        llm = FakeLLM("2, 1")
        retriever = Retriever(
            store, _config(rerank_backend="llm"), llm_factory=lambda: llm
        )

        result = retriever.retrieve("q")

        self.assertEqual(llm.prompts, [])  # reranker skipped
        self.assertEqual([d.metadata["i"] for d in result], [0, 1])


class ParseOrderTests(unittest.TestCase):
    def test_dedupes_clamps_and_completes_permutation(self) -> None:
        # 0 and 12 are out of range (1..5); 3 is duplicated; 5 is omitted.
        self.assertEqual(_parse_order("3, 3, 1, 0, 12, 4", 5), [2, 0, 3, 1, 4])

    def test_garbage_reply_yields_original_order(self) -> None:
        self.assertEqual(_parse_order("no idea, sorry", 4), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
