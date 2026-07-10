from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from rag.config import RAGConfig


class RagConfigTests(unittest.TestCase):
    def test_generation_defaults_reduce_reasoning_truncation_risk(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = RAGConfig()

        self.assertEqual(config.llm_num_predict, 8192)
        self.assertFalse(config.llm_reasoning)

    def test_reasoning_can_be_enabled_explicitly(self) -> None:
        with patch.dict(os.environ, {"RAG_LLM_REASONING": "true"}, clear=True):
            config = RAGConfig()

        self.assertTrue(config.llm_reasoning)

    def test_output_budget_can_be_overridden(self) -> None:
        with patch.dict(os.environ, {"RAG_LLM_NUM_PREDICT": "2048"}, clear=True):
            config = RAGConfig()

        self.assertEqual(config.llm_num_predict, 2048)

    def test_retrieval_defaults_preserve_single_pass_search(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = RAGConfig()

        self.assertEqual(config.rerank_backend, "none")
        self.assertEqual(config.fetch_k, 40)
        self.assertEqual(config.mmr_lambda, 0.5)

    def test_rerank_backend_can_be_configured(self) -> None:
        env = {"RAG_RERANK_BACKEND": "mmr", "RAG_FETCH_K": "60", "RAG_MMR_LAMBDA": "0.7"}
        with patch.dict(os.environ, env, clear=True):
            config = RAGConfig()

        self.assertEqual(config.rerank_backend, "mmr")
        self.assertEqual(config.fetch_k, 60)
        self.assertEqual(config.mmr_lambda, 0.7)


if __name__ == "__main__":
    unittest.main()
