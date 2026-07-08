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


if __name__ == "__main__":
    unittest.main()
