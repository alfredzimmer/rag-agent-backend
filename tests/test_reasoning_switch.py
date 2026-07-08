from __future__ import annotations

import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from rag.config import RAGConfig
from rag.agent import RAGAgent
from rag_agent_server.main import ChatRequest


class ReasoningSwitchTests(unittest.TestCase):
    def test_chat_request_accepts_reasoning_switch(self) -> None:
        conversation_id = uuid4()
        request = ChatRequest(
            query="Check access clearance",
            conversation_id=conversation_id,
            reasoning=True,
        )

        self.assertEqual(request.conversation_id, conversation_id)
        self.assertTrue(request.reasoning)

    def test_chat_request_defaults_to_backend_config(self) -> None:
        request = ChatRequest(query="Check access clearance", conversation_id=uuid4())

        self.assertIsNone(request.reasoning)

    def test_agent_caches_one_llm_per_reasoning_mode(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("rag.agent.create_milvus_store", return_value=object()),
            patch("rag.agent.ChatOllama") as chat_ollama,
        ):
            chat_ollama.side_effect = lambda **kwargs: kwargs
            agent = RAGAgent(RAGConfig())

            no_reasoning = agent._llm_for(False)
            reasoning = agent._llm_for(True)
            no_reasoning_again = agent._llm_for(False)

        self.assertFalse(no_reasoning["reasoning"])
        self.assertTrue(reasoning["reasoning"])
        self.assertIs(no_reasoning, no_reasoning_again)
        self.assertEqual(chat_ollama.call_count, 2)


if __name__ == "__main__":
    unittest.main()
