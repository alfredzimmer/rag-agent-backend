from __future__ import annotations

import json
import os
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PUBLIC_HEALTH_URL = os.getenv("RAG_AGENT_E2E_PUBLIC_HEALTH_URL")
CHAT_ORIGIN = os.getenv("RAG_AGENT_E2E_CHAT_ORIGIN", "https://chat.rag-agent.example")
BLOCKED_ORIGIN = os.getenv("RAG_AGENT_E2E_BLOCKED_ORIGIN", "http://localhost:3000")


@unittest.skipUnless(
    PUBLIC_HEALTH_URL,
    "set RAG_AGENT_E2E_PUBLIC_HEALTH_URL to run public-edge E2E tests",
)
class PublicEdgeEndToEndTests(unittest.TestCase):
    def test_public_health_endpoint_returns_ok(self) -> None:
        with urlopen(PUBLIC_HEALTH_URL, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(response.status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "rag-agent-api")

    def test_public_edge_accepts_chat_origin_preflight(self) -> None:
        request = Request(
            PUBLIC_HEALTH_URL,
            method="OPTIONS",
            headers={
                "Origin": CHAT_ORIGIN,
                "Access-Control-Request-Method": "GET",
            },
        )

        with urlopen(request, timeout=15) as response:
            allow_origin = response.headers.get("access-control-allow-origin")

        self.assertEqual(response.status, 200)
        self.assertEqual(allow_origin, CHAT_ORIGIN)

    def test_public_edge_rejects_localhost_origin_preflight(self) -> None:
        request = Request(
            PUBLIC_HEALTH_URL,
            method="OPTIONS",
            headers={
                "Origin": BLOCKED_ORIGIN,
                "Access-Control-Request-Method": "GET",
            },
        )

        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=15)

        self.assertEqual(error.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
