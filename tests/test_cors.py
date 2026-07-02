from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from edemi_server.config import DEFAULT_CORS_ORIGINS, get_cors_origins


def make_client(cors_origins: list[str]) -> TestClient:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return TestClient(app)


class CorsConfigTests(unittest.TestCase):
    def test_default_origins_keep_development_hosts(self) -> None:
        self.assertEqual(get_cors_origins(), list(DEFAULT_CORS_ORIGINS))

    def test_configured_origins_are_trimmed(self) -> None:
        self.assertEqual(
            get_cors_origins(" https://chat.edemi.org, ,https://pis3.aempro.ca "),
            ["https://chat.edemi.org", "https://pis3.aempro.ca"],
        )

    def test_production_allowlist_rejects_localhost_preflight(self) -> None:
        client = make_client(
            get_cors_origins("https://chat.edemi.org,https://pis3.aempro.ca")
        )

        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_production_allowlist_accepts_chat_origin(self) -> None:
        client = make_client(
            get_cors_origins("https://chat.edemi.org,https://pis3.aempro.ca")
        )

        response = client.options(
            "/health",
            headers={
                "Origin": "https://chat.edemi.org",
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "https://chat.edemi.org",
        )


if __name__ == "__main__":
    unittest.main()
