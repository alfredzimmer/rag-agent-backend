from __future__ import annotations

import os


DEFAULT_CORS_ORIGINS = (
    "https://chat.edemi.org",
    "https://pis3.aempro.ca",
    "http://localhost:3000",
    "http://localhost:5173",
)


def get_cors_origins(value: str | None = None) -> list[str]:
    if value is None:
        value = os.getenv("CORS_ORIGINS")

    if value is None:
        return list(DEFAULT_CORS_ORIGINS)

    return [origin.strip() for origin in value.split(",") if origin.strip()]
