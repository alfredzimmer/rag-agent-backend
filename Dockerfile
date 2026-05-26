FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock readme.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

EXPOSE 9229

CMD ["uv", "run", "--frozen", "uvicorn", "pyapi.main:app", "--host", "0.0.0.0", "--port", "9229"]
