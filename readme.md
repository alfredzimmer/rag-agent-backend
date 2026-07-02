# RAG Agent Backend

A minimal RAG agent: FastAPI serves a streaming chat endpoint, retrieval comes
from Milvus, and generation (plus embeddings) comes from Ollama. Nothing else.

```text
Frontend
  -> FastAPI API (streaming NDJSON)
      -> Milvus  (dense retrieval, top-k)
      -> Ollama  (embeddings + chat model)
```

Conversation history is kept in memory per conversation id and is lost on
restart.

## Network layout

| From | To | Address |
| --- | --- | --- |
| API container | Milvus | `http://milvus:19530` (compose network) |
| API container | Ollama | `http://host.docker.internal:11434` (host gateway) |
| Host (dev API, tooling) | Milvus | `http://localhost:19530` (loopback publish) |
| Host / Funnel | API | `127.0.0.1:9229` (loopback publish) |

The compose file pins `MILVUS_URI` and `OLLAMA_HOST` for the container, so the
`.env` file only needs host-oriented values. All published ports bind to
loopback; nothing is exposed publicly by the stack itself.

## Setup

```bash
cp infra/env.example .env
uv sync
docker compose -f infra/docker-compose.yaml up --build -d
```

Ollama must be running on the host with the configured models pulled
(`RAG_LLM_MODEL`, `DENSE_EMBEDDING_MODEL`).

To run the API on the host instead of in Docker (dev loop):

```bash
docker compose -f infra/docker-compose.yaml up -d etcd minio milvus
uv run rag-agent-api
```

## API

- `GET /health` — returns 503 with per-dependency status when Milvus or Ollama
  is unreachable.
- `GET /api/agent/conversation/create` — new conversation id.
- `GET /api/agent/conversation/list`
- `POST /api/agent/conversation/chat` — `{query, conversation_id}`, streams
  NDJSON `ChatResponse` events (retrieval, reasoning, text deltas, completion).
- `POST /api/agent/conversation/interrupt` — `{conversation_id}`.
- `GET /api/agent/conversation/history?conversation_id=<uuid>`
- `DELETE /api/agent/conversation/clear` — `{conversation_id}`.

Schema: `http://localhost:9229/docs`.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

## Production

`infra/deploy.sh` validates the env file, checks Ollama health on the host,
builds the image, reconciles the compose stack, and smoke-tests
`http://127.0.0.1:9229/health`. Create the server env file from
`infra/env.production.example` and keep it outside the deploy path.

Public exposure is a one-time host concern, outside this repository:

```bash
tailscale funnel --bg 9229
tailscale funnel status
```

The Funnel configuration persists across deploys and reboots, so the deploy
script does not manage it. Use the HTTPS URL printed by
`tailscale funnel status` as the public API base URL.

`.github/workflows/deploy.yml` verifies pull requests (tests, compose config,
Dockerfile, deploy script) and deploys `main` over Tailscale SSH, then
smoke-tests `PRODUCTION_PUBLIC_HEALTH_URL`.
