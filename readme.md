# Edemi Backend

Edemi Backend is the FastAPI, RAG, ingestion, and observability runtime for Edemi.

```text
Frontend
  -> FastAPI API
      -> PostgreSQL for users, conversations, and LangGraph checkpoints
      -> Redis Streams for durable ingestion jobs
      -> Milvus for scoped document retrieval
      -> Ollama for chat and embeddings

Document upload
  -> Redis Stream
      -> Python ingestion worker
          -> parse -> chunk -> embed -> Milvus

API and worker
  -> OpenTelemetry Collector
      -> Tempo traces
      -> Prometheus metrics
      -> Loki logs
      -> Grafana
```

## Setup

Copy the environment template and configure its secrets:

```bash
cp infra/env.example .env
uv sync
```

Start the complete application, infrastructure, tools, and observability stack:

```bash
docker compose \
  --profile observability \
  --profile tools \
  -f infra/docker-compose.yaml \
  up --build -d
```

The main local endpoints are:

- API: `http://localhost:9229`
- API schema: `http://localhost:9229/docs`
- Grafana: `http://localhost:3001`
- Prometheus: `http://localhost:9090`
- OpenTelemetry health: `http://localhost:13133`

## Ingestion API

Create a conversation, then upload a document into that conversation's retrieval scope:

```text
POST /api/ingestion/documents?conversation_id=<uuid>
GET  /api/ingestion/jobs/<job_id>
```

Supported document types are PDF, DOCX, Markdown, and plain text. Uploads are asynchronous and return `202 Accepted`. The worker uses Redis consumer groups, retries failed jobs, recovers stale pending jobs, and moves terminal failures to the dead-letter stream.

Each Milvus chunk includes a `scope_id`. Retrieval only searches the active conversation scope plus the configured global scope.

## Telemetry

The API and worker export traces, metrics, and logs over OTLP. Logs are JSON on stdout and include `trace_id`, `span_id`, `job_id`, `conversation_id`, and `document_id` when available.

See `skills.md` for the operational debugging workflow and queue commands.

## CI/CD

`.github/workflows/deploy.yml` verifies pull requests and deploys `main` to the protected GitHub `production` environment. It synchronizes the repository to the production host, checks that Ollama is healthy (and attempts to start it when necessary), pulls updated infrastructure images, builds the application image on the server, and reconciles every service in `infra/docker-compose.yaml`.

Configure these GitHub environment secrets:

- `PRODUCTION_HOST`
- `PRODUCTION_USER`
- `PRODUCTION_SSH_KEY`
- `PRODUCTION_KNOWN_HOSTS`
- `TS_OAUTH_CLIENT_ID`
- `TS_OAUTH_SECRET`

Configure these GitHub environment variables:

- `PRODUCTION_DEPLOY_PATH`, for example `/opt/edemi-backend`
- `PRODUCTION_ENV_FILE`, for example `/etc/edemi/edemi.env`
- `PRODUCTION_SSH_PORT`, default `22`
- `PRODUCTION_OLLAMA_HEALTH_URL`, default `http://127.0.0.1:11434/api/tags`

Set `PRODUCTION_HOST` to the server's Tailscale address. Configure the Tailscale OAuth client with the `auth_keys` write scope and `tag:ci`. Configure a required reviewer on the `production` environment before enabling the workflow. Create the server environment file from `infra/env.production.example` and keep it outside `PRODUCTION_DEPLOY_PATH` so source synchronization cannot delete it. The deployment user must be able to run Docker, start Ollama, and write to `PRODUCTION_DEPLOY_PATH`; the host also needs `curl`, `flock`, and `rsync`.
