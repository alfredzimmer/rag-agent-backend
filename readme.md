# Python Backend for RAG-Agent

This repo contains the FastAPI service and Python RAG runtime for the agent.

The service is not just a web API. It coordinates a small self-hosted stack:

```text
client
  -> pyapi FastAPI service
      -> Postgres for LangGraph checkpoints and memory store
      -> Milvus for domain-specific vector search
          -> etcd for Milvus metadata
          -> MinIO for Milvus object storage
      -> Ollama for local LLM and embedding models
```

## Repository Layout

```text
src/pyapi/                 FastAPI app and route handlers
src/rag/                   RAG agent, graph, retrieval, memory, and ingestion utilities
infra/compose.dev.yaml     Local/dev dependencies
infra/compose.prod.yaml    Single-server production stack
infra/dev.env.example      Local development environment template
infra/prod.env.example     Production environment template
Dockerfile                 pyapi application image
pyproject.toml             uv project metadata
uv.lock                    locked Python dependency graph
```

`src/rag/main-scripts/pdf_to_db.py` is a legacy PDF-to-Milvus pipeline. Treat it carefully; the long-term direction is to move ingestion into a separate worker/pipeline.

## What Each Container Does

`pyapi` is the Python API and agent runtime. It receives chat requests, builds the LangGraph agent, calls tools, streams responses, and stores conversation state.

`postgres` stores LangGraph checkpoints and long-term memory data. Without it, the agent cannot initialize.

`milvus` stores domain document embeddings for RAG retrieval. Without it, the agent has no domain-specific knowledge base.

`etcd` is required by Milvus. It stores Milvus metadata.

`minio` is required by Milvus. It stores Milvus segment and index data.

`ollama` serves the local LLM and embedding models used by the agent.

`attu` is an optional Milvus web UI. It is useful for debugging collections, but it is not required for runtime.

## Environment

Use different env files for dev and prod because the hostnames are different.

In dev, `pyapi` runs on your host machine, so it reaches dependencies through `localhost`.

In prod, `pyapi` runs inside Docker, so it reaches dependencies through Docker service names like `postgres`, `milvus`, and `ollama`.

For local development:

```bash
cp infra/dev.env.example infra/dev.env
cp infra/dev.env.example .env
```

For production:

```bash
cp infra/prod.env.example infra/prod.env
```

Edit the copied env files, especially:

```text
POSTGRES_PASSWORD
PG_URI
MINIO_ROOT_PASSWORD
RAG_COLLECTION_NAME
RAG_LLM_MODEL
```

If you change `POSTGRES_USER`, `POSTGRES_PASSWORD`, or `POSTGRES_DB`, update `PG_URI` to match.

## Local Development

For local development, run the databases and model server in Docker, then run `pyapi` with `uv` on the host.

Start dependencies:

```bash
docker compose --env-file infra/dev.env -f infra/compose.dev.yaml up -d
```

Install Python dependencies:

```bash
uv sync --python 3.13
```

Run the API locally:

```bash
uv run uvicorn pyapi.main:app --host 0.0.0.0 --port 9229 --reload
```

Or use:

```bash
./start_server.sh
```

Open the API:

```text
http://localhost:9229
http://localhost:9229/docs
```

Optional Milvus UI:

```bash
docker compose --env-file infra/dev.env -f infra/compose.dev.yaml --profile tools up -d attu
```

Then open:

```text
http://localhost:3030
```

## Ollama Models

The agent expects these models by default:

```text
qwen3:30b-instruct
qwen3:8b
qwen3-embedding:8b
nomic-embed-text
```

Pull them into the Ollama container:

```bash
docker compose --env-file infra/dev.env -f infra/compose.dev.yaml exec ollama ollama pull qwen3:30b-instruct
docker compose --env-file infra/dev.env -f infra/compose.dev.yaml exec ollama ollama pull qwen3:8b
docker compose --env-file infra/dev.env -f infra/compose.dev.yaml exec ollama ollama pull qwen3-embedding:8b
docker compose --env-file infra/dev.env -f infra/compose.dev.yaml exec ollama ollama pull nomic-embed-text
```

For production, use the same pulls against the prod compose file:

```bash
docker compose --env-file infra/prod.env -f infra/compose.prod.yaml exec ollama ollama pull qwen3:30b-instruct
docker compose --env-file infra/prod.env -f infra/compose.prod.yaml exec ollama ollama pull qwen3:8b
docker compose --env-file infra/prod.env -f infra/compose.prod.yaml exec ollama ollama pull qwen3-embedding:8b
docker compose --env-file infra/prod.env -f infra/compose.prod.yaml exec ollama ollama pull nomic-embed-text
```

If the server does not have enough memory/GPU for the 30B model, set a smaller model in your env file:

```text
RAG_LLM_MODEL=qwen3:8b
```

## Production On Your Own Server

For your current infra model, production means Docker Compose runs the whole stack on your server.

Start production:

```bash
docker compose --env-file infra/prod.env -f infra/compose.prod.yaml up -d --build
```

View logs:

```bash
docker compose --env-file infra/prod.env -f infra/compose.prod.yaml logs -f pyapi
```

Stop services without deleting data:

```bash
docker compose --env-file infra/prod.env -f infra/compose.prod.yaml down
```

Do not run `down -v` in production unless you intentionally want to delete all persisted database, Milvus, MinIO, and Ollama model data.

Production compose exposes only `pyapi` by default:

```text
http://SERVER_IP:9229
```

Postgres, Milvus, MinIO, and Ollama are internal Docker services in prod. Do not expose them publicly.

## Data Persistence

Docker volumes store the important state:

```text
postgres_data   conversation checkpoints and memory store
milvus_data     Milvus vector database data
etcd_data       Milvus metadata
minio_data      Milvus object/index storage
ollama_data     downloaded Ollama models
```

These volumes are the system's data. Back them up before upgrades or migrations.

## Domain Knowledge Setup

Milvus being up is not enough. The collection also has to contain embedded domain documents.

The runtime defaults to:

```text
RAG_COLLECTION_NAME=HeaderInContentTrial
MILVUS_DB=rag1
```

If that database/collection is empty or missing, the API may still run, but the agent will not have useful domain-specific knowledge.

Current ingestion options:

```bash
uv run python src/rag/main-scripts/pdf_to_db.py src/rag/public/IEEE1584-2018-31-36.pdf ieee HeaderInContentTrial
```

That script is legacy and may be replaced by a separate ingestion worker.

## Updating Python Dependencies

Add dependencies with `uv`:

```bash
uv add package-name
```

Refresh the lockfile after manual `pyproject.toml` edits:

```bash
uv lock
```

Rebuild the production image after dependency changes:

```bash
docker compose --env-file infra/prod.env -f infra/compose.prod.yaml up -d --build pyapi
```

## Common Failure Modes

If `pyapi` does not start, check:

```bash
docker compose --env-file infra/prod.env -f infra/compose.prod.yaml logs pyapi
```

Typical causes:

```text
PG_URI does not match Postgres credentials
Ollama is running but required models are not pulled
Milvus is running but the target database/collection is missing
The host does not have enough memory/GPU for the configured LLM
```

The app currently initializes the full RAG agent during FastAPI startup. That means dependency failures can prevent the API from booting at all. A future hardening step should split basic health from dependency readiness.

## Deepeval Commands

```bash
deepeval set-ollama [model_name]
deepeval unset-ollama
```
