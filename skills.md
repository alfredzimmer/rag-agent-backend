# Agent Operations Guide

This file is the debugging contract for agents working on Edemi Backend.

## Architecture Rules

- The API and ingestion worker are separate processes in the same repository.
- Redis Streams is the only ingestion queue. Do not add Redis lists or in-process background ingestion.
- Milvus uses one production schema. Do not add staging-collection compatibility branches.
- Documents are scoped with `scope_id`; retrieval searches the conversation scope and `RAG_GLOBAL_SCOPE`.
- Production retrieval uses database `rag1` and collection `HeaderInContentTrial`.
- OpenTelemetry is the primary debugging path. Do not add ad hoc debug files.
- Do not restore deleted Qdrant, synthetic-data, session fallback, C++ payload, or token compatibility code.

## Start The System

```bash
docker compose -f infra/docker-compose.yaml --profile observability up -d
uv run edemi-api
uv run edemi-ingestion-worker
```

For code-only tests that should not export telemetry:

```bash
OTEL_SDK_DISABLED=true uv run python -m unittest discover -s tests
```

## Remote Port Forwarding

When services run on `ai-server`, forward the ports you need:

```bash
ssh -N \
  -L 19530:127.0.0.1:19530 \
  -L 5433:127.0.0.1:5433 \
  -L 6380:127.0.0.1:6380 \
  -L 11434:127.0.0.1:11434 \
  -L 4317:127.0.0.1:4317 \
  -L 13133:127.0.0.1:13133 \
  -L 3001:127.0.0.1:3001 \
  -L 9090:127.0.0.1:9090 \
  ai-server -f
```

## Trace A Request

1. Open Grafana at `http://localhost:3001`.
2. In Loki, filter by `service_name` and search for `job_id`, `conversation_id`, or `document_id`.
3. Open the derived `TraceID` link to view the request in Tempo.
4. Follow spans from `edemi-api` through `ingestion.process_job` in `edemi-ingestion-worker`.
5. Check Prometheus metrics beginning with `edemi_ingestion_`.

Service names:

- `edemi-api`
- `edemi-ingestion-worker`

Collector health:

```bash
curl -fsS http://localhost:13133/
```

## Debug An Ingestion Job

API status:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:9229/api/ingestion/jobs/$JOB_ID
```

Queue and consumer-group health:

```bash
redis-cli -p 6380 XINFO GROUPS edemi:ingestion:jobs
redis-cli -p 6380 XPENDING edemi:ingestion:jobs edemi-ingestion-workers
redis-cli -p 6380 XRANGE edemi:ingestion:dead-letter - + COUNT 20
redis-cli -p 6380 GET edemi:ingestion:job:$JOB_ID
```

Interpretation:

- `queued`: accepted by the API and waiting for a consumer.
- `processing`: claimed by a worker.
- `retrying`: a new stream entry was created with an incremented attempt.
- `completed`: chunks were written to Milvus.
- `duplicate`: the same document/configuration/scope was already completed.
- `failed`: retry budget exhausted; inspect the dead-letter stream and correlated trace.

Do not manually acknowledge or delete pending messages until the trace and job state have been inspected. Stale jobs are recovered with `XAUTOCLAIM`.

## Legacy Milvus Migration

The C++ ingestor's `default.ingestion_staging` collection was migrated on
June 24, 2026. The migration produced 838,340 unique global-scope chunks in
`rag1.HeaderInContentTrial`; 129,860 exact duplicate staging rows were omitted.

The migration checkpoint is stored on the production server at:

```text
/home/ziyutecc_ai_wsl/edemi-backend/.deploy/legacy-milvus-migration.json
```

Use `tools/migrate_legacy_milvus.py` only for an intentional rebuild from the
preserved legacy Docker volumes. It is resumable, requires an empty destination
or matching checkpoint, and never modifies the source collection.

## Useful Correlation Fields

- `job_id`: ingestion lifecycle
- `document_id`: SHA-256 of the uploaded document
- `conversation_id` / `scope_id`: retrieval isolation
- `user_id`: ownership
- `trace_id`: cross-service execution
- `ingestion_job_id`, `chunk_index`, `page`: stored Milvus metadata

## Validation Before Handoff

```bash
OTEL_SDK_DISABLED=true PYTHONPATH=src .venv/bin/python -m compileall -q src tests tools
UV_CACHE_DIR=.uv-cache uv lock --check
docker compose -f infra/docker-compose.yaml --profile observability config --quiet
UV_CACHE_DIR=.uv-cache uv build --out-dir /tmp/edemi-backend-build
```

If external services are unavailable, report exactly which integration test could not run. Do not insert a fallback implementation.
