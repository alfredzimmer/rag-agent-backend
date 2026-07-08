# Ingestion Runbook

This project stores retrieval data in Milvus. The API reads these environment
variables at startup:

```bash
MILVUS_URI=http://localhost:19530   # host-side tooling
MILVUS_DB=rag2
RAG_COLLECTION_NAME=rag_documents_v2
DENSE_EMBEDDING_MODEL=nomic-embed-text:latest
```

Inside Docker, `infra/docker-compose.yaml` overrides `MILVUS_URI` to
`http://milvus:19530`; the database and collection still come from the
production env file.

## 2026-07-07 Cutover Plan

Previous production target from `infra/env.production.example`:

```bash
MILVUS_URI=http://localhost:19530
MILVUS_DB=rag1
RAG_COLLECTION_NAME=HeaderInContentTrial
```

New production target:

```bash
MILVUS_URI=http://localhost:19530
MILVUS_DB=rag2
RAG_COLLECTION_NAME=rag_documents_v2
```

Actual backup created during the 2026-07-07 cutover:

```bash
/home/ziyutecc_ai_wsl/.config/rag-agent/backups/milvus-rag1-HeaderInContentTrial-20260707T185109Z
```

Before changing the production env file, back up the old Milvus state on
`ai-server`. A full standalone Milvus restore needs all three compose volumes:

```bash
ssh ai-server
cd /home/ziyutecc_ai_wsl/rag-agent-backend
mkdir -p ~/.config/rag-agent/backups/milvus-20260707
docker run --rm \
  -v rag-agent-backend_milvus_data:/volume:ro \
  -v "$HOME/.config/rag-agent/backups/milvus-20260707:/backup" \
  alpine sh -c 'cd /volume && tar czf /backup/milvus_data.tgz .'
docker run --rm \
  -v rag-agent-backend_minio_data:/volume:ro \
  -v "$HOME/.config/rag-agent/backups/milvus-20260707:/backup" \
  alpine sh -c 'cd /volume && tar czf /backup/minio_data.tgz .'
docker run --rm \
  -v rag-agent-backend_etcd_data:/volume:ro \
  -v "$HOME/.config/rag-agent/backups/milvus-20260707:/backup" \
  alpine sh -c 'cd /volume && tar czf /backup/etcd_data.tgz .'
```

Record the active connection details beside the backup:

```bash
cat > ~/.config/rag-agent/backups/milvus-20260707/connection.env <<'EOF'
MILVUS_URI=http://localhost:19530
MILVUS_DB=rag1
RAG_COLLECTION_NAME=HeaderInContentTrial
DEPLOY_PATH=/home/ziyutecc_ai_wsl/rag-agent-backend
PRODUCTION_ENV=/home/ziyutecc_ai_wsl/.config/rag-agent/production.env
EOF
```

Deploy the updated code from the workstation:

```bash
./infra/push-deploy.sh
```

Then ingest the source folder on `ai-server` into the new DB:

```bash
cd /home/ziyutecc_ai_wsl/rag-agent-backend
uv sync --group ingest
MILVUS_URI=http://localhost:19530 \
MILVUS_DB=rag2 \
RAG_COLLECTION_NAME=rag_documents_v2 \
uv run --group ingest rag-ingest \
  --root "$HOME/RAG Knowledge Codes" \
  --db rag2 \
  --collection rag_documents_v2 \
  --drop-old \
  --pdf-parser text \
  --manifest "$HOME/.config/rag-agent/ingestion/ingest-rag2-20260707.json"
```

After ingestion succeeds, update the production env file to `MILVUS_DB=rag2`
and `RAG_COLLECTION_NAME=rag_documents_v2`, then rerun `./infra/push-deploy.sh`
so the API reconnects to the new database.

Actual ingestion result:

```text
Source root: /home/ziyutecc_ai_wsl/RAG Knowledge Codes
Manifest: /home/ziyutecc_ai_wsl/.config/rag-agent/ingestion/ingest-rag2-20260707.json
Target: rag2/rag_documents_v2
Parser: pymupdf text mode
Files discovered: 35 supported files
Files ingested: 30
Files skipped: 5
Chunks/entities written: 31,333
```

Skipped because parsed text failed the quality filter:

```text
IEEE/LEFT-OUT IEEE Std 241-1990 .pdf
NFPA/NFPA 101-2024.pdf
NFPA/NFPA 780-2020.pdf
NFPA/NFPA 99-2024.pdf
NFPA/NFPA-25-2020.pdf
```

## 2026-07-08 docx ingestion (North America Electrical Design series)

The old `rag1/HeaderInContentTrial` collection carried 95 docx documents
(707,614 line-level chunks, ~84% of the collection); the 2026-07-07 cutover
dropped them. This run restores that coverage in the serving collection using
the structure-aware docx chunker (`src/rag/ingest_docx.py`), appending to the
existing collection — deliberately **no `--drop-old`**:

```bash
cd /home/ziyutecc_ai_wsl/rag-agent-backend
uv sync --group ingest
uv run --group ingest rag-ingest \
  --root "$HOME/North-America-Electrical-Design-Series/Deduplicated English Content (for ingestion)" \
  --db rag2 \
  --collection rag_documents_v2 \
  --manifest "$HOME/.config/rag-agent/ingestion/ingest-rag2-docx-20260708.json"
```

Actual ingestion result:

```text
Files ingested: 29 (28 .docx + DEDUP_REPORT.md), 0 skipped
Chunks written: 2,872 (avg 1,316 chars, 0.1% garbage)
Collection total after append: 34,205 entities (31,333 PDF + 2,872)
```

Chunks are typed (`section_type`: summary / qa / terminology / transcript /
body) with a `{volume} › {video summary} › {section}` header in text and
metadata. To re-do this corpus, delete first with a Milvus expr on
`source_dir` or `source_ext == ".docx"`, then re-run.
