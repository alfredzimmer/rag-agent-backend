# Python Backend for RAG-Agent

This repo contains the FastAPI service and Python RAG runtime for the agent. Read this, this is not ai generated.

```text
client
  -> pyapi FastAPI service
      -> Postgres for LangGraph checkpoints and memory store
      -> Milvus for domain-specific vector search
          -> etcd for Milvus metadata
          -> MinIO for Milvus object storage
      -> Ollama for local LLM and embedding models
```

`src/rag/main-scripts/pdf_to_db.py` is a legacy PDF-to-Milvus pipeline. Consider using the cpp-ingestor if all possible.

## Setup

We feel like explaining the system design decision more than instructing how to use them.

We used docker to host Milvus (including minio and etcd) and postgres which makes them easier to manage and the setup proces more straightforward.

We used uv for package management so theres no point in containerizing it. It would make Hot Module Replacement more complicated.

We run Ollama models natively on the server due to the sole reason that we want the want to take advantage of the optimisations using CUDA. We run two models on Ollama concurrently: the main model Qwen3.6 and the embedding model for online database retrieval. If you want a more thorough speed boost consider changing Ollama to vLLM, despite it is allegedly harder to set up.

That being said, it doesn't hurt to much to document their uses.

```bash
# Set up the containers; run this on server
docker compose -f infra/docker-compose.yaml up -d
```

This compose file should handle port forwarding to your machine automatically.

The ollama should be up an running as of the time we write this readme.

You will then need to start `ollama serve` on server and forward the port to your machine using
```bash
ssh -L 11434:localhost:11434 ziyutecc_ai_wsl@ziyutecc-ai
```
