# Edemi Backend

This repo contains the FastAPI service and Python RAG runtime for the agent. Read this, this is not ai generated.

```text
client
  -> Edemi FastAPI service
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

Verify that you have pulled the required Ollama models:
```bash
# Set up the containers; run this on server
docker compose -f infra/docker-compose.yaml up -d
```

and test it on `http://localhost:11434` using whatever method you wish.

Synchronize project virtual environment and packages using `uv`:
```bash
uv sync
```

Spin up the FastAPI service on the default port `9229`:
```bash
uv run python src/edemi_server/main.py
```

In a separate terminal tab, launch the interactive testing interface on port `8501`:
```bash
uv run streamlit run src/streamlit_app.py --server.port 8501
```

## Headsup
Be duely noted that this is designed to be a working prototype of a project, not one that is production ready. There's non-trivial legacy scaffolds/codes that are not purged; the components are constructed to work cohesively as a whole but not closely examined line-by-line; the error handling logic is basically non-existent and not even close to robust. In other words, this is vibe coded. Please be lenient towards the lack of coding talent in whoever wrote this line that you can spectate in git blame on the side (if you're using an editor to view this readme.md).
