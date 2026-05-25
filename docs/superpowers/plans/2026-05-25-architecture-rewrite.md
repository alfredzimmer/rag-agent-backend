# Industrial System Architecture Rewrite Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completely rewrite the ingestion and runtime architectures to achieve a decoupled, high-performance, industrial-grade system. Phase 1 establishes an asynchronous C++ & Redis ingestion pipeline. Phase 2 rebuilds the Python agentic runtime utilizing LangGraph's parallel execution (Fan-Out/Fan-In) and robust PostgreSQL state checkpointing.

**Architecture:** 
- **Phase 1 (Offline Ingestion):** Raw Data -> Redis Queue -> C++ Workers (Parse, Chunk, Embed via C++ inference) -> Vector DB.
- **Phase 2 (Online Agent Runtime):** User Query -> LangGraph Planner -> Parallel Fan-Out (Vector DB, Web Search, Internal API) -> State Reducer/Synthesizer -> Evaluator Node. State persistence via PostgreSQL.

**Tech Stack:** C++20, Redis, Milvus, ONNX Runtime/LibTorch (for C++ embeddings), Python, LangGraph, FastAPI, PostgreSQL (Checkpointer), `uv` (Python Package Manager).

---

## Part 1: C++ Ingestor Rewrite (`cpp-injestor` repo)

**Goal:** Evolve the current C++ ingestor from a "staging" worker to a fully autonomous pipeline that handles Parsing, Chunking, *and* Embedding before writing directly to the final Vector DB.

### Task 1.1: Integrate C++ Inference Engine for Embeddings
**Context:** Currently, `pyapi` handles embeddings. We must move this to C++.
- [ ] **Step 1: Update CMake configuration**
  Add dependencies for ONNX Runtime (or LibTorch) to support local execution of `qwen3` (dense) and `bge-m3` (sparse) models.
- [ ] **Step 2: Create `Embedder` interface**
  Define `src/Embedder.h` with methods `compute_dense` and `compute_sparse`.
- [ ] **Step 3: Implement ONNX/LibTorch execution**
  Implement the logic in `src/Embedder.cpp` to load the models and process chunked text into vectors.
- [ ] **Step 4: Write unit tests**
  Verify vector outputs against known expected dimensions.

### Task 1.2: Refactor Main Ingestion Pipeline
**Context:** The pipeline must now route chunks through the embedder before Milvus.
- [ ] **Step 1: Update `MilvusClient`**
  Modify `insert_chunks` to accept vector data alongside text and metadata, targeting the final production collection rather than a staging collection.
- [ ] **Step 2: Update `main.cpp` loop**
  Refactor the `while` loop: Consume Job -> Parse PDF -> Chunk -> **Embed** -> Push to Vector DB.
- [ ] **Step 3: Implement robust Redis error handling**
  Add connection retries and dead-letter queue (DLQ) logic for failed jobs.

---

## Part 2: Python Environment & Foundation (`pyapi` repo)

**Goal:** Modernize the Python environment and establish the database infrastructure for the new LangGraph architecture.

### Task 2.1: Migrate to `uv` Package Manager
**Context:** Replace `pip`/`requirements.txt` with `uv` for industrial-grade dependency management.
- [ ] **Step 1: Initialize `uv`**
  Run `uv init` in the project root.
- [ ] **Step 2: Port dependencies**
  Migrate required packages (FastAPI, LangGraph, psycopg, etc.) to `pyproject.toml`.
- [ ] **Step 3: Update `start_server.sh`**
  Modify the script to use `uv run api` (or equivalent uvicorn command).

### Task 2.2: Implement PostgreSQL Checkpointer Setup
**Context:** LangGraph requires a persistent state store.
- [ ] **Step 1: Configure Database Connection**
  Set up asynchronous SQLAlchemy/asyncpg connection utilities in `src/db/`.
- [ ] **Step 2: Integrate `AsyncPostgresSaver`**
  Initialize LangGraph's native Postgres checkpointer to persist the `StateGraph` at every super-step.

---

## Part 3: Agentic Runtime Rewrite (`pyapi` repo)

**Goal:** Rebuild `src/rag/graph.py` to utilize a parallel Fan-Out/Fan-In architecture instead of a linear Tool Node approach.

### Task 3.1: Define the New State Schema
**Context:** The state must support parallel worker results.
- [ ] **Step 1: Update `State` TypedDict**
  Add fields for `vector_results`, `web_results`, and `api_results`. Use `Annotated[list, operator.add]` for reducer patterns where necessary.

### Task 3.2: Implement the Planner Node
**Context:** The orchestrator that decides which parallel workers to invoke.
- [ ] **Step 1: Create `planner_node`**
  Implement an LLM call that analyzes the query and returns a structured plan (e.g., "Need VectorDB and Web Search").
- [ ] **Step 2: Implement conditional edge logic**
  Write the routing function that reads the plan and uses LangGraph's `Send` API to dispatch the required workers concurrently.

### Task 3.3: Implement Parallel Workers
**Context:** The independent nodes that fetch data.
- [ ] **Step 1: Create `vector_db_worker`**
  Refactor existing Milvus hybrid search logic into a dedicated node.
- [ ] **Step 2: Create `web_search_worker`**
  Implement a node utilizing a search API (e.g., Tavily or DuckDuckGo).
- [ ] **Step 3: Create `internal_api_worker`**
  Implement a stub/node for internal data retrieval.

### Task 3.4: Implement Reducer and Evaluator Nodes
**Context:** Synthesizing the parallel data streams.
- [ ] **Step 1: Create `synthesizer_node`**
  Implement the LLM call that takes all populated state fields (`vector_results`, `web_results`) and generates the final user response.
- [ ] **Step 2: Refactor `evaluator_node`**
  Ensure the existing evaluation logic runs *after* the synthesizer.

### Task 3.5: Wire the Graph
**Context:** Connect the nodes into the final architecture.
- [ ] **Step 1: Rebuild `StateGraph`**
  Connect: `START` -> `Planner` -> (Parallel Edges) -> `Workers` -> `Synthesizer` -> `Evaluator` -> `END`.
- [ ] **Step 2: Attach Checkpointer**
  Compile the graph passing the `AsyncPostgresSaver` initialized in Task 2.2.

---

## Part 4: API Integration & Testing

### Task 4.1: Update FastAPI Routes
**Context:** Ensure the web layer correctly interfaces with the new asynchronous graph.
- [ ] **Step 1: Refactor `RAGAgent` class**
  Update the wrapper class to initialize and stream from the newly compiled graph.
- [ ] **Step 2: Update `/chat` endpoint**
  Ensure the endpoint handles streaming responses and metadata correctly based on the new graph structure.

### Task 4.2: End-to-End System Test
**Context:** Verify Phase 1 and Phase 2 integration.
- [ ] **Step 1: Ingestion Test**
  Push a raw PDF to Redis, verify C++ worker processes it, embeds it, and writes to Milvus.
- [ ] **Step 2: Runtime Test**
  Send a query to the FastAPI endpoint, verify the LangGraph planner routes to the Vector DB worker, retrieves the newly ingested data, and synthesizes a response.
