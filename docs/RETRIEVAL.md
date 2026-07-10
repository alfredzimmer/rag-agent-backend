# Retrieval reranking

The agent resolves a query to prompt context in two stages (`src/rag/retrieval.py`):

1. **Over-fetch** a candidate pool of `RAG_FETCH_K` passages from Milvus (dense,
   cosine, HNSW `ef = RAG_SEARCH_EF`).
2. **Rerank** the pool down to `RAG_TOP_K` passages, using the backend selected by
   `RAG_RERANK_BACKEND`.

The default (`none`) skips stage 1 and does a single dense search at `top_k`, i.e.
byte-for-byte the original behaviour. Turning a reranker on is a config change with
no redeploy of new dependencies and no second model loaded onto the GPU.

## Why this exists

Recall of the live system was low (~48% corpus-wide, worse for multi-passage
questions). Re-chunking and a larger `ef`/`top_k` addressed the over-chunking and
graph-traversal causes, but two remained:

- **Near-duplicate crowding.** The corpus has near-duplicate chunks (multiple
  volume variants of the same transcript). At a fixed `top_k` they displace
  distinct evidence. `mmr` selects a diverse subset from a larger pool.
- **No relevance reordering.** Dense similarity alone is a coarse ranker. `llm`
  reorders the pool with a listwise judgement from the resident chat model —
  the on-prem stand-in for a cross-encoder, which a single 32 GB GPU already
  hosting the generator cannot also hold.

## Backends

| `RAG_RERANK_BACKEND` | Behaviour | Added latency |
| --- | --- | --- |
| `none` | Single dense search at `top_k`. | — |
| `mmr` | Over-fetch `fetch_k`, then Maximal Marginal Relevance (`RAG_MMR_LAMBDA`, 0 = max diversity, 1 = pure relevance) selects `top_k`. | one dense search over a larger pool |
| `llm` | Over-fetch `fetch_k`, then the resident non-reasoning model ranks the candidates; `top_k` kept. | one extra generation per query |

Every backend degrades safely: an MMR error or a garbled/failed LLM reply falls
back to plain dense order, so a reranker can never do worse than `none`.

## Tuning

- `RAG_FETCH_K` (default 40): candidate pool size. Bigger pool = more recall
  headroom for the reranker, more cost. Clamped up to `top_k`.
- `RAG_MMR_LAMBDA` (default 0.5): `mmr` relevance/diversity balance.
- `RAG_TOP_K` (default 10): passages passed to the generator.

## Measure before enabling

Do not flip production on intuition — measure. Point the eval at a server whose
API is running the candidate backend:

```bash
# on the server, with RAG_RERANK_BACKEND set for the API process under test
uv run python src/evaluation/evaluate_recall.py
```

Compare the reported recall against the `none` baseline on the same collection,
then set `RAG_RERANK_BACKEND` in the production env file and redeploy. Recall has
not yet been re-measured against `rag_documents_v2`; establishing that baseline is
the prerequisite for choosing a backend.
