import os
from qdrant_client import models 
from modules.embedder import compute_dense_vec, compute_sparse_vec

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "test")
"""
Assumes db client is qdrant instance
  Text should be raw user query text like "Tell me the fault tolerance of xxx"
  When implementing this method ->
  hybrid is assumed to run dense + SPLADE sparse vector queries respectively
  k is the number of results to be fetched in each category
  return type is a pair of lists of ScorePoint obj from Qdrant : dense + sparse
"""
def hybrid_retrieve(text, client, k = 50):

  dense_vec = compute_dense_vec(text)
  sparse_indices, sparse_values = compute_sparse_vec(text)

  # result is a pair of result lists : one for dense and one for sparse
  result = client.search_batch(
    collection_name = COLLECTION_NAME,
    requests = [
      models.SearchRequest(
        vector = models.NamedVector(
          name = "text-dense",
          vector = dense_vec,
        ),
        with_payload=True,
        limit = k,
      ),
      models.SearchRequest(
        vector = models.NamedSparseVector(
          name = "text-sparse",
          vector = models.SparseVector(
            indices = sparse_indices,
            values = sparse_values,
          ),
        ),
        with_payload=True,
        limit = k,
      ),
    ],
  )
  
  
  return result # -> [[ScoredPoint, ScoredPoint, ...], [ScoredPoint, ScoredPoint]]


