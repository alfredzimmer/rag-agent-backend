from typing import Dict, List
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from FlagEmbedding import BGEM3FlagModel
from pymilvus import model


from threading import Lock


class BGEEmbedder(BaseSparseEmbedding):  # inherit from BaseSparseEmbedding
    def __init__(self): 
        self.sparse_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)  # code to init or load model

    def embed_query(self, query: str) -> Dict[int, float]:
        sparse_output = self.sparse_model.encode(query, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        vec = sparse_output["lexical_weights"]
        # vec is already a dictionary of token: weight
        # Note: BGE-M3 usually returns string tokens. Ensure Milvus accepts this or convert to IDs if needed.
        return vec

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        result = []
        for text in texts:
            result.append(self.embed_query(text))
        return result


class SpladeEmbedder(BaseSparseEmbedding):  # inherit from BaseSparseEmbedding
    def __init__(self): 
        self.sparse_model = model.sparse.SpladeEmbeddingFunction(
            model_name="naver/splade-cocondenser-selfdistil", 
            device="cpu"
        )
        self.lock = Lock()

    def embed_query(self, query: str) -> Dict[int, float]:
        with self.lock:
            sparse_output = self.sparse_model.encode_queries([query])
        # sparse_output is a CSR matrix. For a single query, it has shape (1, vocab_size).
        # We want to return a dictionary {token_id: weight}.
        row_idx, col_idx = sparse_output.nonzero()
        weights = sparse_output.data
        return {int(idx): float(weight) for idx, weight in zip(col_idx, weights)}

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        with self.lock:
            sparse_output = self.sparse_model.encode_documents(texts)
        # sparse_output is a CSR matrix of shape (num_docs, vocab_size).
        # We need to convert each row into a dictionary.
        results = []
        for i in range(sparse_output.shape[0]):
            # Get the row corresponding to the i-th document
            row = sparse_output[i]
            # row is a 1D array, so nonzero() returns a tuple with one element (indices,)
            col_idx = row.nonzero()[0]
            weights = row.data
            results.append({int(idx): float(weight) for idx, weight in zip(col_idx, weights)})
        return results

    

