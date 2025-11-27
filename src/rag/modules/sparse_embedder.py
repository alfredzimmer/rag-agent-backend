from typing import Dict, List
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from FlagEmbedding import BGEM3FlagModel


class SparseEmbedder(BaseSparseEmbedding):  # inherit from BaseSparseEmbedding
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

