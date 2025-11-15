from langchain_core.documents import Document

"""
Convert ScoredPoint in Qdrant to Document in Langchain
  Used after retrieval and before reranking
"""
def scoredpoint_to_document(sp):
  text = sp.payload.get("text", "")
  metadata = {k : v for k, v in sp.payload.items() if k != "text"}
  return Document(page_content=text, metadata=metadata)

