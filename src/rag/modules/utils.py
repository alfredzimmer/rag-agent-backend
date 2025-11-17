from langchain_core.documents import Document

"""
Convert ScoredPoint in Qdrant to Document in Langchain
  Used after retrieval and before reranking
"""
def scoredpoint_to_document(sp):
  page_content = sp.payload.get("page_content", "")
  metadata = {k : v for k, v in sp.payload.items() if k != "text"}
  return Document(page_content=page_content, metadata=metadata)

