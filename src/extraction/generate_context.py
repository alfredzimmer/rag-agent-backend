import random
from langchain_core.documents import Document

from embeddings import embed_chunks

def generate_context(chunks: list[Document]):
    content, idx_to_metadata, embeddings = embed_chunks(chunks)
    pivot_index = random.randint(0, len(embeddings) - 1)
    pivot_embedding = embeddings[pivot_index]
    contexts = [content[pivot_index]]
    metadata = idx_to_metadata[pivot_index]
    



