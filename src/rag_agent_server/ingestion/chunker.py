from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .parser import ParsedSection

CHUNKER_VERSION = "recursive-v1"


def chunk_sections(
    sections: list[ParsedSection],
    *,
    chunk_size: int,
    chunk_overlap: int,
    original_filename: str,
    document_id: str,
    job_id: UUID,
    scope_id: UUID,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []
    ingested_at = datetime.now(timezone.utc).isoformat()

    for section_index, section in enumerate(sections):
        for chunk in splitter.split_text(section.text):
            text = chunk.strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        **section.metadata,
                        "chunk_index": len(documents),
                        "section_index": section_index,
                        "chunker_version": CHUNKER_VERSION,
                        "document_id": document_id,
                        "ingestion_job_id": str(job_id),
                        "scope_id": str(scope_id),
                        "source": original_filename,
                        "filename": original_filename,
                        "ingested_at": ingested_at,
                    },
                )
            )

    return documents
