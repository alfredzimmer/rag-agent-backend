import pymupdf
# import pymupdf.layout
import pymupdf4llm
import pathlib
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from IEEE_utils import IEEEHeaderDetector, IEEE_remove_headers_footers

def load_pdf_as_markdown(file: str, HeaderDetector, remove_headers_footers_func) -> str:
    """
    Extract text from a PDF file. Automatically detect headers (IEEE)
    """
    with pymupdf.open(file) as doc:
        chunks = pymupdf4llm.to_markdown(
            doc,
            hdr_info=HeaderDetector(),
            show_progress=True
        )

        # Remove headers and footers
        chunks = remove_headers_footers_func(chunks)

        return chunks

def split_pdf(file: str, HeaderDetector, remove_headers_footers_func) -> list[Document]:
    """
    Split a PDF file into chunks.
    """
    markdown = load_pdf_as_markdown(file, HeaderDetector, remove_headers_footers_func)
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
    )
    raw_splits = markdown_splitter.split_text(markdown)
    
    pathlib.Path("src/rag/outputs/output.md").write_bytes(markdown.encode())

    
    # split raw_splits again with chunk size constraints
    chunk_size = 512
    chunk_overlap = 200
    chunk_size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    sized_splits = chunk_size_splitter.split_documents(raw_splits)

    return sized_splits


# For debugging purpose 
def format_splits_as_list(splits) -> list[dict]:
    """
    Format the markdown header splits into a well-structured list of dictionaries.

    Args:
        splits: List of Document objects from MarkdownHeaderTextSplitter

    Returns:
        List of dictionaries with headers and content
    """
    formatted_splits = []

    for i, doc in enumerate(splits):
        split_data = {
            "chunk_id": i,
            "headers": doc.metadata,
            "content": doc.page_content.strip(),
            "char_count": len(doc.page_content)
        }
        formatted_splits.append(split_data)

    return formatted_splits

# Debug testing
if __name__ == "__main__":
    FILE_PATH = "src/rag/public/IEEE Blue Book Std 1015-2006-13-30.pdf"
    chunks = split_pdf(FILE_PATH, IEEEHeaderDetector, IEEE_remove_headers_footers)
    formatted_list = format_splits_as_list(chunks)

    # Optional: Save to JSON file
    import json
    output_file = pathlib.Path("src/rag/outputs/30pg_outputs.json")
    output_file.write_text(json.dumps(formatted_list, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(formatted_list)} chunks to {output_file}")
