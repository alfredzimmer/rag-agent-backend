
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

def split_pdf(file: str, HeaderDetector, remove_headers_footers_func, chunking: bool = False, chunk_size: int = 512, chunk_overlap: int = 0) -> list[Document]:
    """
    Split a PDF file into chunks.
    """
    markdown = load_pdf_as_markdown(file, HeaderDetector, remove_headers_footers_func)
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
    )
    markdown_splits = markdown_splitter.split_text(markdown)

    if chunking:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        raw_splits = text_splitter.split_documents(markdown_splits)
        return raw_splits
    else:
        return markdown_splits

def format_splits_as_list(splits) -> list[dict]:
    """
    Format the document splits into a well-structured list of dictionaries.

    Args:
        splits: List of Document objects

    Returns:
        List of dictionaries with metadata and content
    """
    formatted_splits = []

    for i, doc in enumerate(splits):
        split_data = {
            "chunk_id": i,
            "metadata": doc.metadata,
            "content": doc.page_content.strip(),
            "char_count": len(doc.page_content),
            "word_count": len(doc.page_content.split())
        }
        formatted_splits.append(split_data)

    return formatted_splits


def main():
    """
    Main function for testing PDF chunking module.
    """
    import json

    print("=" * 70)
    print("PDF Chunker - Module Debug")
    print("=" * 70)

    # Configuration
    pdf_file = "src/extraction/documents/sample.pdf"  # Change this to test different PDFs

    print(f"\nConfiguration:")
    print(f"  PDF: {pdf_file}")
    print()

    # Split the PDF
    print(f"Processing: {pdf_file}")
    chunks = split_pdf(
        pdf_file,
        IEEEHeaderDetector,
        IEEE_remove_headers_footers
    )
    print("Statistics:")
    print(f"  Total chunks: {len(chunks)}")

    # Save to JSON file
    pdf_name = pathlib.Path(pdf_file).stem
    output_file = pathlib.Path(f"src/extraction/outputs/{pdf_name}_debug_chunks.json")

    output_data = { "chunks": {i: {"metadata": doc.metadata, "content": doc.page_content.strip(), "char_count": len(doc.page_content), "word_count": len(doc.page_content.split())} for i, doc in enumerate(chunks)} }
    

    output_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))

    print(f"\n✓ Saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()