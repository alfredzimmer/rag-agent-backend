import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.utils.pdf_chunker import load_pdf_as_markdown, split_pdf, format_splits_as_list
from src.rag.milvus import create_milvus_store
from src.rag.config import RAGConfig
import json
import pathlib
import pymupdf


def preview_pdf_first_pages(pdf_path: str, config_name: str = "ieee", num_pages: int = 5):
    """
    Generate preview of first N pages: markdown and JSON chunks.
    Saves to preview/{filename}/ directory.
    
    Returns: (preview_dir, markdown_text, json_chunks)
    """
    # Resolve path (can be absolute or relative)
    pdf_file = pathlib.Path(pdf_path)
    if not pdf_file.is_absolute():
        if pdf_file.exists():
            pdf_file = pdf_file.resolve()
        elif (project_root / pdf_path).exists():
            pdf_file = (project_root / pdf_path).resolve()
        else:
            pdf_file = (project_root / pdf_path).resolve()
    
    filename_stem = pdf_file.stem
    
    # Create preview directory (relative to project root)
    preview_dir = project_root / "src" / "rag" / "outputs" / "preview" / filename_stem
    preview_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract first N pages to a temporary PDF
    temp_pdf = preview_dir / f"{filename_stem}_preview.pdf"
    with pymupdf.open(str(pdf_file)) as doc:
        total_pages = len(doc)
        pages_to_extract = min(num_pages, total_pages)
        
        # Create new PDF with first N pages
        preview_doc = pymupdf.open()
        for page_num in range(pages_to_extract):
            preview_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        preview_doc.save(temp_pdf)
        preview_doc.close()
    
    # Generate markdown from preview PDF
    markdown_text = load_pdf_as_markdown(str(temp_pdf), config_name)
    
    # Generate chunks from preview PDF
    chunks = split_pdf(str(temp_pdf), config_name)
    formatted_chunks = format_splits_as_list(chunks)
    
    # Save markdown
    md_file = preview_dir / f"{filename_stem}.md"
    md_file.write_text(markdown_text, encoding='utf-8')
    
    # Save JSON chunks
    json_file = preview_dir / f"{filename_stem}.json"
    json_file.write_text(json.dumps(formatted_chunks, indent=2, ensure_ascii=False), encoding='utf-8')
    
    # Clean up temp PDF
    temp_pdf.unlink()
    
    print(f"\n{'='*60}")
    print(f"Preview generated for: {filename_stem}")
    print(f"{'='*60}")
    print(f"Total pages in PDF: {total_pages}")
    print(f"Preview pages: {pages_to_extract}")
    print(f"Preview directory: {preview_dir}")
    print(f"  - Markdown: {md_file.name}")
    print(f"  - JSON chunks: {json_file.name} ({len(formatted_chunks)} chunks)")
    print(f"{'='*60}\n")
    
    return preview_dir, markdown_text, formatted_chunks


def process_pdf_to_db(pdf_path: str, collection_name: str = None, config_name: str = "ieee", preview_mode: bool = True):
    """
    Process PDF to Milvus DB with optional preview.
    
    Args:
        pdf_path: Path to PDF file (absolute or relative)
        collection_name: Name of Milvus collection (defaults to sparse_embedding_model from config)
        config_name: PDF parsing config ("ieee" or "nfpa")
        preview_mode: If True, show preview and ask for confirmation before processing
    """
    # Try to resolve the PDF path intelligently
    pdf_file = pathlib.Path(pdf_path)
    
    # If not absolute, try multiple resolution strategies
    if not pdf_file.is_absolute():
        # Strategy 1: Relative to current working directory
        if pdf_file.exists():
            pdf_file = pdf_file.resolve()
        # Strategy 2: Relative to project root
        elif (project_root / pdf_path).exists():
            pdf_file = (project_root / pdf_path).resolve()
        # Strategy 3: Already in project, just prepend project root
        else:
            pdf_file = (project_root / pdf_path).resolve()
    
    if not pdf_file.exists():
        print(f"Error: PDF file not found!")
        print(f"Searched for: {pdf_path}")
        print(f"Resolved to: {pdf_file}")
        print(f"Project root: {project_root}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Preview mode: show first 20 pages
    if preview_mode:
        preview_dir, markdown_preview, chunks_preview = preview_pdf_first_pages(str(pdf_file), config_name, num_pages=20)
        
        # Ask for confirmation
        print("Preview files generated. Review them before proceeding.")
        print(f"Location: {preview_dir}")
        response = input("\nPress ENTER to process the entire PDF, or 'q' to quit: ").strip().lower()
        
        if response == 'q':
            print("Processing cancelled.")
            return
    
    # Process full PDF
    print(f"\n{'='*60}")
    print(f"Processing full PDF: {pdf_file.name}")
    print(f"{'='*60}\n")
    
    # 1. PDF -> chunks
    chunks = split_pdf(str(pdf_file), config_name)
    formatted = format_splits_as_list(chunks)
    
    # 2. Save intermediate JSON (relative to project root)
    output_json = project_root / "src" / "rag" / "outputs" / f"{pdf_file.stem}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(formatted, indent=2, ensure_ascii=False))
    
    print(f"✓ Saved {len(formatted)} chunks to {output_json}")
    
    # 3. Load to Milvus
    config = RAGConfig()
    
    # Override collection name if provided
    if collection_name:
        original_sparse_model = config.sparse_embedding_model
        config.sparse_embedding_model = collection_name
        print(f"Using custom collection name: {collection_name}")
    
    vector_store = create_milvus_store(config)
    
    # Load just this file's chunks
    documents = vector_store.load_chunk_file(str(output_json))
    vector_store.vector_store.add_documents(documents=documents)
    
    print(f"✓ Stored {len(documents)} chunks to Milvus collection: {config.sparse_embedding_model}")
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_db.py <pdf_path> [config_name] [collection_name]")
        print("\nArguments:")
        print("  pdf_path         : Path to PDF file (required)")
        print("  config_name      : PDF parsing config - 'ieee' or 'nfpa' (default: 'ieee')")
        print("  collection_name  : Milvus collection name (default: 'splade' from RAGConfig)")
        print("\nExamples:")
        print("  # Use all defaults (ieee config, splade collection)")
        print("  python pdf_to_db.py src/rag/public/IEEE1584-2018-31-36.pdf")
        print()
        print("  # Specify config only")
        print("  python pdf_to_db.py src/rag/public/NFPA-110-2019-9-24.pdf nfpa")
        print()
        print("  # Specify both config and collection")
        print("  python pdf_to_db.py src/rag/public/IEEE1584-2018-31-36.pdf ieee my_custom_collection")
        print()
        print("Defaults:")
        print(f"  - Config: ieee")
        print(f"  - Collection: splade (from RAGConfig.sparse_embedding_model)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    config_name = sys.argv[2] if len(sys.argv) > 2 else "ieee"
    collection_name = sys.argv[3] if len(sys.argv) > 3 else None  # None = use default from RAGConfig
    
    print(f"\n{'='*60}")
    print("PDF to Milvus DB Pipeline")
    print(f"{'='*60}")
    print(f"PDF Path: {pdf_path}")
    print(f"Config: {config_name}")
    print(f"Collection: {collection_name if collection_name else 'default (splade)'}")
    print(f"{'='*60}\n")
    
    process_pdf_to_db(pdf_path, collection_name=collection_name, config_name=config_name, preview_mode=True)

