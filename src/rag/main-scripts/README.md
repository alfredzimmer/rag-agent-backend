# PDF to DB Pipeline - Quick Reference

## Overview
This script processes PDFs through the complete pipeline: PDF → Markdown → Chunks → Milvus DB

## Features
✅ **Preview Mode**: Shows first 5 pages before processing entire PDF
✅ **Custom Collection Names**: Specify which Milvus collection to store in
✅ **Configurable Parsing**: Support for IEEE and NFPA document formats
✅ **Interactive Confirmation**: Review preview before committing to full processing

## Usage

### Command Line Arguments
```
python pdf_to_db.py <pdf_path> [config_name] [collection_name]
```

**Arguments:**
1. `pdf_path` - Path to PDF file (required)
2. `config_name` - PDF parsing config: "ieee" or "nfpa" (optional, default: "ieee")
3. `collection_name` - Milvus collection name (optional, default: "splade")

### Basic Usage (all defaults: ieee config, splade collection)
```bash
python src/rag/main-scripts/pdf_to_db.py src/rag/public/IEEE1584-2018-31-36.pdf
```

### Specify Config Only (NFPA document)
```bash
python src/rag/main-scripts/pdf_to_db.py src/rag/public/NFPA-110-2019-9-24.pdf nfpa
```

### Specify Both Config and Collection
```bash
python src/rag/main-scripts/pdf_to_db.py src/rag/public/IEEE1584-2018-31-36.pdf ieee my_custom_collection
```

### Programmatic Usage (Python)
```python
from src.rag.main_scripts.pdf_to_db import process_pdf_to_db

# With preview mode (default)
process_pdf_to_db(
    pdf_path="src/rag/public/IEEE1584-2018-31-36.pdf",
    collection_name="ieee_standards",  # Optional
    config_name="ieee",                # "ieee" or "nfpa"
    preview_mode=True                  # Show preview first
)

# Skip preview and process directly
process_pdf_to_db(
    pdf_path="src/rag/public/IEEE1584-2018-31-36.pdf",
    collection_name="ieee_standards",
    preview_mode=False
)
```

## Workflow

1. **Preview Generation** (if `preview_mode=True`)
   - Extracts first 5 pages
   - Generates markdown and JSON chunks
   - Saves to `src/rag/outputs/preview/{filename}/`
     - `{filename}.md` - Markdown preview
     - `{filename}.json` - JSON chunks preview
   
2. **User Confirmation**
   - Press ENTER to continue with full processing
   - Press 'q' to quit

3. **Full Processing**
   - Converts entire PDF to markdown
   - Splits into chunks with header hierarchy
   - Saves JSON to `src/rag/outputs/{filename}.json`
   - Stores in Milvus DB with dense + sparse embeddings

## Collection Name Configuration

### Answer to Question 1: How to set collection name?

**Method 1: Command Line (3rd argument)**
```bash
python src/rag/main-scripts/pdf_to_db.py path/to/file.pdf ieee YOUR_COLLECTION_NAME
```

**Method 2: Programmatic**
```python
process_pdf_to_db(pdf_path="...", collection_name="YOUR_COLLECTION_NAME")
```

**Method 3: Default (uses RAGConfig)**
If you don't specify a collection name, it uses `RAGConfig.sparse_embedding_model`:
- Default: `"splade"` (from `src/rag/config.py`)
- Other options: `"bm25"`, `"bge"`

**Examples:**
```bash
# Uses default collection "splade"
python pdf_to_db.py src/rag/public/IEEE1584-2018-31-36.pdf

# Uses default collection "splade" with NFPA config
python pdf_to_db.py src/rag/public/NFPA-110-2019-9-24.pdf nfpa

# Custom collection with IEEE config
python pdf_to_db.py src/rag/public/IEEE1584-2018-31-36.pdf ieee my_collection
```

## Preview Mode Details

### Answer to Question 2: Preview with MD + JSON

The script automatically:
1. Creates preview directory: `src/rag/outputs/preview/{filename}/`
2. Saves two files:
   - `{filename}.md` - Full markdown of first 5 pages
   - `{filename}.json` - JSON chunks of first 5 pages
3. Displays summary in terminal
4. Waits for ENTER key before processing full PDF

**Preview Output Example:**
```
============================================================
Preview generated for: IEEE1584-2018-31-36
============================================================
Total pages in PDF: 6
Preview pages: 5
Preview directory: src/rag/outputs/preview/IEEE1584-2018-31-36
  - Markdown: IEEE1584-2018-31-36.md
  - JSON chunks: IEEE1584-2018-31-36.json (12 chunks)
============================================================

Preview files generated. Review them before proceeding.
Location: src/rag/outputs/preview/IEEE1584-2018-31-36

Press ENTER to process the entire PDF, or 'q' to quit: 
```

## Output Files

### Preview Files
- `src/rag/outputs/preview/{filename}/{filename}.md`
- `src/rag/outputs/preview/{filename}/{filename}.json`

### Full Processing Files
- `src/rag/outputs/{filename}.md` (created by pdf_chunker.py)
- `src/rag/outputs/{filename}.json` (full chunks)

### Database
- Milvus collection: `{collection_name}` or default from config
- Contains dense + sparse embeddings

## Configuration Options

### RAGConfig (src/rag/config.py)
```python
@dataclass
class RAGConfig:
    dense_embedding_model: str = "qwen3-embedding:8b"
    sparse_embedding_model: str = "splade"  # Used as default collection name
```

### PDF Parsing Configs (src/rag/utils/pdf_chunker.py)
- `"ieee"` - For IEEE standards documents
- `"nfpa"` - For NFPA standards documents

## Examples

### Example 1: IEEE Document with Default Collection (splade)
```bash
python src/rag/main-scripts/pdf_to_db.py \
    src/rag/public/IEEE1584-2018-31-36.pdf
```

### Example 2: IEEE Document with Custom Collection
```bash
python src/rag/main-scripts/pdf_to_db.py \
    src/rag/public/IEEE1584-2018-31-36.pdf \
    ieee \
    ieee_arc_flash
```

### Example 3: NFPA Document with Custom Collection
```bash
python src/rag/main-scripts/pdf_to_db.py \
    src/rag/public/NFPA-110-2019-9-24.pdf \
    nfpa \
    nfpa_emergency_power
```

### Example 4: Skip Preview Mode (Programmatic)
```python
from src.rag.main_scripts.pdf_to_db import process_pdf_to_db

process_pdf_to_db(
    pdf_path="src/rag/public/IEEE1584-2018-31-36.pdf",
    config_name="ieee",
    collection_name="quick_test",
    preview_mode=False  # Skip preview
)
```

## Troubleshooting

### Issue: Collection already exists
- The script will add to existing collection
- To create fresh collection, use `create_db()` from `milvus.py` first

### Issue: PDF not found
- Use absolute or relative path from project root
- Example: `src/rag/public/yourfile.pdf`

### Issue: Wrong parsing format
- Make sure to specify correct config: `ieee` or `nfpa`
- Default is `ieee` if not specified
