# Quick Summary: Argument Order Update

## ✅ Confirmed Defaults

**Yes, you're correct!** When no arguments are given (except the PDF path):
- **Collection name**: defaults to `"splade"` (from `RAGConfig.sparse_embedding_model`)
- **Config name**: defaults to `"ieee"`

## 🔄 New Argument Order

**Old order:**
```bash
python pdf_to_db.py <pdf_path> [collection_name] [config_name]
```

**New order (as requested):**
```bash
python pdf_to_db.py <pdf_path> [config_name] [collection_name]
```

### Arguments:
1. **`pdf_path`** (required) - Path to PDF file
2. **`config_name`** (optional, default: `"ieee"`) - PDF parsing config: "ieee" or "nfpa"
3. **`collection_name`** (optional, default: `"splade"`) - Milvus collection name

## 📝 Usage Examples

### Use all defaults (ieee + splade)
```bash
python pdf_to_db.py src/codes_and_standards/IEEE/_IEEE_Std_1100-2005.pdf
```
- Config: ieee ✓
- Collection: splade ✓

### Specify config only (nfpa + splade)
```bash
python pdf_to_db.py src/codes_and_standards/NFPA/some_nfpa.pdf nfpa
```
- Config: nfpa ✓
- Collection: splade ✓

### Specify both config and collection
```bash
python pdf_to_db.py src/codes_and_standards/IEEE/_IEEE_Std_1100-2005.pdf ieee my_custom_collection
```
- Config: ieee ✓
- Collection: my_custom_collection ✓

## 🎯 What Changed

1. **Swapped argument positions**: config_name now comes before collection_name
2. **Better help text**: Shows all defaults and examples
3. **Startup banner**: Displays what settings will be used before processing
4. **Path resolution**: Script now works from any directory (resolves paths relative to project root)

## 🚀 Ready to Use

Try it now:
```bash
cd /home/wayne_hao/pyapi/src/rag/main-scripts
python pdf_to_db.py src/codes_and_standards/IEEE/_IEEE_Std_1100-2005.pdf
```

This will:
1. Generate preview of first 5 pages → `src/rag/outputs/preview/_IEEE_Std_1100-2005/`
2. Wait for your confirmation (press ENTER)
3. Process full PDF with ieee config
4. Store in Milvus collection "splade"
