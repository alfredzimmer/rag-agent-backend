"""Native document ingestion for the RAG agent.

parse -> clean -> header-aware chunk -> embed -> Milvus, reusing the agent's
own RAGConfig and Milvus store so the collection it builds is a drop-in for
retrieval (only RAG_COLLECTION_NAME changes at cutover).

The previous ingestor produced line-level fragments polluted with base64 image
bytes. Here PDFs are parsed to Markdown with either pymupdf4llm or fast
page-text extraction (real text + headers, not image bytes), TOC/binary/page
number noise is stripped, and chunks are split on Markdown headers (header path
kept in metadata AND text).

Run (on the server, where Milvus + Ollama + the documents live):

    uv run --group ingest rag-ingest \
        --root "/home/ziyutecc_ai_wsl/RAG Knowledge Codes" --dry-run

    uv run --group ingest rag-ingest \
        --root "/home/ziyutecc_ai_wsl/RAG Knowledge Codes" \
        --db rag2 --collection rag_documents_v2 --drop-old \
        --pdf-parser text \
        --manifest ~/.config/rag-agent/ingestion/ingest-rag2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .config import RAGConfig
from .milvus import create_milvus_store

SUPPORTED = {".pdf", ".txt", ".md"}
SKIP_DIRS = {"parsed", "sample_set"}

_WS = re.compile(r"\s+")
_DOTLEADER = re.compile(r"\.{6,}")            # TOC dot leaders
_B64RUN = re.compile(r"[A-Za-z0-9+/=]{40,}")  # base64 / binary blobs
_PAGENUM = re.compile(r"^\s*\|?\s*(?:[ivxlcdm]{1,6}|\d{1,4})\s*\|?\s*$", re.I)

HEADERS = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]

INDEX_PARAMS = {"index_type": "HNSW", "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200}}


# --------------------------------------------------------------------------
# parse / clean / chunk
# --------------------------------------------------------------------------
def garbage_frac(text: str) -> float:
    """Fraction of characters living in >30-char whitespace-free runs."""
    body = text.replace(" ", "").replace("\n", "")
    if not body:
        return 1.0
    return sum(len(w) for w in _WS.split(text) if len(w) > 30) / len(body)


def parse_pdf_text(path: Path) -> str:
    import pymupdf  # imported lazily so the api image needn't carry ingest deps

    parts = [f"# {path.stem}"]
    with pymupdf.open(str(path)) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                parts.append(f"## Page {page_number}\n\n{text}")
    return "\n\n".join(parts)


def parse_document(path: Path, *, pdf_parser: str, fast_pdf_mb: int) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        use_text = pdf_parser == "text" or (
            pdf_parser == "auto" and path.stat().st_size >= fast_pdf_mb * 1024 * 1024
        )
        if use_text:
            return parse_pdf_text(path), "pymupdf-text"
        try:
            import pymupdf4llm  # imported lazily so the api image needn't carry it

            return pymupdf4llm.to_markdown(str(path), show_progress=False), "pymupdf4llm"
        except Exception:
            if pdf_parser == "markdown":
                raise
            return parse_pdf_text(path), "pymupdf-text-fallback"
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace"), "text"
    raise ValueError(f"unsupported file type: {ext}")


def clean_markdown(md: str) -> str:
    out = []
    for line in md.splitlines():
        if _DOTLEADER.search(line):          # table-of-contents leader line
            continue
        if _PAGENUM.match(line):             # bare page number / roman numeral
            continue
        out.append(_B64RUN.sub(" ", line))   # nuke any binary/base64 run
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out))
    return text.strip()


def looks_unparseable(md: str, *, ext: str) -> bool:
    """Detect word-soup / image-only PDFs (e.g. the NBC scan) worth skipping."""
    min_chars = 500 if ext == ".pdf" else 60
    if len(md.strip()) < min_chars:
        return True
    words = [w for w in _WS.split(md) if w]
    if not words:
        return True
    avg = sum(len(w) for w in words) / len(words)
    return avg < 2.2 or avg > 18 or garbage_frac(md) > 0.35


def chunk_markdown(md: str, *, chunk_size: int, overlap: int):
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
    sections = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False).split_text(md)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    kept = []
    for d in size_splitter.split_documents(sections):
        text = d.page_content.strip()
        if sum(c.isalnum() for c in text) < 60 or garbage_frac(text) > 0.5:
            continue
        kept.append(d)
    return kept


def process_file(
    path: Path,
    *,
    chunk_size: int,
    overlap: int,
    category: str,
    pdf_parser: str,
    fast_pdf_mb: int,
    source_root: Path | None = None,
):
    parsed, parser_used = parse_document(path, pdf_parser=pdf_parser, fast_pdf_mb=fast_pdf_mb)
    md = clean_markdown(parsed)
    if looks_unparseable(md, ext=path.suffix.lower()):
        raise RuntimeError("document parsed to low-quality text (scan/word-soup)")
    docs = chunk_markdown(md, chunk_size=chunk_size, overlap=overlap)
    source_path = path.name
    if source_root is not None:
        try:
            source_path = str(path.relative_to(source_root))
        except ValueError:
            source_path = str(path)
    for i, d in enumerate(docs):
        header_meta = {k: v for k, v in d.metadata.items() if k.startswith("Header")}
        d.metadata = {
            "name": path.name,
            "original_filename": path.name,
            "source_path": source_path,
            "source_dir": str(Path(source_path).parent),
            "source_ext": path.suffix.lower(),
            "parser": parser_used,
            "category": category,
            "chunk_index": i,
            "n_chunks": len(docs),
            **header_meta,
        }
    return docs


# --------------------------------------------------------------------------
# discovery / helpers
# --------------------------------------------------------------------------
def discover(root: Path, include_parsed: bool) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED or p.name.startswith("."):
            continue
        if not include_parsed and any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        files.append(p)
    return files


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def category_for(path: Path, *, root: Path | None, override: str | None) -> str:
    if override:
        return override
    if root is not None:
        try:
            rel = path.relative_to(root)
            if len(rel.parts) > 1:
                return rel.parts[0]
        except ValueError:
            pass
    return path.parent.name


def write_manifest(path: Path, payload: dict) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {path}", flush=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    run_started = datetime.now(timezone.utc)
    load_dotenv()
    ap = argparse.ArgumentParser(description="Native RAG ingestion pipeline.")
    ap.add_argument("--root", help="Directory of source documents.")
    ap.add_argument("--file", help="Ingest a single file instead of --root.")
    ap.add_argument("--db", default=None, help="Target Milvus database (default: MILVUS_DB).")
    ap.add_argument("--collection", default=None, help="Target collection (default: RAG_COLLECTION_NAME).")
    ap.add_argument("--category", default=None, help="Override category metadata for all chunks.")
    ap.add_argument(
        "--pdf-parser",
        choices=("auto", "markdown", "text"),
        default="auto",
        help="PDF parser: markdown uses pymupdf4llm, text uses fast page text, auto uses text for large PDFs.",
    )
    ap.add_argument(
        "--fast-pdf-mb",
        type=int,
        default=15,
        help="In auto mode, PDFs at or above this size use the fast text parser.",
    )
    ap.add_argument("--chunk-size", type=int, default=1800, help="Max chunk length (chars).")
    ap.add_argument("--overlap", type=int, default=250)
    ap.add_argument("--batch", type=int, default=256, help="Docs per embed/insert batch.")
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N files.")
    ap.add_argument("--include-parsed", action="store_true", help="Also ingest parsed/ and sample_set/ subdirs.")
    ap.add_argument("--drop-old", action="store_true", help="Drop the collection first.")
    ap.add_argument("--dry-run", action="store_true", help="Parse+chunk only; write nothing.")
    ap.add_argument("--manifest", help="Write a JSON audit record for this ingestion run.")
    ap.add_argument("--sample", type=int, default=3, help="Sample chunks to print in dry-run.")
    args = ap.parse_args()

    config = RAGConfig()
    if args.db:
        config.milvus_db = args.db
    target = args.collection or config.collection_name
    root = None
    if args.file:
        file_path = Path(args.file).expanduser()
        files = [file_path]
        root = file_path.parent
    elif args.root:
        root = Path(args.root).expanduser()
        if not root.is_dir():
            sys.exit(f"root not found: {root}")
        files = discover(root, args.include_parsed)
    else:
        sys.exit("provide --root or --file")
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit("no supported files found.")

    print(
        f"Discovered {len(files)} file(s). db={config.milvus_db} collection={target} "
        f"chunk_size={args.chunk_size} overlap={args.overlap} dry_run={args.dry_run}",
        flush=True,
    )

    all_docs, seen_sha, skipped, per_file = [], set(), [], []
    for path in files:
        sha = sha256(path)
        source_path = str(path.relative_to(root)) if root is not None else str(path)
        if sha in seen_sha:
            skipped.append({"path": source_path, "name": path.name, "reason": "duplicate content"})
            continue
        seen_sha.add(sha)
        category = category_for(path, root=root, override=args.category)
        t0 = time.time()
        try:
            print(f"  parse {path.name:58.58s}", flush=True)
            docs = process_file(
                path,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                category=category,
                pdf_parser=args.pdf_parser,
                fast_pdf_mb=args.fast_pdf_mb,
                source_root=root,
            )
        except Exception as e:
            skipped.append({"path": source_path, "name": path.name, "reason": str(e)})
            print(f"  SKIP  {path.name}: {e}", flush=True)
            continue
        for d in docs:
            d.metadata["doc_sha"] = sha
        all_docs.extend(docs)
        per_file.append(
            {
                "path": source_path,
                "name": path.name,
                "category": category,
                "sha256": sha,
                "chunks": len(docs),
                "seconds": round(time.time() - t0, 3),
            }
        )
        print(f"  ok    {path.name:58.58s} -> {len(docs):4d} chunks ({time.time()-t0:.1f}s)", flush=True)

    if not all_docs:
        sys.exit("no chunks produced.")

    lens = [len(d.page_content) for d in all_docs]
    gfrac = sum(garbage_frac(d.page_content) for d in all_docs) / len(all_docs)
    print("\n" + "=" * 70)
    print(f"Files ingested : {len(per_file)}   skipped: {len(skipped)}")
    print(f"Total chunks   : {len(all_docs)}")
    print(f"Chunk chars    : min={min(lens)} avg={sum(lens)//len(lens)} max={max(lens)}")
    print(f"Avg garbage    : {gfrac:.3%}")
    for item in skipped:
        print(f"    skip {item['path']}: {item['reason']}")

    manifest = {
        "started_at": run_started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "source": {
            "root": str(root) if root is not None else None,
            "include_parsed": args.include_parsed,
            "supported_extensions": sorted(SUPPORTED),
        },
        "target": {
            "milvus_uri": config.milvus_uri,
            "milvus_db": config.milvus_db,
            "collection": target,
            "embedding_model": config.embedding_model,
            "index_params": INDEX_PARAMS,
        },
        "chunking": {
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "pdf_parser": args.pdf_parser,
            "fast_pdf_mb": args.fast_pdf_mb,
        },
        "totals": {
            "files_discovered": len(files),
            "files_ingested": len(per_file),
            "files_skipped": len(skipped),
            "chunks": len(all_docs),
            "chunk_chars_min": min(lens),
            "chunk_chars_avg": sum(lens) // len(lens),
            "chunk_chars_max": max(lens),
            "avg_garbage_fraction": gfrac,
        },
        "files": per_file,
        "skipped": skipped,
    }

    if args.dry_run:
        print("\n--- sample chunks ---")
        step = max(1, len(all_docs) // max(1, args.sample))
        for d in all_docs[::step][:args.sample]:
            hdr = " / ".join(v for k, v in sorted(d.metadata.items()) if k.startswith("Header"))
            print(f"\n[{d.metadata['name']} #{d.metadata['chunk_index']}] {hdr}")
            print(d.page_content[:600].strip())
        print("\nDry run — nothing written.")
        if args.manifest:
            write_manifest(Path(args.manifest), manifest)
        return

    store = create_milvus_store(
        config, collection_name=target, drop_old=args.drop_old, index_params=INDEX_PARAMS,
    )
    print(f"\nWriting {len(all_docs)} chunks to {config.milvus_db}/{target} …", flush=True)
    t0, written = time.time(), 0
    for batch in batched(all_docs, args.batch):
        store.add_documents(batch)
        written += len(batch)
        print(f"  {written}/{len(all_docs)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nDone. Wrote {written} chunks to {config.milvus_db}/{target} in {time.time()-t0:.0f}s.")
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["totals"]["chunks_written"] = written
    if args.manifest:
        write_manifest(Path(args.manifest), manifest)


if __name__ == "__main__":
    main()
