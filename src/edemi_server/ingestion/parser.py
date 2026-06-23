from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
from docx import Document as DocxDocument

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


@dataclass(frozen=True)
class ParsedSection:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_document(path: Path) -> list[ParsedSection]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {suffix}")

    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix == ".docx":
        return _parse_docx(path)
    return _parse_text(path)


def _parse_pdf(path: Path) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                sections.append(ParsedSection(text=text, metadata={"page": page_number}))
    return sections


def _parse_docx(path: Path) -> list[ParsedSection]:
    document = DocxDocument(path)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
    return [ParsedSection(text=text)] if text else []


def _parse_text(path: Path) -> list[ParsedSection]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [ParsedSection(text=text)] if text else []
