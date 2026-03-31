"""Extract text from PDF and split into chunks (sizes are placeholders for later tuning)."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from config import CHUNK_CHAR_OVERLAP, CHUNK_CHAR_SIZE


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_CHAR_SIZE,
    overlap: int = CHUNK_CHAR_OVERLAP,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(text):
        piece = text[i : i + chunk_size].strip()
        if piece:
            chunks.append(piece)
        i += step
    return chunks
