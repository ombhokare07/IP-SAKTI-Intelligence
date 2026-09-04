from hashlib import sha256
from pathlib import Path
from typing import TypedDict

from rag.ingestion.chunker import chunk_text
from rag.ingestion.pdf_loader import load_pdf
from rag.ingestion.text_cleaner import clean_text


class ProcessedChunk(TypedDict):
    chunk_id: str
    text: str
    source: str
    page: int
    path: str
    chunk_index_on_page: int


def _chunk_id(path: str, page: int, chunk_index: int, text: str) -> str:
    identity = f"{path}\0{page}\0{chunk_index}\0{text}".encode("utf-8")
    return sha256(identity).hexdigest()


def process_pdf(
    file_path: str | Path,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[ProcessedChunk]:
    """Load, clean, and chunk a PDF while preserving source metadata."""
    documents: list[ProcessedChunk] = []
    for page in load_pdf(file_path):
        cleaned_text = clean_text(page["text"])
        page_chunks = chunk_text(cleaned_text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, text in enumerate(page_chunks, start=1):
            documents.append(
                {
                    "chunk_id": _chunk_id(
                        page["path"], page["page"], chunk_index, text
                    ),
                    "text": text,
                    "source": page["source"],
                    "page": page["page"],
                    "path": page["path"],
                    "chunk_index_on_page": chunk_index,
                }
            )
    return documents
