from pathlib import Path
from typing import TypedDict


class PDFPage(TypedDict):
    page: int
    text: str
    source: str
    path: str


def load_pdf(file_path: str | Path) -> list[PDFPage]:
    """Extract non-empty PDF pages with citation-safe source metadata."""
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")

    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - installed runtime only
        raise RuntimeError("PyMuPDF is required to load PDF documents") from exc

    resolved_path = path.resolve()
    pages: list[PDFPage] = []
    with pymupdf.open(str(resolved_path)) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                        "source": resolved_path.name,
                        "path": str(resolved_path),
                    }
                )
    return pages
