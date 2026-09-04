import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

if __package__ in (None, ""):
    from _bootstrap import PROJECT_ROOT, ensure_project_root
else:
    from scripts._bootstrap import PROJECT_ROOT, ensure_project_root

ensure_project_root()

from config.settings import settings
from database.vector.vector_store import ChromaVectorStore
from rag.embeddings.embedding_service import BGEEmbeddingService
from rag.ingestion.pipeline import ProcessedChunk, process_pdf


DEFAULT_PDF_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "india" / "patents"


class EmbeddingService(Protocol):
    def embed_documents(
        self, texts: Sequence[str], *, batch_size: int = 32
    ) -> list[list[float]]: ...


class VectorStore(Protocol):
    def upsert_chunks(
        self,
        chunks: Sequence[ProcessedChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int: ...


@dataclass(frozen=True)
class IngestionSummary:
    pdf_count: int
    page_count: int
    chunk_count: int
    embedding_count: int
    stored_count: int


def discover_pdfs(inputs: Sequence[str | Path]) -> list[Path]:
    """Resolve files and recursively discover PDFs under input directories."""
    discovered: set[Path] = set()
    for raw_path in inputs:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
        if path.is_file():
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"Expected a PDF file: {path}")
            discovered.add(path.resolve())
            continue
        discovered.update(
            child.resolve()
            for child in path.rglob("*")
            if child.is_file() and child.suffix.lower() == ".pdf"
        )
    return sorted(discovered, key=lambda item: str(item).casefold())


def ingest_pdf(
    pdf_path: str | Path,
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    *,
    batch_size: int = 32,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> int:
    """Process, embed, and upsert one PDF in bounded batches."""
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero")
    chunks = process_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)
    stored = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = embedding_service.embed_documents(
            [chunk["text"] for chunk in batch], batch_size=batch_size
        )
        stored += vector_store.upsert_chunks(batch, embeddings)
    return stored


def _ingest_pdfs(
    pdfs: Sequence[Path],
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    *,
    batch_size: int = 32,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> IngestionSummary:
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero")

    page_count = 0
    chunk_count = 0
    embedding_count = 0
    stored_count = 0
    for pdf in pdfs:
        chunks = process_pdf(pdf, chunk_size=chunk_size, overlap=overlap)
        page_count += len({chunk["page"] for chunk in chunks})
        chunk_count += len(chunks)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = embedding_service.embed_documents(
                [chunk["text"] for chunk in batch], batch_size=batch_size
            )
            embedding_count += len(embeddings)
            stored_count += vector_store.upsert_chunks(batch, embeddings)

    return IngestionSummary(
        pdf_count=len(pdfs),
        page_count=page_count,
        chunk_count=chunk_count,
        embedding_count=embedding_count,
        stored_count=stored_count,
    )


def ingest_paths(
    inputs: Sequence[str | Path],
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    *,
    batch_size: int = 32,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> IngestionSummary:
    """Ingest every discovered PDF and return an auditable summary."""
    pdfs = discover_pdfs(inputs)
    return _ingest_pdfs(
        pdfs,
        embedding_service,
        vector_store,
        batch_size=batch_size,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed processed PDF chunks into persistent Chroma storage."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[DEFAULT_PDF_DIRECTORY],
        help=(
            "PDF files or directories to scan recursively "
            "(default: data/raw/india/patents)"
        ),
    )
    default_vector_path = settings.vector_db_path
    if not default_vector_path.is_absolute():
        default_vector_path = PROJECT_ROOT / default_vector_path
    parser.add_argument("--vector-db-path", type=Path, default=default_vector_path)
    parser.add_argument("--collection", default=settings.vector_collection)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pdfs = discover_pdfs(args.paths)
    if not pdfs:
        searched = ", ".join(str(Path(path)) for path in args.paths)
        print(f"No PDF documents found under: {searched}")
        print("Documents found: 0")
        return 0

    embedder = BGEEmbeddingService(model_name=args.model)
    vector_store = ChromaVectorStore(
        path=args.vector_db_path,
        collection_name=args.collection,
    )
    summary = _ingest_pdfs(
        pdfs,
        embedder,
        vector_store,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    print(f"Documents found: {summary.pdf_count}")
    print(f"Pages processed: {summary.page_count}")
    print(f"Chunks created: {summary.chunk_count}")
    print(f"Embeddings generated: {summary.embedding_count}")
    print(f"Chunks stored: {summary.stored_count}")
    print(f"Collection name: {vector_store.collection_name}")
    print(f"Vector database path: {vector_store.path.resolve()}")
    print(f"Total vectors in collection: {vector_store.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
