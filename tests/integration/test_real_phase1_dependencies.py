from pathlib import Path

import pytest

from database.vector.vector_store import ChromaVectorStore
from rag.ingestion.pipeline import process_pdf


fitz = pytest.importorskip("fitz")
pytest.importorskip("chromadb")


def _write_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Source-backed Ayurveda regulation evidence.")
    document.save(path)
    document.close()


def test_pdf_chunks_round_trip_through_persistent_chroma(tmp_path) -> None:
    pdf_path = tmp_path / "authority.pdf"
    _write_pdf(pdf_path)
    chunks = process_pdf(pdf_path, chunk_size=200, overlap=20)

    assert len(chunks) == 1
    assert chunks[0]["source"] == "authority.pdf"
    assert chunks[0]["page"] == 1
    assert chunks[0]["path"] == str(pdf_path.resolve())

    database_path = tmp_path / "chroma"
    first_store = ChromaVectorStore(database_path, "phase1-smoke")
    assert first_store.upsert_chunks(chunks, [[1.0, 0.0]]) == 1

    reopened_store = ChromaVectorStore(database_path, "phase1-smoke")
    assert reopened_store.upsert_chunks(chunks, [[1.0, 0.0]]) == 1
    result = reopened_store.query([1.0, 0.0], top_k=1)

    assert reopened_store.count() == 1
    assert result[0]["chunk_id"] == chunks[0]["chunk_id"]
    assert result[0]["metadata"]["source"] == "authority.pdf"
    assert result[0]["metadata"]["page"] == 1
    assert result[0]["metadata"]["path"] == str(pdf_path.resolve())
