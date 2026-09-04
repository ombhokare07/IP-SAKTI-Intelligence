from pathlib import Path

from rag.ingestion import pipeline


def test_process_pdf_preserves_metadata_and_stable_chunk_ids(monkeypatch) -> None:
    pages = [
        {
            "page": 7,
            "text": "abcdefghij",
            "source": "authority.pdf",
            "path": "C:\\sources\\authority.pdf",
        }
    ]
    monkeypatch.setattr(pipeline, "load_pdf", lambda _: pages)

    first = pipeline.process_pdf(Path("ignored.pdf"), chunk_size=6, overlap=2)
    second = pipeline.process_pdf(Path("ignored.pdf"), chunk_size=6, overlap=2)

    assert first == second
    assert [chunk["text"] for chunk in first] == ["abcdef", "efghij"]
    assert len({chunk["chunk_id"] for chunk in first}) == 2
    assert first[0]["source"] == "authority.pdf"
    assert first[0]["page"] == 7
    assert first[0]["path"] == "C:\\sources\\authority.pdf"
