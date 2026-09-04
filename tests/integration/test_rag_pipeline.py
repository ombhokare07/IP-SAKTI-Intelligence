from scripts import ingest_documents


class FakeEmbedder:
    def __init__(self) -> None:
        self.text_batches: list[list[str]] = []

    def embed_documents(self, texts, *, batch_size=32):
        self.text_batches.append(list(texts))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.stored = []

    def upsert_chunks(self, chunks, embeddings):
        self.stored.extend(zip(chunks, embeddings, strict=True))
        return len(chunks)


def test_ingestion_embeds_and_stores_processed_chunks(tmp_path, monkeypatch) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"placeholder")
    chunks = [
        {
            "chunk_id": f"chunk-{index}",
            "text": f"text-{index}",
            "source": "source.pdf",
            "page": index,
            "path": str(pdf),
            "chunk_index_on_page": 1,
        }
        for index in range(1, 4)
    ]
    monkeypatch.setattr(ingest_documents, "process_pdf", lambda *args, **kwargs: chunks)
    embedder = FakeEmbedder()
    store = FakeVectorStore()

    summary = ingest_documents.ingest_paths(
        [tmp_path], embedder, store, batch_size=2
    )

    assert summary.pdf_count == 1
    assert summary.chunk_count == 3
    assert embedder.text_batches == [["text-1", "text-2"], ["text-3"]]
    assert [item[0]["chunk_id"] for item in store.stored] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
