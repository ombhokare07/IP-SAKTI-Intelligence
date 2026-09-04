import pytest

from rag.retrieval.retriever import ChromaRetriever


class FakeEmbedder:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.questions.append(text)
        return [0.25, 0.75]


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[float], int | None]] = []

    def query(self, embedding, *, top_k=None):
        self.calls.append((list(embedding), top_k))
        return [{"chunk_id": "chunk-1", "text": "evidence", "distance": 0.1}]


def test_chroma_retriever_embeds_and_queries_configured_top_k() -> None:
    embedder = FakeEmbedder()
    store = FakeStore()
    retriever = ChromaRetriever(embedder, store, top_k=3)

    assert retriever.retrieve("  patent question  ")[0]["chunk_id"] == "chunk-1"
    assert embedder.questions == ["patent question"]
    assert store.calls == [([0.25, 0.75], 3)]


def test_chroma_retriever_rejects_blank_questions() -> None:
    retriever = ChromaRetriever(FakeEmbedder(), FakeStore())

    with pytest.raises(ValueError, match="Question cannot be empty"):
        retriever.retrieve("   ")
