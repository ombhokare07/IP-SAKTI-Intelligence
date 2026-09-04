from database.vector.vector_store import ChromaVectorStore


class FakeCollection:
    def __init__(self) -> None:
        self.upsert_call = None

    def upsert(self, **kwargs) -> None:
        self.upsert_call = kwargs

    def query(self, **kwargs):
        return {
            "ids": [["chunk-1"]],
            "documents": [["document text"]],
            "metadatas": [[{"source": "source.pdf", "page": 4}]],
            "distances": [[0.125]],
        }

    def count(self) -> int:
        return 1


class FakeClient:
    def __init__(self) -> None:
        self.collection = FakeCollection()
        self.collection_args = None

    def get_or_create_collection(self, **kwargs):
        self.collection_args = kwargs
        return self.collection


def test_chroma_upsert_preserves_required_citation_metadata() -> None:
    client = FakeClient()
    store = ChromaVectorStore(client=client, collection_name="test")
    chunk = {
        "chunk_id": "chunk-1",
        "text": "document text",
        "source": "source.pdf",
        "page": 4,
        "path": "C:\\sources\\source.pdf",
        "chunk_index_on_page": 2,
    }

    assert store.upsert_chunks([chunk], [[0.1, 0.2]]) == 1
    assert client.collection_args == {
        "name": "test",
        "metadata": {"hnsw:space": "cosine"},
    }
    assert client.collection.upsert_call["ids"] == ["chunk-1"]
    assert client.collection.upsert_call["metadatas"] == [
        {
            "chunk_id": "chunk-1",
            "source": "source.pdf",
            "page": 4,
            "path": "C:\\sources\\source.pdf",
            "chunk_index_on_page": 2,
        }
    ]


def test_chroma_query_returns_text_metadata_and_distance() -> None:
    store = ChromaVectorStore(client=FakeClient(), collection_name="test")

    assert store.query([0.1, 0.2], top_k=1) == [
        {
            "chunk_id": "chunk-1",
            "text": "document text",
            "metadata": {"source": "source.pdf", "page": 4},
            "distance": 0.125,
        }
    ]
