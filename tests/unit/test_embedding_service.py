from rag.embeddings.embedding_service import BGEEmbeddingService, BGE_QUERY_INSTRUCTION


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return [[index + 0.25, index + 0.75] for index, _ in enumerate(texts)]


def test_bge_embeds_documents_as_normalized_float_lists() -> None:
    model = FakeModel()
    service = BGEEmbeddingService(model=model)

    vectors = service.embed_documents(["first", "second"], batch_size=2)

    assert vectors == [[0.25, 0.75], [1.25, 1.75]]
    assert model.calls[0][0] == ["first", "second"]
    assert model.calls[0][1]["normalize_embeddings"] is True
    assert model.calls[0][1]["batch_size"] == 2


def test_bge_adds_retrieval_instruction_to_queries() -> None:
    model = FakeModel()
    service = BGEEmbeddingService(model=model)

    assert service.embed_query("  turmeric regulation  ") == [0.25, 0.75]
    assert model.calls[0][0] == [f"{BGE_QUERY_INSTRUCTION}turmeric regulation"]
