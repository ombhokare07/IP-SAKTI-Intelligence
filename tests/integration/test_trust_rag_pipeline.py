from rag.pipeline import INSUFFICIENT_EVIDENCE_ANSWER, RAGPipeline


class FakeRetriever:
    def __init__(self, chunks) -> None:
        self.chunks = chunks

    def retrieve(self, question: str):
        return self.chunks


class FakeGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = []

    def generate(self, *, question, chunks, citations) -> str:
        self.calls.append((question, chunks, citations))
        return self.answer


def test_full_pipeline_with_mocked_retrieval_and_llm(tmp_path) -> None:
    source_path = tmp_path / "authority.pdf"
    source_path.write_text("source", encoding="utf-8")
    chunks = [
        {
            "chunk_id": "chunk-1",
            "text": "The application requires a manufacturing license.",
            "metadata": {
                "chunk_id": "chunk-1",
                "source": "authority.pdf",
                "page": 3,
                "path": str(source_path),
            },
            "distance": 0.1,
        }
    ]
    generator = FakeGenerator("A manufacturing license is required [1].")
    pipeline = RAGPipeline(FakeRetriever(chunks), generator)

    result = pipeline.run("Is a manufacturing license required?")

    assert result["status"] == "grounded"
    assert result["citations"][0]["source"] == "authority.pdf"
    assert result["citations"][0]["page"] == 3
    assert result["trust"]["hallucination_risk"] == "low"
    assert result["trust"]["unsupported_claims"] == 0
    assert len(generator.calls) == 1


def test_insufficient_evidence_skips_llm() -> None:
    generator = FakeGenerator("This must not be returned.")
    pipeline = RAGPipeline(
        FakeRetriever(
            [
                {
                    "chunk_id": "weak",
                    "text": "Unrelated text.",
                    "source": "source.pdf",
                    "page": 1,
                    "distance": 1.8,
                }
            ]
        ),
        generator,
    )

    result = pipeline.run("What license is required?")

    assert result["status"] == "insufficient_evidence"
    assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    assert result["citations"] == []
    assert result["trust"]["evidence_score"] == 0
    assert result["trust"]["hallucination_risk"] == "high"
    assert generator.calls == []
