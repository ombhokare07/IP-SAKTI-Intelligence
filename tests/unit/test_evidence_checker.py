from intelligence.trust.evidence_checker import check_evidence


def _chunk(chunk_id: str, distance: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": "A manufacturing license is required by the regulation.",
        "metadata": {
            "chunk_id": chunk_id,
            "source": "authority.pdf",
            "page": 1,
        },
        "distance": distance,
    }


def test_sufficient_evidence_preserves_metadata() -> None:
    result = check_evidence(
        "Is a manufacturing license required?",
        [_chunk("strong", 0.2), _chunk("weak", 1.4)],
        relevance_threshold=0.7,
        sufficiency_threshold=0.8,
    )

    assert result["sufficient"] is True
    assert result["score"] == 0.9
    assert [chunk["chunk_id"] for chunk in result["relevant_chunks"]] == ["strong"]
    assert result["relevant_chunks"][0]["metadata"]["source"] == "authority.pdf"
    assert [chunk["chunk_id"] for chunk in result["rejected_chunks"]] == ["weak"]


def test_insufficient_evidence_returns_no_relevant_chunks() -> None:
    result = check_evidence(
        "Is a manufacturing license required?",
        [_chunk("weak", 1.4)],
        relevance_threshold=0.7,
        sufficiency_threshold=0.8,
    )

    assert result["sufficient"] is False
    assert result["relevant_chunks"] == []
    assert result["rejected_chunks"][0]["chunk_id"] == "weak"


def test_lexical_relevance_is_used_when_retriever_has_no_score() -> None:
    result = check_evidence(
        "manufacturing license required",
        [{"chunk_id": "one", "text": "A manufacturing license is required."}],
        relevance_threshold=0.9,
        sufficiency_threshold=0.9,
    )

    assert result["sufficient"] is True
    assert result["score"] == 1.0
