from intelligence.trust.contradiction_detector import detect_contradictions


def test_document_version_metadata_requires_review() -> None:
    chunks = [
        {
            "chunk_id": "one",
            "text": "Version one",
            "metadata": {"regulation_id": "REG-1", "version": "2024"},
        },
        {
            "chunk_id": "two",
            "text": "Version two",
            "metadata": {"regulation_id": "REG-1", "version": "2025"},
        },
    ]

    result = detect_contradictions(chunks)

    assert result["detected"] is True
    assert result["requires_review"] is True
    assert result["conflicts"][0]["type"] == "document_version_conflict"


def test_no_metadata_conflict_does_not_make_semantic_claims() -> None:
    result = detect_contradictions(
        [{"chunk_id": "one", "source": "reg.pdf", "text": "Evidence text"}]
    )

    assert result == {"detected": False, "conflicts": [], "requires_review": False}


def test_exact_opposing_requirement_statements_are_flagged() -> None:
    result = detect_contradictions(
        [
            {"chunk_id": "one", "text": "A manufacturing license is required."},
            {"chunk_id": "two", "text": "A manufacturing license is not required."},
        ]
    )

    assert result["detected"] is True
    assert result["conflicts"][0]["type"] == "explicit_requirement_statement_conflict"
