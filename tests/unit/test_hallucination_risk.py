from intelligence.trust.hallucination_detector import detect_hallucination_risk


def test_grounded_answer_has_low_hallucination_risk() -> None:
    chunk = {
        "chunk_id": "one",
        "source": "authority.pdf",
        "page": 2,
        "text": "A license is required.",
    }
    citation = {"citation_id": 1, **chunk}

    result = detect_hallucination_risk(
        "A license is required [1].",
        [citation],
        retrieved_chunks=[chunk],
        evidence_check={"sufficient": True},
    )

    assert result == {"risk": "low", "score": 0, "issues": []}


def test_fabricated_legal_citation_has_high_risk() -> None:
    result = detect_hallucination_risk(
        "Section 42 legally requires approval [9].",
        [],
        retrieved_chunks=[],
        evidence_check={"sufficient": True},
    )

    assert result["risk"] == "high"
    assert any("Unknown citation" in issue for issue in result["issues"])
