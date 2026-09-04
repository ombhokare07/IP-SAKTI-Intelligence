from intelligence.trust.confidence_score import calculate_evidence_strength


def _complete_chunk(index: int, score: float = 0.9) -> dict:
    return {
        "citation_id": index,
        "chunk_id": f"chunk-{index}",
        "source": f"source-{index}.pdf",
        "page": index,
        "text": "Relevant regulatory evidence.",
        "relevance_score": score,
    }


def test_strong_evidence_uses_objective_coverage_factors() -> None:
    result = calculate_evidence_strength(
        [_complete_chunk(1), _complete_chunk(2), _complete_chunk(3)]
    )

    assert result["evidence_score"] >= 85
    assert result["level"] == "strong"
    assert result["factors"]["source_coverage"] == 100
    assert result["factors"]["citation_completeness"] == 100


def test_weak_evidence_reports_missing_metadata() -> None:
    result = calculate_evidence_strength(
        [{"chunk_id": "one", "text": "Thin evidence", "relevance_score": 0.2}]
    )

    assert result["evidence_score"] < 40
    assert result["level"] == "weak"
    assert result["factors"]["citation_completeness"] == 0
