from intelligence.trust.trust_score import calculate_trust_score


def test_trust_score_rewards_grounded_consistent_output() -> None:
    result = calculate_trust_score(
        {"evidence_score": 90},
        {"valid_count": 3, "total_count": 3},
        {"unsupported_claims": []},
        {"risk": "low", "score": 0},
        {"detected": False, "requires_review": False},
    )

    assert result["trust_score"] >= 85
    assert result["level"] == "very_high"
    assert "not a probability" in result["disclaimer"]


def test_trust_score_penalizes_unsupported_risky_conflicting_output() -> None:
    result = calculate_trust_score(
        {"evidence_score": 60},
        {"valid_count": 0, "total_count": 2},
        {"unsupported_claims": ["claim one", "claim two"]},
        {"risk": "high", "score": 80},
        {"detected": True, "requires_review": True},
    )

    assert result["trust_score"] < 40
    assert result["level"] == "low"
