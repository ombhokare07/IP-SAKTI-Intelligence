from scripts.test_rag_query import format_rag_result


def test_manual_output_has_required_trust_sections() -> None:
    output = format_rag_result(
        {
            "question": "Question?",
            "answer": "Answer [1].",
            "status": "grounded",
            "citations": [{"citation_id": 1, "source": "source.pdf", "page": 2}],
            "trust": {
                "evidence_score": 86,
                "hallucination_risk": "low",
                "contradictions_detected": False,
                "trust_score": 88,
                "level": "very_high",
            },
        }
    )

    assert "EVIDENCE STRENGTH\n86/100" in output
    assert "HALLUCINATION RISK\nLOW" in output
    assert "CONTRADICTIONS\nNONE" in output
    assert "TRUST SCORE\n88/100 - VERY HIGH" in output
    assert "STATUS\nGROUNDED" in output
