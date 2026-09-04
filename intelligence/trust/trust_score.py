from collections.abc import Mapping
from typing import Any


DISCLAIMER = (
    "Trust score reflects grounding quality and evidence consistency within the "
    "current knowledge base; it is not a probability of legal correctness."
)


def _level(score: int) -> str:
    if score < 40:
        return "low"
    if score < 70:
        return "moderate"
    if score < 85:
        return "high"
    return "very_high"


def calculate_trust_score(
    evidence_strength: Mapping[str, Any],
    citation_validation: Mapping[str, Any],
    claim_verification: Mapping[str, Any],
    hallucination_risk: Mapping[str, Any],
    contradictions: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine grounding signals into a transparent, non-probabilistic score."""
    evidence_score = float(evidence_strength.get("evidence_score", 0))
    total_citations = int(citation_validation.get("total_count", 0))
    valid_citations = int(citation_validation.get("valid_count", 0))
    citation_quality = (
        valid_citations / total_citations * 100 if total_citations else 0.0
    )

    unsupported_count = len(claim_verification.get("unsupported_claims", []))
    claim_quality = max(0.0, 100.0 - unsupported_count * 25.0)
    if evidence_score <= 0:
        claim_quality = 0.0

    hallucination_score = float(hallucination_risk.get("score", 0))
    contradiction_penalty = 0.0
    if contradictions.get("detected"):
        contradiction_penalty += 12.0
    if contradictions.get("requires_review"):
        contradiction_penalty += 8.0

    raw_score = (
        evidence_score * 0.55
        + citation_quality * 0.25
        + claim_quality * 0.20
        - hallucination_score * 0.25
        - contradiction_penalty
    )
    score = round(max(0.0, min(100.0, raw_score)))

    explanation = [
        f"Evidence strength contributed from a score of {round(evidence_score)}/100.",
        f"{valid_citations} of {total_citations} citations passed metadata validation.",
        f"{unsupported_count} substantive claim(s) lacked valid citation support.",
        f"Hallucination risk signal was {hallucination_risk.get('risk', 'unknown')} "
        f"({round(hallucination_score)}/100).",
    ]
    if contradictions.get("detected"):
        explanation.append("Evidence metadata conflicts require review.")
    else:
        explanation.append("No conservative evidence metadata conflict was detected.")

    return {
        "trust_score": score,
        "level": _level(score),
        "explanation": explanation,
        "disclaimer": DISCLAIMER,
    }


compute_trust_score = calculate_trust_score
