from collections.abc import Mapping, Sequence
from typing import Any

from intelligence.trust.evidence_checker import retrieval_similarity
from intelligence.trust.source_validator import validate_source
from rag.citations.citation_generator import chunk_value


def _level(score: int) -> str:
    if score < 40:
        return "weak"
    if score < 70:
        return "moderate"
    if score < 85:
        return "good"
    return "strong"


def _source_coverage(chunks: Sequence[Mapping[str, Any]]) -> float:
    sources = {
        str(chunk_value(chunk, "source")).strip().casefold()
        for chunk in chunks
        if chunk_value(chunk, "source") not in (None, "")
    }
    if not sources:
        return 0.0
    return min(100.0, 50.0 + (len(sources) - 1) * 25.0)


def calculate_evidence_strength(
    chunks: Sequence[Mapping[str, Any]],
    *,
    question: str = "",
    contradictions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score objective evidence strength, not legal or LLM correctness.

    The 0-100 score weights normalized retrieval quality (45%), independent
    chunks (15%), distinct source coverage (15%), metadata agreement (10%),
    and citation completeness (15%). Three independent chunks and three
    distinct sources saturate their respective coverage factors.
    """
    if not chunks:
        factors = {
            "retrieval_quality": 0,
            "independent_evidence": 0,
            "source_coverage": 0,
            "evidence_agreement": 0,
            "citation_completeness": 0,
        }
        return {"evidence_score": 0, "level": "weak", "factors": factors}

    similarities = [retrieval_similarity(question, chunk) for chunk in chunks]
    retrieval_quality = 100 * sum(similarities) / len(similarities)

    chunk_ids = {
        str(chunk_value(chunk, "chunk_id"))
        for chunk in chunks
        if chunk_value(chunk, "chunk_id") not in (None, "")
    }
    independent_evidence = min(100.0, len(chunk_ids) / 3 * 100)
    source_coverage = _source_coverage(chunks)

    if contradictions and contradictions.get("detected"):
        evidence_agreement = 25.0 if contradictions.get("requires_review") else 60.0
    else:
        evidence_agreement = 100.0

    valid_citations = sum(
        1
        for chunk in chunks
        if validate_source(chunk, check_file_exists=False)["valid"]
    )
    citation_completeness = valid_citations / len(chunks) * 100

    raw_score = (
        retrieval_quality * 0.45
        + independent_evidence * 0.15
        + source_coverage * 0.15
        + evidence_agreement * 0.10
        + citation_completeness * 0.15
    )
    score = round(max(0.0, min(100.0, raw_score)))
    factors = {
        "retrieval_quality": round(retrieval_quality),
        "independent_evidence": round(independent_evidence),
        "source_coverage": round(source_coverage),
        "evidence_agreement": round(evidence_agreement),
        "citation_completeness": round(citation_completeness),
    }
    return {"evidence_score": score, "level": _level(score), "factors": factors}


calculate_confidence_score = calculate_evidence_strength
