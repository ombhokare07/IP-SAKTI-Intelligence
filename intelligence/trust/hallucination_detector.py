import re
from collections.abc import Mapping, Sequence
from typing import Any

from rag.citations.citation_generator import chunk_value
from rag.citations.claim_mapper import map_claims


_PDF_NAME = re.compile(r"(?i)\b([A-Za-z0-9][A-Za-z0-9_.-]*\.pdf)\b")
_PAGE = re.compile(r"(?i)\bpages?\s+(\d+)\b")
_LEGAL_SPECIFICITY = re.compile(
    r"(?i)\b(section|regulation|act|rule|authority|mandatory|prohibited|"
    r"must|required|license|effective\s+date|penalty)\b"
)


def _risk_level(score: int) -> str:
    if score < 25:
        return "low"
    if score < 55:
        return "medium"
    return "high"


def detect_hallucination_risk(
    answer: str,
    citations: Sequence[Mapping[str, Any]],
    *,
    retrieved_chunks: Sequence[Mapping[str, Any]] = (),
    claim_verification: Mapping[str, Any] | None = None,
    evidence_check: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect heuristic hallucination risk signals, not hallucinations.

    The score measures observable grounding problems in the current knowledge
    base. It is not a probability and does not establish legal correctness.
    """
    claims = claim_verification or map_claims(
        answer, citations, retrieved_chunks=retrieved_chunks
    )
    issues: list[str] = []
    score = 0

    if not answer.strip():
        issues.append("The generated answer is empty")
        score += 55

    invalid_ids = list(claims.get("invalid_citation_ids", []))
    if invalid_ids:
        issues.append(f"Unknown citation markers were used: {invalid_ids}")
        score += min(50, 30 + 5 * len(invalid_ids))

    retrieved_sources = {
        str(chunk_value(chunk, "source")).casefold()
        for chunk in retrieved_chunks
        if chunk_value(chunk, "source") not in (None, "")
    }
    named_sources = {name.casefold() for name in _PDF_NAME.findall(answer)}
    unknown_sources = sorted(named_sources - retrieved_sources)
    if unknown_sources:
        issues.append(f"Answer names sources that were not retrieved: {unknown_sources}")
        score += min(40, 20 + 5 * len(unknown_sources))

    retrieved_pages = {
        int(page)
        for chunk in retrieved_chunks
        if (page := chunk_value(chunk, "page")) is not None
        and str(page).isdigit()
    }
    named_pages = {int(page) for page in _PAGE.findall(answer)}
    unknown_pages = sorted(named_pages - retrieved_pages)
    if unknown_pages:
        issues.append(f"Answer names pages that were not retrieved: {unknown_pages}")
        score += min(35, 15 + 5 * len(unknown_pages))

    unsupported = list(claims.get("unsupported_claims", []))
    if unsupported:
        issues.append(f"{len(unsupported)} substantive claim(s) lack valid evidence citations")
        score += min(40, 10 + 8 * len(unsupported))
        if any(_LEGAL_SPECIFICITY.search(str(claim)) for claim in unsupported):
            issues.append("Unsupported legal or regulatory specificity was detected")
            score += 15

    if evidence_check is not None and not evidence_check.get("sufficient", False):
        issues.append("The evidence checker found insufficient relevant context")
        score += 60

    final_score = round(max(0, min(100, score)))
    return {
        "risk": _risk_level(final_score),
        "score": final_score,
        "issues": issues,
    }


detect_hallucinations = detect_hallucination_risk
