import re
from collections.abc import Mapping, Sequence
from typing import Any

from config.settings import settings
from rag.citations.citation_generator import chunk_value


_WORD = re.compile(r"[A-Za-z0-9]+")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lexical_relevance(question: str, text: str) -> float:
    query_terms = {term.lower() for term in _WORD.findall(question) if len(term) > 1}
    if not query_terms:
        return 0.0
    text_terms = {term.lower() for term in _WORD.findall(text)}
    return len(query_terms & text_terms) / len(query_terms)


def retrieval_similarity(question: str, chunk: Mapping[str, Any]) -> float:
    """Normalize supported retrieval signals to 0-1.

    Chroma cosine distance is normalized as ``1 - distance / 2`` because its
    cosine distance range is 0-2. Explicit similarity/relevance scores may be
    supplied as either 0-1 or 0-100. Lexical query coverage is used only when
    the retriever supplied no score.
    """
    for field in ("similarity_score", "relevance_score", "score"):
        raw = chunk_value(chunk, field)
        score = _numeric(raw)
        if score is not None:
            return _clamp(score / 100 if score > 1 else score)

    distance = _numeric(chunk_value(chunk, "distance"))
    if distance is not None:
        return _clamp(1 - (distance / 2))

    return _clamp(_lexical_relevance(question, str(chunk_value(chunk, "text") or "")))


def check_evidence(
    question: str,
    retrieved_chunks: Sequence[Mapping[str, Any]],
    *,
    relevance_threshold: float | None = None,
    sufficiency_threshold: float | None = None,
    min_relevant_chunks: int | None = None,
) -> dict[str, Any]:
    """Deterministically decide whether retrieved evidence is usable."""
    if not question.strip():
        raise ValueError("Question cannot be empty")

    relevance_cutoff = (
        settings.evidence_relevance_threshold
        if relevance_threshold is None
        else relevance_threshold
    )
    sufficient_cutoff = (
        settings.evidence_sufficiency_threshold
        if sufficiency_threshold is None
        else sufficiency_threshold
    )
    required_count = (
        settings.evidence_min_relevant_chunks
        if min_relevant_chunks is None
        else min_relevant_chunks
    )
    if not 0 <= relevance_cutoff <= 1 or not 0 <= sufficient_cutoff <= 1:
        raise ValueError("Evidence thresholds must be between 0 and 1")
    if required_count <= 0:
        raise ValueError("min_relevant_chunks must be greater than zero")

    scored: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for original in retrieved_chunks:
        chunk = dict(original)
        chunk["relevance_score"] = round(retrieval_similarity(question, original), 4)
        scored.append(chunk)
        if chunk["relevance_score"] >= relevance_cutoff:
            candidates.append(chunk)
        else:
            rejected.append(chunk)

    ranked_scores = sorted(
        (float(chunk["relevance_score"]) for chunk in candidates), reverse=True
    )[:3]
    score = sum(ranked_scores) / len(ranked_scores) if ranked_scores else 0.0
    sufficient = len(candidates) >= required_count and score >= sufficient_cutoff

    if not retrieved_chunks:
        reason = "The retriever returned no evidence."
    elif not candidates:
        reason = "No retrieved chunk met the configured relevance threshold."
    elif len(candidates) < required_count:
        reason = "Too few retrieved chunks met the configured relevance threshold."
    elif score < sufficient_cutoff:
        reason = "Relevant chunks did not meet the configured sufficiency threshold."
    else:
        reason = "Retrieved evidence met the configured relevance and sufficiency thresholds."

    if not sufficient:
        rejected = scored
        candidates = []

    return {
        "sufficient": sufficient,
        "score": round(score, 4),
        "relevant_chunks": candidates,
        "rejected_chunks": rejected,
        "reason": reason,
    }
