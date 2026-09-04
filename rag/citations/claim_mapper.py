import re
from collections.abc import Mapping, Sequence
from typing import Any

from rag.citations.citation_generator import chunk_value


_CITATION = re.compile(r"\[(\d+)\]")
_CLAIM_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[A-Za-z0-9]+")
_SUBSTANTIVE_TERMS = {
    "allowed",
    "authority",
    "compliance",
    "effective",
    "illegal",
    "law",
    "legal",
    "license",
    "mandatory",
    "must",
    "prohibited",
    "regulation",
    "required",
    "requires",
    "section",
}


def _is_substantive(text: str) -> bool:
    without_markers = _CITATION.sub("", text)
    words = [word.lower() for word in _WORD.findall(without_markers)]
    return len(words) >= 4 or bool(set(words) & _SUBSTANTIVE_TERMS)


def _citation_is_retrieved(
    citation: Mapping[str, Any], retrieved_chunks: Sequence[Mapping[str, Any]]
) -> bool:
    chunk_id = chunk_value(citation, "chunk_id")
    if chunk_id in (None, ""):
        return False
    for chunk in retrieved_chunks:
        if str(chunk_value(chunk, "chunk_id") or "") != str(chunk_id):
            continue
        cited_source = chunk_value(citation, "source")
        source = chunk_value(chunk, "source")
        return (
            cited_source in (None, "")
            or source in (None, "")
            or str(cited_source) == str(source)
        )
    return False


def map_claims(
    answer: str,
    citations: Sequence[Mapping[str, Any]],
    *,
    retrieved_chunks: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Map answer citation markers to retrieved evidence conservatively."""
    citation_by_id = {
        chunk_value(citation, "citation_id"): citation
        for citation in citations
        if isinstance(chunk_value(citation, "citation_id"), int)
    }
    claims: list[dict[str, Any]] = []
    unsupported_claims: list[str] = []
    invalid_ids: set[int] = set()

    for segment in _CLAIM_BOUNDARY.split(answer.strip()):
        text = segment.strip(" \t-*#")
        if not text or not _is_substantive(text):
            continue
        referenced_ids = list(dict.fromkeys(int(value) for value in _CITATION.findall(text)))
        known = bool(referenced_ids) and all(
            citation_id in citation_by_id for citation_id in referenced_ids
        )
        invalid_ids.update(
            citation_id
            for citation_id in referenced_ids
            if citation_id not in citation_by_id
        )

        belongs_to_retrieval = True
        if retrieved_chunks is not None and known:
            belongs_to_retrieval = all(
                _citation_is_retrieved(citation_by_id[citation_id], retrieved_chunks)
                for citation_id in referenced_ids
            )
        supported = known and belongs_to_retrieval
        claim = {
            "text": text,
            "citations": referenced_ids,
            "supported": supported,
        }
        claims.append(claim)
        if not supported:
            unsupported_claims.append(text)

    return {
        "claims": claims,
        "unsupported_claims": unsupported_claims,
        "invalid_citation_ids": sorted(invalid_ids),
    }


verify_claims = map_claims
