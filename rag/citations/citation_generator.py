from collections.abc import Mapping, Sequence
from typing import Any


def chunk_value(chunk: Mapping[str, Any], name: str) -> Any:
    """Read a chunk field while supporting Chroma's nested metadata shape."""
    value = chunk.get(name)
    if value not in (None, ""):
        return value
    metadata = chunk.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(name)
    return None


def build_citations(
    chunks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create stable display citations without inventing missing metadata."""
    citations: list[dict[str, Any]] = []
    for citation_id, chunk in enumerate(chunks, start=1):
        citation: dict[str, Any] = {"citation_id": citation_id}
        for field in ("chunk_id", "source", "page", "path"):
            value = chunk_value(chunk, field)
            if value not in (None, ""):
                citation[field] = value
        text = chunk_value(chunk, "text")
        if text not in (None, ""):
            citation["text"] = text
        citations.append(citation)
    return citations
