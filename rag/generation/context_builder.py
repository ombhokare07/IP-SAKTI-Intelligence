from collections.abc import Mapping, Sequence
from typing import Any

from rag.citations.citation_generator import chunk_value


def build_context(
    chunks: Sequence[Mapping[str, Any]],
    citations: Sequence[Mapping[str, Any]],
) -> str:
    """Render retrieved evidence with the citation IDs exposed to the model."""
    if len(chunks) != len(citations):
        raise ValueError("Chunks and citations must have the same length")
    if not chunks:
        raise ValueError("Retrieved evidence cannot be empty")

    sections: list[str] = []
    for chunk, citation in zip(chunks, citations, strict=True):
        citation_id = citation.get("citation_id")
        source = chunk_value(citation, "source") or "UNKNOWN SOURCE"
        page = chunk_value(citation, "page") or "UNKNOWN PAGE"
        text = str(chunk_value(chunk, "text") or "").strip()
        if not text:
            continue
        sections.append(
            f"[{citation_id}] Source: {source}; Page: {page}\n{text}"
        )

    if not sections:
        raise ValueError("Retrieved evidence contains no text")
    return "\n\n".join(sections)
