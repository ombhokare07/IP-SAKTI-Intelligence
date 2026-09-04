from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rag.citations.citation_generator import chunk_value


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_source(
    citation: Mapping[str, Any],
    *,
    retrieved_chunks: Sequence[Mapping[str, Any]] | None = None,
    check_file_exists: bool = True,
) -> dict[str, Any]:
    """Validate citation metadata without filling or guessing missing values."""
    issues: list[str] = []
    citation_id = chunk_value(citation, "citation_id")
    chunk_id = chunk_value(citation, "chunk_id")
    source = chunk_value(citation, "source")
    page = chunk_value(citation, "page")
    text = chunk_value(citation, "text") or chunk_value(citation, "evidence")
    path = chunk_value(citation, "path")

    if not _is_positive_int(citation_id):
        issues.append("citation_id is missing or is not a positive integer")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        issues.append("chunk_id is missing")
    if not isinstance(source, str) or not source.strip():
        issues.append("source is missing")
    if not _is_positive_int(page):
        issues.append("page is missing or is not a positive integer")
    if not isinstance(text, str) or not text.strip():
        issues.append("citation text/evidence is missing")
    if check_file_exists and isinstance(path, str) and path.strip():
        if not Path(path).expanduser().is_file():
            issues.append("source path does not exist")

    metadata = citation.get("metadata")
    if isinstance(metadata, Mapping):
        for field in ("chunk_id", "source", "page", "path"):
            top_value = citation.get(field)
            metadata_value = metadata.get(field)
            if (
                top_value not in (None, "")
                and metadata_value not in (None, "")
                and str(top_value) != str(metadata_value)
            ):
                issues.append(f"{field} conflicts with nested metadata")

    if retrieved_chunks is not None and isinstance(chunk_id, str) and chunk_id:
        matching = [
            chunk
            for chunk in retrieved_chunks
            if str(chunk_value(chunk, "chunk_id") or "") == chunk_id
        ]
        if not matching:
            issues.append("chunk_id was not present in retrieved evidence")
        else:
            retrieved = matching[0]
            for field, cited_value in (
                ("source", source),
                ("page", page),
                ("path", path),
                ("text", text),
            ):
                retrieved_value = chunk_value(retrieved, field)
                if (
                    cited_value not in (None, "")
                    and retrieved_value not in (None, "")
                    and str(cited_value) != str(retrieved_value)
                ):
                    issues.append(f"{field} does not match retrieved evidence")

    return {"valid": not issues, "issues": issues}


def validate_sources(
    citations: Sequence[Mapping[str, Any]],
    *,
    retrieved_chunks: Sequence[Mapping[str, Any]] | None = None,
    check_file_exists: bool = True,
) -> dict[str, Any]:
    """Validate a citation collection and retain per-citation results."""
    results: list[dict[str, Any]] = []
    issues: list[str] = []
    for citation in citations:
        result = validate_source(
            citation,
            retrieved_chunks=retrieved_chunks,
            check_file_exists=check_file_exists,
        )
        item = {
            "citation_id": chunk_value(citation, "citation_id"),
            **result,
        }
        results.append(item)
        issues.extend(
            f"Citation {item['citation_id']}: {issue}" for issue in result["issues"]
        )

    if not citations:
        issues.append("No citations were provided")
    valid_count = sum(1 for result in results if result["valid"])
    return {
        "valid": bool(citations) and valid_count == len(citations),
        "issues": issues,
        "results": results,
        "valid_count": valid_count,
        "total_count": len(citations),
    }


validate_citation = validate_source
