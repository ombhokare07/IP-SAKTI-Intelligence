from collections import defaultdict
from collections.abc import Mapping, Sequence
import re
from typing import Any, Protocol

from rag.citations.citation_generator import chunk_value


class OptionalSemanticDetector(Protocol):
    """Optional interface for semantic checks; Phase-1 tests do not require it."""

    def detect(self, chunks: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]: ...


_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def _explicit_requirement_assertions(text: str) -> list[tuple[str, str]]:
    """Return exact positive/negative requirement forms with a shared key."""
    assertions: list[tuple[str, str]] = []
    forms = (
        (" is not required", " is required", "negative"),
        (" are not required", " are required", "negative"),
        (" must not ", " must ", "negative"),
        (" shall not ", " shall ", "negative"),
        (" is required", " is required", "positive"),
        (" are required", " are required", "positive"),
        (" must ", " must ", "positive"),
        (" shall ", " shall ", "positive"),
    )
    for raw_sentence in _SENTENCE.split(text.casefold()):
        sentence = " ".join(raw_sentence.strip(" .;:-").split())
        for phrase, canonical_phrase, polarity in forms:
            if phrase in sentence:
                canonical = sentence.replace(phrase, canonical_phrase, 1)
                assertions.append((canonical, polarity))
                break
    return assertions


def _metadata(chunk: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = chunk.get("metadata")
    return nested if isinstance(nested, Mapping) else chunk


def _document_identity(chunk: Mapping[str, Any]) -> str | None:
    metadata = _metadata(chunk)
    for field in ("regulation_id", "document_id", "source"):
        value = metadata.get(field) or chunk_value(chunk, field)
        if value not in (None, ""):
            return str(value).strip().casefold()
    return None


def detect_contradictions(
    chunks: Sequence[Mapping[str, Any]],
    *,
    semantic_detector: OptionalSemanticDetector | None = None,
) -> dict[str, Any]:
    """Find conservative metadata conflicts without making legal conclusions."""
    conflicts: list[dict[str, Any]] = []
    documents: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        identity = _document_identity(chunk)
        if identity:
            documents[identity].append(chunk)

    for identity, group in documents.items():
        versions = sorted(
            {
                str(_metadata(chunk).get("version")).strip()
                for chunk in group
                if _metadata(chunk).get("version") not in (None, "")
            }
        )
        if len(versions) > 1:
            conflicts.append(
                {
                    "type": "document_version_conflict",
                    "document": identity,
                    "values": versions,
                    "reason": "The same document identity has multiple versions.",
                }
            )

        effective_dates = sorted(
            {
                str(_metadata(chunk).get("effective_date")).strip()
                for chunk in group
                if _metadata(chunk).get("effective_date") not in (None, "")
            }
        )
        if len(effective_dates) > 1:
            conflicts.append(
                {
                    "type": "effective_date_conflict",
                    "document": identity,
                    "values": effective_dates,
                    "reason": (
                        "The same document identity has different effective dates; "
                        "metadata does not establish which one controls."
                    ),
                }
            )

    requirements: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        metadata = _metadata(chunk)
        requirement = metadata.get("requirement_id") or metadata.get("requirement")
        status = (
            metadata.get("requirement_status")
            or metadata.get("obligation")
            or metadata.get("status")
        )
        if isinstance(metadata.get("required"), bool):
            status = "required" if metadata["required"] else "not_required"
        if requirement not in (None, "") and status not in (None, ""):
            requirements[str(requirement).strip().casefold()].add(
                str(status).strip().casefold()
            )

    conflicting_status_sets = (
        {"required", "not_required"},
        {"required", "prohibited"},
        {"allowed", "prohibited"},
        {"active", "repealed"},
    )
    for requirement, statuses in requirements.items():
        if any(pair <= statuses for pair in conflicting_status_sets):
            conflicts.append(
                {
                    "type": "requirement_status_conflict",
                    "requirement": requirement,
                    "values": sorted(statuses),
                    "reason": "Structured requirement metadata contains opposing statuses.",
                }
            )

    statement_polarities: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        text = str(chunk_value(chunk, "text") or "")
        for statement, polarity in _explicit_requirement_assertions(text):
            statement_polarities[statement].add(polarity)
    for statement, polarities in statement_polarities.items():
        if {"positive", "negative"} <= polarities:
            conflicts.append(
                {
                    "type": "explicit_requirement_statement_conflict",
                    "statement": statement,
                    "values": ["negative", "positive"],
                    "reason": (
                        "Retrieved evidence contains exact opposing requirement "
                        "forms and needs manual review."
                    ),
                }
            )

    if semantic_detector is not None:
        for conflict in semantic_detector.detect(chunks):
            conflicts.append(dict(conflict))

    return {
        "detected": bool(conflicts),
        "conflicts": conflicts,
        "requires_review": bool(conflicts),
    }


detect_conflicts = detect_contradictions
