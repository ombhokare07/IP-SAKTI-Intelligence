from rag.citations.claim_mapper import map_claims


CHUNK = {
    "chunk_id": "chunk-1",
    "source": "authority.pdf",
    "page": 1,
    "text": "A manufacturing license is required.",
}
CITATION = {"citation_id": 1, **CHUNK}


def test_cited_claim_maps_to_retrieved_evidence() -> None:
    result = map_claims(
        "A manufacturing license is required [1].",
        [CITATION],
        retrieved_chunks=[CHUNK],
    )

    assert result["claims"][0]["supported"] is True
    assert result["unsupported_claims"] == []


def test_fabricated_citation_marker_is_not_supported() -> None:
    result = map_claims(
        "A manufacturing license is required [99].",
        [CITATION],
        retrieved_chunks=[CHUNK],
    )

    assert result["invalid_citation_ids"] == [99]
    assert result["claims"][0]["supported"] is False


def test_uncited_substantive_claim_is_reported() -> None:
    result = map_claims(
        "A manufacturing license is required.",
        [CITATION],
        retrieved_chunks=[CHUNK],
    )

    assert result["unsupported_claims"] == ["A manufacturing license is required."]
