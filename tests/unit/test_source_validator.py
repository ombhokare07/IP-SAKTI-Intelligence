from intelligence.trust.source_validator import validate_source, validate_sources


def test_valid_citation_matches_retrieved_chunk(tmp_path) -> None:
    source_path = tmp_path / "authority.pdf"
    source_path.write_text("evidence", encoding="utf-8")
    chunk = {
        "chunk_id": "chunk-1",
        "text": "A license is required.",
        "source": "authority.pdf",
        "page": 2,
        "path": str(source_path),
    }
    citation = {"citation_id": 1, **chunk}

    assert validate_source(citation, retrieved_chunks=[chunk]) == {
        "valid": True,
        "issues": [],
    }


def test_missing_source_metadata_is_invalid() -> None:
    result = validate_source(
        {"citation_id": 1, "chunk_id": "chunk-1", "page": 1, "text": "Evidence"}
    )

    assert result["valid"] is False
    assert "source is missing" in result["issues"]


def test_invalid_citation_id_is_reported() -> None:
    result = validate_sources(
        [
            {
                "citation_id": 0,
                "chunk_id": "chunk-1",
                "source": "source.pdf",
                "page": 1,
                "text": "Evidence",
            }
        ],
        check_file_exists=False,
    )

    assert result["valid"] is False
    assert any("citation_id" in issue for issue in result["issues"])
