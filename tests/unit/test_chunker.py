import pytest

from rag.ingestion.chunker import chunk_text


def test_chunk_text_creates_overlap() -> None:
    assert chunk_text("abcdefghij", chunk_size=6, overlap=2) == [
        "abcdef",
        "efghij",
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (4, -1), (4, 4), (4, 5)],
)
def test_chunk_text_rejects_invalid_windows(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("abcdef", chunk_size=chunk_size, overlap=overlap)


def test_chunk_text_ignores_empty_input() -> None:
    assert chunk_text("", chunk_size=10, overlap=2) == []
