from rag.ingestion.text_cleaner import clean_text


def test_clean_text_removes_noise_without_flattening_paragraphs() -> None:
    raw = "Hello\x00   world\r\nPage 1 of 3\r\n\r\n\r\nSecond paragraph"

    assert clean_text(raw) == "Hello world\n\nSecond paragraph"
