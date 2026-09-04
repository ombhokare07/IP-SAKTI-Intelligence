from config.settings import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into deterministic, overlapping character windows."""
    effective_chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    effective_overlap = settings.chunk_overlap if overlap is None else overlap

    if effective_chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")
    if effective_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative")
    if effective_overlap >= effective_chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + effective_chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - effective_overlap
    return chunks
