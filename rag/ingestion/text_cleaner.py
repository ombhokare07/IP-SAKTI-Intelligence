import re


_PAGE_MARKER = re.compile(r"(?im)^\s*page\s+\d+\s+of\s+\d+\s*$")


def clean_text(text: str) -> str:
    """Apply conservative cleanup without altering substantive wording."""
    cleaned = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _PAGE_MARKER.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
