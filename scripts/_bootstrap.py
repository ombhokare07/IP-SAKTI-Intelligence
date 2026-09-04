"""Shared bootstrap for running scripts directly from the repository checkout."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_project_root() -> Path:
    """Make root-level packages importable for ``python scripts/<name>.py``."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


def configure_console_output() -> None:
    """Use UTF-8 for extracted PDF text on Windows command prompts."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
