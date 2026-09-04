from pathlib import Path

if __package__ in (None, ""):
    from _bootstrap import PROJECT_ROOT, configure_console_output, ensure_project_root
else:
    from scripts._bootstrap import (
        PROJECT_ROOT,
        configure_console_output,
        ensure_project_root,
    )

ensure_project_root()
configure_console_output()

from rag.ingestion.pipeline import process_pdf


PDF_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "india" / "patents"


def first_pdf(directory: Path = PDF_DIRECTORY) -> Path | None:
    pdfs = (
        sorted(
            (
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() == ".pdf"
            ),
            key=lambda path: str(path).casefold(),
        )
        if directory.exists()
        else []
    )
    return pdfs[0] if pdfs else None


def main() -> int:
    pdf_path = first_pdf()
    if pdf_path is None:
        print("No PDF found in data/raw/india/patents/")
        return 0

    chunks = process_pdf(pdf_path)
    pages = {chunk["page"] for chunk in chunks}

    print("PDF:")
    print(pdf_path.name)
    print("\nTOTAL PAGES:")
    print(len(pages))
    print("\nTOTAL CHUNKS:")
    print(len(chunks))

    for chunk in chunks[:3]:
        print("\n" + "=" * 50)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Page: {chunk['page']}\n")
        print(chunk["text"][:500])
        print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
