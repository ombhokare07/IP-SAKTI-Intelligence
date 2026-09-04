import argparse
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    from _bootstrap import PROJECT_ROOT, ensure_project_root
else:
    from scripts._bootstrap import PROJECT_ROOT, ensure_project_root

ensure_project_root()

from config.settings import settings
from database.vector.vector_store import ChromaVectorStore


def build_parser() -> argparse.ArgumentParser:
    default_path = settings.vector_db_path
    if not default_path.is_absolute():
        default_path = PROJECT_ROOT / default_path
    parser = argparse.ArgumentParser(description="Verify persistent Chroma storage.")
    parser.add_argument("--vector-db-path", type=Path, default=default_path)
    parser.add_argument("--collection", default=settings.vector_collection)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = ChromaVectorStore(args.vector_db_path, args.collection)
    count = store.count()

    print(f"Vector database path: {store.path.resolve()}")
    print(f"Collection: {store.collection_name}")
    print(f"Stored chunks: {count}")
    if count <= 0:
        print("Vector store verification failed: collection contains no chunks.")
        return 1

    sample = store.collection.peek(limit=1)
    metadata = (sample.get("metadatas") or [{}])[0] or {}
    missing = [
        field
        for field in ("source", "page", "path", "chunk_id")
        if metadata.get(field) in (None, "")
    ]
    if missing:
        print(f"Vector store verification failed: missing metadata {', '.join(missing)}")
        return 1

    print("\nSample metadata:")
    print(f"source: {metadata['source']}")
    print(f"page: {metadata['page']}")
    print(f"chunk_id: {metadata['chunk_id']}")
    print(f"path: {metadata['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
