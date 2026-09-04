import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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

from config.settings import settings
from database.vector.vector_store import ChromaVectorStore
from intelligence.trust.confidence_score import calculate_evidence_strength
from intelligence.trust.evidence_checker import check_evidence
from intelligence.trust.source_validator import validate_sources
from rag.citations.citation_generator import build_citations, chunk_value
from rag.embeddings.embedding_service import BGEEmbeddingService
from rag.retrieval.retriever import ChromaRetriever


DEFAULT_QUESTION = "Can an Ayurvedic formulation be patented in India?"


def build_parser() -> argparse.ArgumentParser:
    default_path = settings.vector_db_path
    if not default_path.is_absolute():
        default_path = PROJECT_ROOT / default_path
    parser = argparse.ArgumentParser(
        description="Test embedding-to-Chroma retrieval and trust checks without Gemini."
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--vector-db-path", type=Path, default=default_path)
    parser.add_argument("--collection", default=settings.vector_collection)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def print_result(index: int, result: Mapping[str, Any]) -> None:
    print(f"\nRESULT {index}")
    print(f"Similarity/Distance: {result.get('distance', 'N/A')}")
    print(f"Source: {chunk_value(result, 'source') or 'UNKNOWN'}")
    print(f"Page: {chunk_value(result, 'page') or 'UNKNOWN'}")
    print(f"Chunk ID: {chunk_value(result, 'chunk_id') or 'UNKNOWN'}")
    print("Evidence:")
    print(str(chunk_value(result, "text") or "")[:500])


def trust_results(question: str, results: Sequence[Mapping[str, Any]]) -> None:
    evidence = check_evidence(question, results)
    citations = build_citations(results)
    scoring_chunks = [
        {**dict(chunk), "citation_id": citation["citation_id"]}
        for chunk, citation in zip(results, citations, strict=True)
    ]
    strength = calculate_evidence_strength(scoring_chunks, question=question)
    sources = validate_sources(citations, retrieved_chunks=results)

    print("\nEVIDENCE SUFFICIENT:")
    print(str(evidence["sufficient"]).lower())
    print("\nEVIDENCE SCORE:")
    print(f"{strength['evidence_score']}/100")
    print("\nVALID SOURCES:")
    print(f"{sources['valid_count']}/{sources['total_count']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = ChromaVectorStore(args.vector_db_path, args.collection)
    if store.count() <= 0:
        print(
            f"Collection '{store.collection_name}' contains no chunks. "
            "Run python -m scripts.ingest_documents first."
        )
        return 1

    retriever = ChromaRetriever(
        BGEEmbeddingService(model_name=args.model),
        store,
        top_k=args.top_k,
    )
    results = retriever.retrieve(args.question)
    print(f"QUESTION:\n{args.question}")
    for index, result in enumerate(results, start=1):
        print_result(index, result)
    trust_results(args.question, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
