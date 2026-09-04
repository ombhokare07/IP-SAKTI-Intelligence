import argparse
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

if __package__ in (None, ""):
    from _bootstrap import configure_console_output, ensure_project_root
else:
    from scripts._bootstrap import configure_console_output, ensure_project_root

ensure_project_root()
configure_console_output()

from config.settings import settings


def format_rag_result(result: Mapping[str, Any]) -> str:
    """Format the public RAG response for a quick manual trust review."""
    citations = result.get("citations") or []
    source_lines = []
    for citation in citations:
        citation_id = citation.get("citation_id", "?")
        source = citation.get("source", "UNKNOWN SOURCE")
        page = citation.get("page")
        page_label = f", page {page}" if page is not None else ""
        source_lines.append(f"[{citation_id}] {source}{page_label}")
    sources = "\n".join(source_lines) if source_lines else "NONE"

    trust = result.get("trust") or {}
    evidence_score = trust.get("evidence_score", 0)
    hallucination = str(trust.get("hallucination_risk", "unknown")).upper()
    contradictions = "DETECTED" if trust.get("contradictions_detected") else "NONE"
    trust_score = trust.get("trust_score", 0)
    trust_level = str(trust.get("level", "unknown")).replace("_", " ").upper()
    status = str(result.get("status", "unknown")).replace("_", " ").upper()

    return (
        f"QUESTION\n{result.get('question', '')}\n\n"
        f"ANSWER\n{result.get('answer', '')}\n\n"
        f"SOURCES\n{sources}\n\n"
        f"EVIDENCE STRENGTH\n{evidence_score}/100\n\n"
        f"HALLUCINATION RISK\n{hallucination}\n\n"
        f"CONTRADICTIONS\n{contradictions}\n\n"
        f"TRUST SCORE\n{trust_score}/100 - {trust_level}\n\n"
        f"STATUS\n{status}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and inspect a RAG query.")
    parser.add_argument("question", help="Question to send to the local RAG API")
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/api/chat",
        help="Local POST /api/chat endpoint",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not settings.gemini_api_key.get_secret_value().strip():
        print(
            "Gemini API key not configured — retrieval and trust tests completed, "
            "LLM test skipped."
        )
        return 0
    try:
        response = httpx.post(
            args.api_url,
            json={"question": args.question},
            timeout=60,
        )
    except httpx.RequestError:
        print("RAG API request failed. Check that the API is running and reachable.")
        return 1
    if response.is_error:
        try:
            detail = response.json().get("detail", "Request failed")
        except ValueError:
            detail = "Request failed"
        print(f"RAG API error ({response.status_code}): {detail}")
        return 1
    print(format_rag_result(response.json()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
