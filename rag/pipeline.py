from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from intelligence.trust.confidence_score import calculate_evidence_strength
from intelligence.trust.contradiction_detector import detect_contradictions
from intelligence.trust.evidence_checker import check_evidence
from intelligence.trust.hallucination_detector import detect_hallucination_risk
from intelligence.trust.source_validator import validate_sources
from intelligence.trust.trust_score import calculate_trust_score
from rag.citations.citation_generator import build_citations
from rag.citations.claim_mapper import map_claims


INSUFFICIENT_EVIDENCE_ANSWER = (
    "Insufficient reliable evidence was found in the current knowledge base."
)


class Retriever(Protocol):
    def retrieve(self, question: str) -> Sequence[Mapping[str, Any]]: ...


class GroundedAnswerGenerator(Protocol):
    def generate(
        self,
        *,
        question: str,
        chunks: Sequence[Mapping[str, Any]],
        citations: Sequence[Mapping[str, Any]],
    ) -> str: ...


class RAGPipeline:
    """Run retrieval, grounded generation, and deterministic trust checks."""

    def __init__(
        self,
        retriever: Retriever,
        generator: GroundedAnswerGenerator,
        *,
        relevance_threshold: float | None = None,
        sufficiency_threshold: float | None = None,
        min_relevant_chunks: int | None = None,
        check_source_files: bool = True,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.relevance_threshold = relevance_threshold
        self.sufficiency_threshold = sufficiency_threshold
        self.min_relevant_chunks = min_relevant_chunks
        self.check_source_files = check_source_files

    @staticmethod
    def _trust_summary(
        trust_score: Mapping[str, Any],
        evidence_strength: Mapping[str, Any],
        hallucination: Mapping[str, Any],
        contradictions: Mapping[str, Any],
        claims: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            **trust_score,
            "evidence_score": evidence_strength["evidence_score"],
            "hallucination_risk": hallucination["risk"],
            "contradictions_detected": bool(contradictions["detected"]),
            "unsupported_claims": len(claims.get("unsupported_claims", [])),
        }

    def run(self, question: str) -> dict[str, Any]:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty")

        retrieved = list(self.retriever.retrieve(clean_question))
        evidence = check_evidence(
            clean_question,
            retrieved,
            relevance_threshold=self.relevance_threshold,
            sufficiency_threshold=self.sufficiency_threshold,
            min_relevant_chunks=self.min_relevant_chunks,
        )

        if not evidence["sufficient"]:
            evidence_strength = calculate_evidence_strength([])
            claims = {
                "claims": [],
                "unsupported_claims": [],
                "invalid_citation_ids": [],
            }
            citation_validation = validate_sources([])
            contradictions = detect_contradictions([])
            hallucination = detect_hallucination_risk(
                INSUFFICIENT_EVIDENCE_ANSWER,
                [],
                retrieved_chunks=retrieved,
                claim_verification=claims,
                evidence_check=evidence,
            )
            trust_score = calculate_trust_score(
                evidence_strength,
                citation_validation,
                claims,
                hallucination,
                contradictions,
            )
            return {
                "question": clean_question,
                "answer": INSUFFICIENT_EVIDENCE_ANSWER,
                "status": "insufficient_evidence",
                "citations": [],
                "trust": self._trust_summary(
                    trust_score,
                    evidence_strength,
                    hallucination,
                    contradictions,
                    claims,
                ),
            }

        relevant_chunks = evidence["relevant_chunks"]
        citations = build_citations(relevant_chunks)
        scoring_chunks = [
            {**chunk, "citation_id": citation["citation_id"]}
            for chunk, citation in zip(relevant_chunks, citations, strict=True)
        ]
        contradictions = detect_contradictions(relevant_chunks)
        evidence_strength = calculate_evidence_strength(
            scoring_chunks,
            question=clean_question,
            contradictions=contradictions,
        )

        answer = self.generator.generate(
            question=clean_question,
            chunks=relevant_chunks,
            citations=citations,
        ).strip()
        citation_validation = validate_sources(
            citations,
            retrieved_chunks=relevant_chunks,
            check_file_exists=self.check_source_files,
        )
        claims = map_claims(
            answer,
            citations,
            retrieved_chunks=relevant_chunks,
        )
        hallucination = detect_hallucination_risk(
            answer,
            citations,
            retrieved_chunks=relevant_chunks,
            claim_verification=claims,
            evidence_check=evidence,
        )
        trust_score = calculate_trust_score(
            evidence_strength,
            citation_validation,
            claims,
            hallucination,
            contradictions,
        )
        return {
            "question": clean_question,
            "answer": answer,
            "status": "grounded",
            "citations": citations,
            "trust": self._trust_summary(
                trust_score,
                evidence_strength,
                hallucination,
                contradictions,
                claims,
            ),
        }

    answer = run
