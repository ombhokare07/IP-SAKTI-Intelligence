from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question cannot be blank")
        return cleaned


class CitationResponse(BaseModel):
    citation_id: int = Field(gt=0)
    chunk_id: str | None = None
    source: str | None = None
    page: int | None = Field(default=None, gt=0)
    path: str | None = None
    text: str | None = None


class TrustResponse(BaseModel):
    trust_score: int = Field(ge=0, le=100)
    level: Literal["low", "moderate", "high", "very_high"]
    evidence_score: int = Field(ge=0, le=100)
    hallucination_risk: Literal["low", "medium", "high"]
    contradictions_detected: bool
    unsupported_claims: int = Field(ge=0)
    explanation: list[str]
    disclaimer: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationResponse]
    status: Literal["grounded", "insufficient_evidence"]
    trust: TrustResponse
