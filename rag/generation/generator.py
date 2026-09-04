from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from rag.generation.context_builder import build_context
from rag.generation.prompt_builder import build_grounded_prompt


class LLMService(Protocol):
    def generate(self, prompt: str) -> str: ...


class GroundedAnswerGenerator:
    """Generate only after retrieval and the evidence gate have succeeded."""

    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    def generate(
        self,
        *,
        question: str,
        chunks: Sequence[Mapping[str, Any]],
        citations: Sequence[Mapping[str, Any]],
    ) -> str:
        context = build_context(chunks, citations)
        prompt = build_grounded_prompt(question, context)
        return self.llm_service.generate(prompt)
