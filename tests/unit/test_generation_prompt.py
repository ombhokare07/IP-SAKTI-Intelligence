from rag.generation.context_builder import build_context
from rag.generation.prompt_builder import build_grounded_prompt


def test_grounded_prompt_exposes_evidence_and_citation_contract() -> None:
    chunks = [{"text": "Retrieved evidence."}]
    citations = [{"citation_id": 1, "source": "source.pdf", "page": 4}]

    context = build_context(chunks, citations)
    prompt = build_grounded_prompt("Question?", context)

    assert "[1] Source: source.pdf; Page: 4" in prompt
    assert "Retrieved evidence." in prompt
    assert "do not state an uncited yes/no conclusion" in prompt
    assert "[2][3]" in prompt
