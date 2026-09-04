import asyncio

import httpx
from pydantic import SecretStr

from backend.main import app, create_app
from config.settings import Settings


def test_health_is_lightweight() -> None:
    async def request_health() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.get("/health")

    response = asyncio.run(request_health())

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_chat_response_includes_public_trust_schema() -> None:
    class FakePipeline:
        def run(self, question: str) -> dict:
            return {
                "question": question,
                "answer": "Grounded answer [1].",
                "status": "grounded",
                "citations": [
                    {
                        "citation_id": 1,
                        "chunk_id": "chunk-1",
                        "source": "source.pdf",
                        "page": 1,
                        "text": "Grounded evidence.",
                        "internal_debug_value": "must not leak",
                    }
                ],
                "trust": {
                    "trust_score": 88,
                    "level": "very_high",
                    "evidence_score": 90,
                    "hallucination_risk": "low",
                    "contradictions_detected": False,
                    "unsupported_claims": 0,
                    "explanation": ["Grounded."],
                    "disclaimer": "Not legal correctness.",
                    "internal_factors": {"hidden": True},
                },
            }

    async def request_chat() -> httpx.Response:
        app.state.rag_pipeline = FakePipeline()
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                return await client.post("/api/chat", json={"question": "Question?"})
        finally:
            del app.state.rag_pipeline

    response = asyncio.run(request_chat())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "grounded"
    assert body["trust"]["trust_score"] == 88
    assert "internal_factors" not in body["trust"]
    assert "internal_debug_value" not in body["citations"][0]


def test_chat_rejects_blank_question() -> None:
    async def request_chat() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post("/api/chat", json={"question": "   "})

    response = asyncio.run(request_chat())

    assert response.status_code == 422


def test_chat_reports_missing_llm_configuration() -> None:
    application = create_app(
        Settings(_env_file=None, gemini_api_key=SecretStr(""))
    )

    async def request_chat() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post("/api/chat", json={"question": "Question?"})

    response = asyncio.run(request_chat())

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Gemini API key not configured - full RAG chat is unavailable."
    }
