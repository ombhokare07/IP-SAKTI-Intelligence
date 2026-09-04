from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import SecretStr

from backend.main import RAGRuntime, build_rag_runtime, create_app
from config.settings import Settings
from rag.generation.generator import GroundedAnswerGenerator
from rag.pipeline import RAGPipeline
from services.llm_service import GeminiLLMService


class FakeModels:
    def __init__(self, *, text: str | None = None, error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text)


class FakeGeminiClient:
    def __init__(self, *, text: str | None = None, error: Exception | None = None):
        self.models = FakeModels(text=text, error=error)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question: str):
        return self.chunks


def configured_settings(tmp_path, *, secret: str = "unit-test-secret") -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key=SecretStr(secret),
        gemini_model="gemini-2.5-flash",
        vector_db_path=tmp_path / "chroma",
        vector_collection="startup-test",
    )


def test_lifespan_initializes_real_phase1_components(tmp_path) -> None:
    fake_client = FakeGeminiClient(text="Grounded answer [1].")
    runtime_settings = configured_settings(tmp_path)
    application = create_app(
        runtime_settings,
        runtime_builder=lambda current: build_rag_runtime(
            current, llm_client=fake_client
        ),
    )

    with TestClient(application):
        pipeline = application.state.rag_pipeline
        assert isinstance(pipeline, RAGPipeline)
        assert pipeline.retriever.vector_store.collection_name == "startup-test"
        assert pipeline.generator.llm_service.model_name == "gemini-2.5-flash"
        assert application.state.rag_configuration_error is None

    assert fake_client.closed is True


def test_startup_failure_is_generic_and_does_not_expose_key(tmp_path) -> None:
    secret = "must-not-appear"
    runtime_settings = configured_settings(tmp_path, secret=secret)

    def fail_to_build(_settings):
        raise RuntimeError(f"provider rejected {secret}")

    application = create_app(runtime_settings, runtime_builder=fail_to_build)
    with TestClient(application) as client:
        response = client.post("/api/chat", json={"question": "Question?"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The RAG pipeline could not be initialized."
    }
    assert secret not in response.text


def test_mocked_gemini_request_runs_full_rag_through_api(tmp_path) -> None:
    source_path = tmp_path / "authority.pdf"
    source_path.write_bytes(b"source")
    chunks = [
        {
            "chunk_id": "chunk-1",
            "text": (
                "Ayush inventions are examined under the Patents Act and each "
                "application depends on its individual merits."
            ),
            "source": "authority.pdf",
            "page": 9,
            "path": str(source_path),
            "distance": 0.1,
        }
    ]
    fake_client = FakeGeminiClient(
        text=(
            "An Ayurvedic formulation may be patentable when it meets the "
            "statutory requirements [1]."
        )
    )
    llm_service = GeminiLLMService(
        "unit-test-secret", "gemini-2.5-flash", client=fake_client
    )
    pipeline = RAGPipeline(
        FakeRetriever(chunks),
        GroundedAnswerGenerator(llm_service),
    )
    runtime = RAGRuntime(pipeline=pipeline, llm_service=llm_service)
    application = create_app(
        configured_settings(tmp_path), runtime_builder=lambda _settings: runtime
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/chat",
            json={"question": "Can an Ayurvedic formulation be patented in India?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "grounded"
    assert body["citations"][0]["source"] == "authority.pdf"
    assert body["trust"]["hallucination_risk"] == "low"
    assert body["trust"]["unsupported_claims"] == 0
    call = fake_client.models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert "Ayush inventions are examined" in call["contents"]


def test_gemini_provider_failure_returns_controlled_error(tmp_path) -> None:
    secret = "must-not-appear"
    source_path = tmp_path / "authority.pdf"
    source_path.write_bytes(b"source")
    chunks = [
        {
            "chunk_id": "chunk-1",
            "text": "Ayush inventions are examined under the Patents Act.",
            "source": "authority.pdf",
            "page": 9,
            "path": str(source_path),
            "distance": 0.1,
        }
    ]
    fake_client = FakeGeminiClient(error=RuntimeError(f"network error {secret}"))
    llm_service = GeminiLLMService(
        secret, "gemini-2.5-flash", client=fake_client
    )
    runtime = RAGRuntime(
        pipeline=RAGPipeline(
            FakeRetriever(chunks), GroundedAnswerGenerator(llm_service)
        ),
        llm_service=llm_service,
    )
    application = create_app(
        configured_settings(tmp_path, secret=secret),
        runtime_builder=lambda _settings: runtime,
    )

    with TestClient(application) as client:
        response = client.post("/api/chat", json={"question": "Question?"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Gemini service request failed. Please try again."
    }
    assert secret not in response.text
