from types import SimpleNamespace

import pytest

from services.llm_service import (
    GeminiLLMService,
    LLMConfigurationError,
    LLMServiceError,
)


class FakeModels:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def generate_content(self, **_kwargs):
        if self.error is not None:
            raise self.error
        return self.result


class FakeClient:
    def __init__(self, result=None, error=None):
        self.models = FakeModels(result=result, error=error)


def test_gemini_service_returns_trimmed_text() -> None:
    service = GeminiLLMService(
        "secret",
        "gemini-2.5-flash",
        client=FakeClient(result=SimpleNamespace(text="  Answer [1].  ")),
    )

    assert service.generate("Prompt") == "Answer [1]."


def test_gemini_service_requires_configuration() -> None:
    with pytest.raises(LLMConfigurationError, match="API key"):
        GeminiLLMService("", "gemini-2.5-flash", client=FakeClient())


def test_gemini_service_hides_provider_error_details() -> None:
    secret = "must-not-appear"
    service = GeminiLLMService(
        secret,
        "gemini-2.5-flash",
        client=FakeClient(error=RuntimeError(f"failure {secret}")),
    )

    with pytest.raises(LLMServiceError) as captured:
        service.generate("Prompt")

    assert secret not in str(captured.value)
