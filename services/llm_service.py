from typing import Any


class LLMConfigurationError(RuntimeError):
    """Raised when the configured LLM cannot be initialized safely."""


class LLMServiceError(RuntimeError):
    """Raised when the configured LLM provider cannot produce a response."""


class GeminiLLMService:
    """Small google-genai adapter that never exposes its API key."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        *,
        client: Any | None = None,
    ) -> None:
        clean_key = api_key.strip()
        if not clean_key:
            raise LLMConfigurationError("Gemini API key is not configured.")
        if not model_name.strip():
            raise LLMConfigurationError("Gemini model is not configured.")

        if client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - installed runtime only
                raise LLMConfigurationError(
                    "google-genai is required for Gemini generation."
                ) from exc
            client = genai.Client(api_key=clean_key)

        self.model_name = model_name.strip()
        self._client = client

    def generate(self, prompt: str) -> str:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Prompt cannot be empty")
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=clean_prompt,
            )
        except Exception as exc:
            raise LLMServiceError(
                "Gemini could not generate a response."
            ) from exc

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise LLMServiceError("Gemini returned an empty response.")
        return text.strip()

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
