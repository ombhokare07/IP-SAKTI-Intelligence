from collections.abc import Sequence
from typing import Any

from config.settings import settings


BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEEmbeddingService:
    """Lazy SentenceTransformers adapter for BGE document and query embeddings."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - installed runtime only
                raise RuntimeError(
                    "sentence-transformers is required for BGE embeddings"
                ) from exc
            kwargs = {"device": self.device} if self.device else {}
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    @staticmethod
    def _as_float_lists(vectors: Any) -> list[list[float]]:
        values = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        return [[float(value) for value in vector] for vector in values]

    def embed_documents(
        self, texts: Sequence[str], *, batch_size: int = 32
    ) -> list[list[float]]:
        """Embed passages with normalized vectors for cosine retrieval."""
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero")
        if not texts:
            return []
        vectors = self._get_model().encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self._as_float_lists(vectors)

    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query using the BGE query instruction."""
        if not text.strip():
            raise ValueError("Query text cannot be empty")
        vectors = self._get_model().encode(
            [f"{BGE_QUERY_INSTRUCTION}{text.strip()}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self._as_float_lists(vectors)[0]
