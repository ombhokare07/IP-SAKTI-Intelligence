from collections.abc import Sequence
from typing import Any, Protocol

from config.settings import settings
from database.vector.vector_store import ChromaVectorStore
from rag.embeddings.embedding_service import BGEEmbeddingService


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class QueryVectorStore(Protocol):
    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]: ...


class ChromaRetriever:
    """Embed a question and retrieve nearest Phase-1 chunks from Chroma."""

    def __init__(
        self,
        embedding_service: QueryEmbedder | None = None,
        vector_store: QueryVectorStore | None = None,
        *,
        top_k: int | None = None,
    ) -> None:
        self.embedding_service = embedding_service or BGEEmbeddingService()
        self.vector_store = vector_store or ChromaVectorStore()
        self.top_k = settings.top_k if top_k is None else top_k
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty")
        query_embedding = self.embedding_service.embed_query(clean_question)
        return self.vector_store.query(query_embedding, top_k=self.top_k)
