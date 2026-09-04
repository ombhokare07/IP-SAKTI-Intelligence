from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config.settings import settings


class ChromaVectorStore:
    """Persistent Chroma collection for source-cited Phase-1 chunks."""

    def __init__(
        self,
        path: str | Path | None = None,
        collection_name: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.path = Path(path or settings.vector_db_path)
        self.collection_name = collection_name or settings.vector_collection
        if client is None:
            try:
                import chromadb
            except ImportError as exc:  # pragma: no cover - installed runtime only
                raise RuntimeError("chromadb is required for vector storage") from exc
            self.path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.path.resolve()))
        self._client = client
        self.collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _metadata(chunk: Mapping[str, Any]) -> dict[str, str | int]:
        required = ("chunk_id", "source", "page", "path")
        missing = [field for field in required if chunk.get(field) in (None, "")]
        if missing:
            raise ValueError(f"Chunk metadata missing: {', '.join(missing)}")
        metadata: dict[str, str | int] = {
            "chunk_id": str(chunk["chunk_id"]),
            "source": str(chunk["source"]),
            "page": int(chunk["page"]),
            "path": str(chunk["path"]),
        }
        if chunk.get("chunk_index_on_page") is not None:
            metadata["chunk_index_on_page"] = int(chunk["chunk_index_on_page"])
        return metadata

    def upsert_chunks(
        self,
        chunks: Sequence[Mapping[str, Any]],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Idempotently store chunks using their deterministic chunk IDs."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")
        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int]] = []
        vectors: list[list[float]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            text = str(chunk.get("text", "")).strip()
            if not text:
                raise ValueError("Chunk text cannot be empty")
            metadata = self._metadata(chunk)
            ids.append(str(metadata["chunk_id"]))
            documents.append(text)
            metadatas.append(metadata)
            vectors.append([float(value) for value in embedding])

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vectors,
        )
        return len(ids)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest chunks with their stored citation metadata."""
        limit = settings.top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be greater than zero")
        kwargs: dict[str, Any] = {
            "query_embeddings": [[float(value) for value in query_embedding]],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = dict(where)
        result = self.collection.query(**kwargs)

        def first(name: str) -> list[Any]:
            batches = result.get(name) or [[]]
            return batches[0] if batches else []

        ids = first("ids")
        documents = first("documents")
        metadatas = first("metadatas")
        distances = first("distances")
        return [
            {
                "chunk_id": chunk_id,
                "text": documents[index],
                "metadata": metadatas[index],
                "distance": distances[index],
            }
            for index, chunk_id in enumerate(ids)
        ]

    def count(self) -> int:
        return int(self.collection.count())
