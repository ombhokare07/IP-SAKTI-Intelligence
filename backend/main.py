from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

from backend.api.router import api_router
from config.settings import Settings, settings
from database.vector.vector_store import ChromaVectorStore
from rag.embeddings.embedding_service import BGEEmbeddingService
from rag.generation.generator import GroundedAnswerGenerator
from rag.pipeline import RAGPipeline
from rag.retrieval.retriever import ChromaRetriever
from services.llm_service import GeminiLLMService


@dataclass(frozen=True)
class RAGRuntime:
    pipeline: RAGPipeline
    llm_service: GeminiLLMService


def build_rag_runtime(
    runtime_settings: Settings,
    *,
    llm_client: Any | None = None,
) -> RAGRuntime:
    """Build the existing Phase-1 stack without ingesting or loading the BGE model."""
    api_key = runtime_settings.gemini_api_key.get_secret_value()
    embedder = BGEEmbeddingService(model_name=runtime_settings.embedding_model)
    vector_store = ChromaVectorStore(
        path=runtime_settings.vector_db_path,
        collection_name=runtime_settings.vector_collection,
    )
    retriever = ChromaRetriever(
        embedding_service=embedder,
        vector_store=vector_store,
        top_k=runtime_settings.top_k,
    )
    llm_service = GeminiLLMService(
        api_key=api_key,
        model_name=runtime_settings.gemini_model,
        client=llm_client,
    )
    generator = GroundedAnswerGenerator(llm_service)
    pipeline = RAGPipeline(
        retriever,
        generator,
        relevance_threshold=runtime_settings.evidence_relevance_threshold,
        sufficiency_threshold=runtime_settings.evidence_sufficiency_threshold,
        min_relevant_chunks=runtime_settings.evidence_min_relevant_chunks,
    )
    return RAGRuntime(pipeline=pipeline, llm_service=llm_service)


RuntimeBuilder = Callable[[Settings], RAGRuntime]


def create_app(
    runtime_settings: Settings = settings,
    *,
    runtime_builder: RuntimeBuilder = build_rag_runtime,
) -> FastAPI:
    """Create the Phase-1 API and initialize its RAG runtime once."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime: RAGRuntime | None = None
        api_key = runtime_settings.gemini_api_key.get_secret_value().strip()
        if not api_key:
            application.state.rag_pipeline = None
            application.state.rag_configuration_error = (
                "Gemini API key not configured - full RAG chat is unavailable."
            )
        else:
            try:
                runtime = runtime_builder(runtime_settings)
            except Exception:
                application.state.rag_pipeline = None
                application.state.rag_configuration_error = (
                    "The RAG pipeline could not be initialized."
                )
            else:
                application.state.rag_pipeline = runtime.pipeline
                application.state.rag_configuration_error = None

        try:
            yield
        finally:
            application.state.rag_pipeline = None
            if runtime is not None:
                runtime.llm_service.close()

    application = FastAPI(
        title=runtime_settings.app_name,
        version="0.1.0",
        description="Trustworthy source-cited RAG for Ayurveda IP intelligence",
        lifespan=lifespan,
    )
    application.state.rag_pipeline = None
    if runtime_settings.gemini_api_key.get_secret_value().strip():
        application.state.rag_configuration_error = (
            "The RAG pipeline has not been initialized."
        )
    else:
        application.state.rag_configuration_error = (
            "Gemini API key not configured - full RAG chat is unavailable."
        )

    @application.get("/")
    def root() -> dict[str, str]:
        return {
            "application": runtime_settings.app_name,
            "status": "running",
            "version": "0.1.0",
        }

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    application.include_router(api_router)
    return application


app = create_app()
