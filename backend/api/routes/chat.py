from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend.api.schemas.chat_schema import ChatRequest, ChatResponse
from services.llm_service import LLMServiceError


router = APIRouter(prefix="/chat", tags=["chat"])


def _pipeline_from(request: Request) -> Any:
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(
                request.app.state,
                "rag_configuration_error",
                "The RAG pipeline has not been configured.",
            ),
        )
    return pipeline


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
    """Answer through the configured RAG pipeline and expose trust summary only."""
    pipeline = _pipeline_from(request)
    run = getattr(pipeline, "run", None) or getattr(pipeline, "answer", None)
    if run is None:
        raise HTTPException(status_code=503, detail="The RAG pipeline is invalid.")
    try:
        return run(payload.question)
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=502,
            detail="Gemini service request failed. Please try again.",
        ) from exc
