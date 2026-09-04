from fastapi import APIRouter

from backend.api.routes.chat import router as chat_router


api_router = APIRouter(prefix="/api")
api_router.include_router(chat_router)
