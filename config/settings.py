from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env."""

    app_name: str = "IP-SAKTI Intelligence"
    app_env: str = "development"
    debug: bool = True

    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-2.5-flash"

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    vector_db_path: Path = Path("chroma_db")
    vector_collection: str = "ip_sakti_documents"

    top_k: int = Field(default=5, gt=0)
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    evidence_relevance_threshold: float = Field(default=0.55, ge=0, le=1)
    evidence_sufficiency_threshold: float = Field(default=0.55, ge=0, le=1)
    evidence_min_relevant_chunks: int = Field(default=1, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def validate_chunk_window(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


settings = Settings()
