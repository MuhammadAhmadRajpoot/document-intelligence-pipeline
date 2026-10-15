"""Application settings loaded from environment variables and an optional `.env` file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden with a `DOCINTEL_*` env var."""

    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # LLM
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    llm_base_url: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-20b"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=4, ge=0, description="HTTP-level retries (429/5xx).")
    response_format: Literal["json_object", "json_schema"] = "json_object"
    max_repair_attempts: int = Field(default=2, ge=0, le=5)
    max_prompt_chars: int = Field(default=12_000, gt=1000)

    # Ingestion / OCR
    pdf_render_dpi: int = Field(default=200, ge=72, le=400)
    min_text_layer_chars: int = Field(
        default=25, ge=0, description="Below this, a PDF page is treated as scanned and OCR'd."
    )
    max_pages: int = Field(default=10, ge=1)
    ocr_min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    preprocess: bool = True

    # Validation
    amount_abs_tolerance: float = Field(default=0.02, ge=0)
    amount_rel_tolerance: float = Field(default=0.005, ge=0)
    review_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # API
    max_upload_mb: int = Field(default=15, ge=1)
    log_level: str = "INFO"
    log_json: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
