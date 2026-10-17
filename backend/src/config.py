from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-2"
    chunk_words: int = Field(default=500, ge=1, le=5000)
    chunk_overlap: float = Field(default=0.15, ge=0.0, lt=0.9)
    data_dir: Path = Path("data")
    top_k: int = Field(default=5, ge=1, le=100)


settings = Settings()
if not settings.gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")
if not settings.data_dir.is_absolute():
    settings.data_dir = REPO_ROOT / settings.data_dir
