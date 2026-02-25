"""
config.py — Centralized settings management using Pydantic Settings.
All configuration is loaded from environment variables / .env file.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    openai_embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")

    # Database
    database_url: str = Field("sqlite:///./data/sample_db/business.db", env="DATABASE_URL")

    # Vector Store
    chroma_persist_dir: str = Field("./data/chroma_store", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("company_docs", env="CHROMA_COLLECTION_NAME")

    # Router
    router_confidence_threshold: float = Field(0.75, env="ROUTER_CONFIDENCE_THRESHOLD")
    router_model: str = Field("gpt-4o-mini", env="ROUTER_MODEL")

    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_debug: bool = Field(True, env="API_DEBUG")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — call this everywhere."""
    return Settings()
