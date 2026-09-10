from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DentalAI Copilot"
    database_url: str = "postgresql+psycopg://dentalai:dentalai@localhost:5438/dentalai"
    dox_mysql_url: str = ""
    # Set this when the API runs on the host while DOX MySQL is local. It
    # deliberately overrides only the host portion of DOX_MYSQL_URL.
    dox_mysql_host: str = ""
    dox_import_patient_limit: int = 10
    dox_import_batch_size: int = 200
    upload_dir: str = "uploads"
    llm_provider: str = "openrouter"
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openrouter/free"
    openrouter_api_key: str = ""
    google_api_key: str = ""
    google_model: str = "gemini-3.8-flash"
    embedding_dim: int = 384
    mock_llm: bool = True

    # Load the backend-local file no matter which directory starts Uvicorn.
    model_config = SettingsConfigDict(
        env_file=(".env", Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
