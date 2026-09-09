from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DentalAI Copilot"
    database_url: str = "postgresql+psycopg://dentalai:dentalai@localhost:5438/dentalai"
    dox_mysql_url: str = ""
    dox_import_patient_limit: int = 10
    dox_import_batch_size: int = 200
    upload_dir: str = "uploads"
    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4.1-mini"
    google_api_key: str = ""
    google_model: str = "gemini-3.8-flash"
    embedding_dim: int = 384
    mock_llm: bool = True
    mock_vision: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
