from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DentalAI Copilot"
    # 本地 HTTP 使用 False；HTTPS 部署设为 True，并配置明确的前端 Origin。
    cookie_secure: bool = False
    allowed_origins: list[str] = ["http://localhost:4200", "http://127.0.0.1:4200"]
    database_url: str = "postgresql+psycopg://dentalai:dentalai@localhost:5438/dentalai"
    dox_mysql_url: str = ""
    # Set this when the API runs on the host while DOX MySQL is local. It
    # deliberately overrides only the host portion of DOX_MYSQL_URL.
    dox_mysql_host: str = ""
    dox_import_patient_limit: int = 10
    dox_import_batch_size: int = 200
    upload_dir: str = "uploads"
    # 聊天模型配置只影响回答/工具调用；检索向量仍由本地 EmbeddingService 生成。
    llm_provider: str = "openrouter"
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openrouter/free"
    openrouter_api_key: str = ""
    google_api_key: str = ""
    google_model: str = "gemini-3.8-flash"
    # 向量长度同时用于编码器和数据库列定义，不是聊天模型的上下文窗口大小。
    embedding_dim: int = 384
    # True 时禁用真实聊天模型；检索和数据库操作仍可运行，不需要模型密钥。
    mock_llm: bool = True

    # Load the backend-local file no matter which directory starts Uvicorn.
    model_config = SettingsConfigDict(
        env_file=(".env", Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    # 缓存配置，避免每次请求重复读取；修改环境文件后通常需要重启进程。
    return Settings()
