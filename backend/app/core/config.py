from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
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
    # 聊天模型和 Embedding 独立配置；MOCK_LLM 不控制向量模型。
    llm_provider: str = "openrouter"
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openrouter/free"
    openrouter_api_key: str = ""
    google_api_key: str = ""
    google_model: str = "gemini-3.8-flash"
    # 向量长度同时用于编码器和数据库列定义，不是聊天模型的上下文窗口大小。
    embedding_dim: int = Field(default=1024, ge=1, le=16000)
    embedding_provider: Literal["openrouter", "hash"] = "openrouter"
    embedding_model: str = "liquid/lfm-2.5-embedding-350m:free"
    embedding_base_url: str = "https://openrouter.ai/api/v1"
    embedding_api_key: str = ""
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    embedding_timeout: float = Field(default=30, gt=0, le=300)
    embedding_max_retries: int = Field(default=2, ge=0, le=5)
    # UTF-8 字节预算是保守的切片长度代理，不是精确的模型 token 数。
    # 预留模型特殊 token 空间，避免使用 OpenAI tokenizer 估算 Liquid 输入。
    rag_chunk_size: int = Field(default=480, ge=4, le=480)
    rag_chunk_overlap: int = Field(default=72, ge=0)
    # True 只禁用真实聊天模型；完全离线还需设置 EMBEDDING_PROVIDER=hash。
    mock_llm: bool = True

    @model_validator(mode="after")
    def validate_embedding_settings(self):
        """启动时拒绝不可能工作的维度或切片窗口，避免直到写库才报错。"""
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if (self.embedding_provider == "openrouter"
                and self.embedding_model == "liquid/lfm-2.5-embedding-350m:free"
                and self.embedding_dim != 1024):
            raise ValueError("LFM2.5 embeddings require EMBEDDING_DIM=1024; rebuild the old index")
        return self

    # Load the backend-local file no matter which directory starts Uvicorn.
    model_config = SettingsConfigDict(
        env_file=(".env", Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    # 缓存配置，避免每次请求重复读取；修改环境文件后通常需要重启进程。
    return Settings()
