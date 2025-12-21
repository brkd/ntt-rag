from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"


class BaseConfig(BaseSettings):
    LLM_MODEL: str = "llama3.2:3b"
    INFERENCE_SERVER_URL: str = Field(..., alias="NTT_RAG_INFERENCE_SERVER_URL")
    LLM_MAX_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.0

    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "ntt-rag"
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"

    PDF_LOCATION: str = Field(..., alias="NTT_RAG_PDF_LOCATION")
    DATA_VERSION_FILE: str = ".document_versions.json"
    CHUNK_SIZE: int = 880
    CHUNK_OVERLAP: int = 100

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 9632

    model_config = SettingsConfigDict(
        env_prefix="NTT_RAG_",
        env_file=None
    )

class AppConfig(BaseConfig):
    model_config = SettingsConfigDict(
        env_prefix="NTT_RAG_",
        env_file=ENV_PATH
    )

def get_config() -> BaseConfig:
    return BaseConfig()