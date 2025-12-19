from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    LLM_MODEL: str = "llama3.2:3b"
    INFERENCE_SERVER_URL: str
    LLM_MAX_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.0

    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "ntt-rag"
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"

    PDF_LOCATION: str

    model_config = SettingsConfigDict(
        env_prefix="NTT_RAG_",
        env_file=".env"
    )

def get_config() -> BaseConfig:
    return BaseConfig()