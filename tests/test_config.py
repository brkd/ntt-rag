import pytest
from pydantic import ValidationError
from config.config import BaseConfig


def test_settings_requires_inference_url(monkeypatch):
    monkeypatch.delenv("NTT_RAG_INFERENCE_SERVER_URL", raising=False)
    monkeypatch.setenv("NTT_RAG_PDF_LOCATION", "data/dir")

    with pytest.raises(ValidationError):
        BaseConfig()


def test_settings_requires_pdf_location(monkeypatch):
    monkeypatch.delenv("NTT_RAG_PDF_LOCATION", raising=False)
    monkeypatch.setenv("NTT_RAG_INFERENCE_SERVER_URL", "http://localhost:8001")

    with pytest.raises(ValidationError):
        BaseConfig()


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("NTT_RAG_LLM_MODEL", raising=False)
    monkeypatch.delenv("NTT_RAG_LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("NTT_RAG_LLM_TEMPERATURE", raising=False)

    monkeypatch.delenv("NTT_RAG_CHROMA_HOST", raising=False)
    monkeypatch.delenv("NTT_RAG_CHROMA_PORT", raising=False)
    monkeypatch.delenv("NTT_RAG_CHROMA_COLLECTION", raising=False)
    monkeypatch.delenv("NTT_RAG_EMBEDDING_MODEL", raising=False)

    monkeypatch.delenv("NTT_RAG_API_HOST", raising=False)
    monkeypatch.delenv("NTT_RAG_API_PORT", raising=False)

    monkeypatch.delenv("NTT_RAG_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("NTT_RAG_CHUNK_OVERLAP", raising=False)

    monkeypatch.delenv("NTT_RAG_DATA_VERSION_FILE", raising=False)


    monkeypatch.setenv("NTT_RAG_INFERENCE_SERVER_URL", "http://localhost:8001")
    monkeypatch.setenv("NTT_RAG_PDF_LOCATION", "data/dir")

    base_config = BaseConfig()

    assert base_config.LLM_MODEL == "llama3.2:3b"
    assert base_config.LLM_MAX_TOKENS == 512
    assert base_config.LLM_TEMPERATURE == 0.0

    assert base_config.CHROMA_HOST == "localhost"
    assert base_config.CHROMA_PORT == 8000
    assert base_config.CHROMA_COLLECTION == "ntt-rag"
    assert base_config.EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"

    assert base_config.API_HOST == "0.0.0.0"
    assert base_config.API_PORT == 9632

    assert base_config.CHUNK_SIZE == 880
    assert base_config.CHUNK_OVERLAP == 100

    assert base_config.DATA_VERSION_FILE == ".document_versions.json"


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("NTT_RAG_INFERENCE_SERVER_URL", "http://localhost:8001")
    monkeypatch.setenv("NTT_RAG_PDF_LOCATION", "data/dir")

    monkeypatch.setenv("NTT_RAG_LLM_MAX_TOKENS", "1024")

    settings = BaseConfig()

    assert settings.LLM_MAX_TOKENS == 1024