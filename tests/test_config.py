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
    monkeypatch.setenv("NTT_RAG_INFERENCE_SERVER_URL", "http://localhost:8001")
    monkeypatch.setenv("NTT_RAG_PDF_LOCATION", "data/dir")


    base_config = BaseConfig()

    assert base_config.LLM_MODEL == "llama3.2:3b"
    assert base_config.LLM_MAX_TOKENS == 512
    assert base_config.LLM_TEMPERATURE == 0.0


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("NTT_RAG_INFERENCE_SERVER_URL", "http://localhost:8001")
    monkeypatch.setenv("NTT_RAG_PDF_LOCATION", "data/dir")

    monkeypatch.setenv("NTT_RAG_LLM_MAX_TOKENS", "1024")

    settings = BaseConfig()

    assert settings.LLM_MAX_TOKENS == 1024