import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from api.app import app
from api.app_services import get_rag_pipeline

@pytest.fixture
def mock_rag_pipeline(monkeypatch):
    mock = AsyncMock()
    mock.ask.return_value = {
        "answer": "mocked answer",
        "sources": [{"source": "data/doc1.pdf", "file_name": "doc1.pdf", "page": 0}]
    }
    return mock

@pytest.fixture
def client(mock_rag_pipeline):
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag_pipeline

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()