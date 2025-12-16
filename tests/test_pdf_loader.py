import pytest
from unittest.mock import patch

from ingestion.loader import PDFLoader
from langchain_core.documents import Document

def test_pdf_loader_loads_documents():
    fake_docs = [
        Document(page_content="hello", metadata={"source": "a.pdf", "page": 1})
    ]

    with patch("ingestion.loader.DirectoryLoader") as MockDirectoryLoader:
        instance = MockDirectoryLoader.return_value
        instance.load.return_value = fake_docs

        loader = PDFLoader("/fake/path")

        docs = loader.load()

        assert docs == fake_docs
        MockDirectoryLoader.assert_called_once_with(
            path="/fake/path",
            glob="**/*.pdf",
            loader_cls=pytest.importorskip(
                "langchain_community.document_loaders"
            ).PyMuPDFLoader
        )


def test_loader_is_initialized_once():
    with patch("ingestion.loader.DirectoryLoader") as MockDirectoryLoader:
        instance = MockDirectoryLoader.return_value
        instance.load.return_value = []

        loader = PDFLoader("/fake/path")

        loader.load()
        loader.load()

        # DirectoryLoader should only be instantiated once
        assert MockDirectoryLoader.call_count == 1
