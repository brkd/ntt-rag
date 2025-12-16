from ingestion.chunker import Chunker
from langchain_core.documents import Document


def test_chunker_splits_long_document():
    chunker = Chunker(chunk_size=50, chunk_overlap=10)

    long_text = "This is a sentence. " * 20  # long enough to chunk

    doc = Document(
        page_content=long_text,
        metadata={"source": "test.pdf", "page": 1},
    )

    chunks = chunker.chunk([doc])

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.page_content) <= 50
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1


def test_chunker_respects_overlap():
    chunker = Chunker(chunk_size=40, chunk_overlap=20)

    text = "abcdefghijklmnopqrstuvwxyz" * 5
    doc = Document(page_content=text, metadata={})

    chunks = chunker.chunk([doc])

    assert len(chunks) >= 2

    # Ensure overlap exists
    overlap = set(chunks[0].page_content) & set(chunks[1].page_content)
    assert overlap


def test_chunker_handles_short_documents():
    chunker = Chunker(chunk_size=200)

    doc = Document(
        page_content="Short document.",
        metadata={"source": "short.pdf"},
    )

    chunks = chunker.chunk([doc])

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short document."
    assert chunks[0].metadata["source"] == "short.pdf"


def test_chunker_handles_empty_input():
    chunker = Chunker()

    chunks = chunker.chunk([])

    assert chunks == []
