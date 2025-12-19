import hashlib
from ingestion.chunker import Chunker
from langchain_core.documents import Document


def test_chunk_id_is_deterministic():
    doc = Document(
        page_content="Same content every time " * 50,
        metadata={"source": "file.pdf", "page": 0},
    )

    chunker = Chunker(chunk_size=100, chunk_overlap=0)

    chunks_run_1 = chunker.chunk([doc])
    chunks_run_2 = chunker.chunk([doc])

    ids_1 = [c.metadata["chunk_id"] for c in chunks_run_1]
    ids_2 = [c.metadata["chunk_id"] for c in chunks_run_2]

    assert ids_1 == ids_2


def test_chunk_index_resets_for_different_pages():
    docs = [
        Document("A " * 300, metadata={"source": "file.pdf", "page": 1}),
        Document("B " * 300, metadata={"source": "file.pdf", "page": 2}),
    ]

    chunker = Chunker(chunk_size=50, chunk_overlap=0)
    chunks = chunker.chunk(docs)

    page_1_indices = [
        c.metadata["chunk_index"]
        for c in chunks if c.metadata["page"] == 1
    ]
    page_2_indices = [
        c.metadata["chunk_index"]
        for c in chunks if c.metadata["page"] == 2
    ]

    assert page_1_indices[0] == 0
    assert page_2_indices[0] == 0



def test_chunker_adds_required_metadata():
    doc = Document(
        page_content="This is a test document. " * 100,
        metadata={"source": "/tmp/test.pdf", "page": 0},
    )

    chunker = Chunker(chunk_size=100, chunk_overlap=0)
    chunks = chunker.chunk([doc])

    assert len(chunks) > 0

    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "file_name" in chunk.metadata
        assert "page" in chunk.metadata


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
