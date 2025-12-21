from pathlib import Path
from unittest.mock import MagicMock
from langchain_core.documents import Document

from vectorstore.versioned_store import VersionedVectorStore
from ingestion.version_manager import VersionManager


def make_chunk(text, chunk_id, source="doc.pdf"):
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "source": source,
        },
    )


def make_store():
    store = MagicMock()
    store.add = MagicMock()
    store.vector_store.delete = MagicMock()
    return store


def test_first_time_ingest(tmp_path):
    store = MagicMock()
    versions = VersionManager(tmp_path / "versions.json")

    vs = VersionedVectorStore(store, versions)

    chunks = [
        make_chunk("hello", "1"),
        make_chunk("world", "2"),
    ]

    result = vs.ingest("doc.pdf", chunks)

    store.add.assert_called_once()
    assert result["added"] == 2


def test_rename_same_content_is_skipped(tmp_path):
    store = make_store()
    versions = VersionManager(tmp_path / "versions.json")
    vvs = VersionedVectorStore(store, versions)

    chunks = [
        Document("hello", metadata={"chunk_id": "c1"}),
        Document("world", metadata={"chunk_id": "c2"}),
    ]

    vvs.ingest("doc1.pdf", chunks)
    store.add.reset_mock()

    result = vvs.ingest("doc2.pdf", chunks)

    assert result["skipped"] is True
    store.add.assert_not_called()


def test_skip_unchanged_document(tmp_path):
    store = MagicMock()
    versions = VersionManager(tmp_path / "versions.json")

    vs = VersionedVectorStore(store, versions)

    chunks = [make_chunk("hello", "1")]

    vs.ingest("doc.pdf", chunks)
    result = vs.ingest("doc.pdf", chunks)

    store.add.assert_called_once()
    assert result["skipped"] is True


def test_partial_update(tmp_path):
    store = MagicMock()
    store.vector_store.delete = MagicMock()

    versions = VersionManager(tmp_path / "versions.json")
    vs = VersionedVectorStore(store, versions)

    old_chunks = [
        make_chunk("hello", "1"),
        make_chunk("world", "2"),
    ]
    vs.ingest("doc.pdf", old_chunks)

    new_chunks = [
        make_chunk("hello updated", "1"),  # modified
        make_chunk("world", "2"),           # unchanged
        make_chunk("new", "3"),              # added
    ]

    result = vs.ingest("doc.pdf", new_chunks)

    store.vector_store.delete.assert_called_once()
    assert result["added"] == 2


def test_partial_chunk_content_change(tmp_path):
    store = make_store()
    versions = VersionManager(tmp_path / "versions.json")
    vvs = VersionedVectorStore(store, versions)

    original_chunks = [
        Document("chunk A", metadata={"chunk_id": "c1"}),
        Document("chunk B", metadata={"chunk_id": "c2"}),
        Document("chunk C", metadata={"chunk_id": "c3"}),
    ]

    vvs.ingest("doc.pdf", original_chunks)

    store.add.reset_mock()
    store.vector_store.delete.reset_mock()

    modified_chunks = [
        Document("chunk A", metadata={"chunk_id": "c1"}),     # unchanged
        Document("chunk B UPDATED", metadata={"chunk_id": "c2"}),  # modified
        Document("chunk C", metadata={"chunk_id": "c3"}),     # unchanged
    ]

    result = vvs.ingest("doc.pdf", modified_chunks)

    assert result["added"] == 1
    assert result["deleted"] == 1

    store.vector_store.delete.assert_called_once()
    store.add.assert_called_once()

    deleted_ids = store.vector_store.delete.call_args.kwargs["ids"]
    added_chunks = store.add.call_args.args[0]

    assert deleted_ids == ["c2"]
    assert added_chunks[0].metadata["chunk_id"] == "c2"