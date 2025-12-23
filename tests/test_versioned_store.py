from pathlib import Path
from unittest.mock import MagicMock
from langchain_core.documents import Document

from vectorstore.versioned_store import VersionedVectorStore
from ingestion.version_manager import VersionManager


def make_chunk(text, chunk_id, source="doc.pdf", document_id="doc"):
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "source": source,
            "document_id": document_id,
        },
    )


def make_store():
    store = MagicMock()
    store.add = MagicMock()
    store.vector_store.delete = MagicMock()
    return store


def test_first_time_ingest(tmp_path):
    store = make_store()
    versions = VersionManager(tmp_path / "versions.json")
    vs = VersionedVectorStore(store, versions)

    chunks = [
        make_chunk("hello", "1", document_id="doc"),
        make_chunk("world", "2", document_id="doc"),
    ]

    result = vs.ingest(
        document_id="doc",
        source="doc_v1.pdf",
        chunks=chunks,
    )

    store.add.assert_called_once()
    assert result["added"] == 2



def test_rename_same_content_is_skipped(tmp_path):
    store = make_store()
    versions = VersionManager(tmp_path / "versions.json")
    vvs = VersionedVectorStore(store, versions)

    chunks = [
        make_chunk("hello", "c1", document_id="doc"),
        make_chunk("world", "c2", document_id="doc"),
    ]

    vvs.ingest(
        document_id="doc",
        source="doc_v1.pdf",
        chunks=chunks,
    )

    store.add.reset_mock()

    result = vvs.ingest(
        document_id="doc",
        source="doc_v2.pdf",
        chunks=chunks,
    )

    assert result["skipped"] is True
    store.add.assert_not_called()



def test_skip_unchanged_document(tmp_path):
    store = make_store()
    versions = VersionManager(tmp_path / "versions.json")
    vs = VersionedVectorStore(store, versions)

    chunks = [make_chunk("hello", "1", document_id="doc")]

    vs.ingest("doc", "doc_v1.pdf", chunks)
    result = vs.ingest("doc", "doc_v1.pdf", chunks)

    store.add.assert_called_once()
    assert result["skipped"] is True



def test_partial_update(tmp_path):
    store = make_store()
    versions = VersionManager(tmp_path / "versions.json")
    vs = VersionedVectorStore(store, versions)

    old_chunks = [
        make_chunk("hello", "1", document_id="doc"),
        make_chunk("world", "2", document_id="doc"),
    ]

    vs.ingest("doc", "doc_v1.pdf", old_chunks)

    new_chunks = [
        make_chunk("hello updated", "1", document_id="doc"),  # modified
        make_chunk("world", "2", document_id="doc"),          # unchanged
        make_chunk("new", "3", document_id="doc"),            # added
    ]

    result = vs.ingest("doc", "doc_v1.pdf", new_chunks)

    store.vector_store.delete.assert_called_once()
    assert result["added"] == 2



def test_partial_chunk_content_change(tmp_path):
    store = make_store()
    versions = VersionManager(tmp_path / "versions.json")
    vvs = VersionedVectorStore(store, versions)

    original_chunks = [
        make_chunk("chunk A", "c1", document_id="doc"),
        make_chunk("chunk B", "c2", document_id="doc"),
        make_chunk("chunk C", "c3", document_id="doc"),
    ]

    vvs.ingest("doc", "doc_v1.pdf", original_chunks)

    store.add.reset_mock()
    store.vector_store.delete.reset_mock()

    modified_chunks = [
        make_chunk("chunk A", "c1", document_id="doc"),
        make_chunk("chunk B UPDATED", "c2", document_id="doc"),
        make_chunk("chunk C", "c3", document_id="doc"),
    ]

    result = vvs.ingest("doc", "doc_v2.pdf", modified_chunks)

    assert result["added"] == 1
    assert result["deleted"] == 1

    store.vector_store.delete.assert_called_once()
    store.add.assert_called_once()

    deleted_ids = store.vector_store.delete.call_args.kwargs["ids"]
    added_chunks = store.add.call_args.args[0]

    assert deleted_ids == ["c2"]
    assert added_chunks[0].metadata["chunk_id"] == "c2"
