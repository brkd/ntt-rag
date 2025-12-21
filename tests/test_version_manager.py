from pathlib import Path
from langchain_core.documents import Document

from ingestion.version_manager import (
    VersionManager,
    hash_chunk_content,
    hash_document_chunks,
    diff_chunks,
)


def test_chunk_hash_changes_on_content_change():
    h1 = hash_chunk_content("hello")
    h2 = hash_chunk_content("hello world")

    assert h1 != h2


def test_document_hash_order_independent():
    c1 = Document("a", metadata={"chunk_id": "1"})
    c2 = Document("b", metadata={"chunk_id": "2"})

    h1 = hash_document_chunks([c1, c2])
    h2 = hash_document_chunks([c2, c1])

    assert h1 == h2


def test_diff_chunks_detects_changes():
    old = {"1": "a", "2": "b"}
    new = {"2": "b", "3": "c"}

    diff = diff_chunks(old, new)

    assert diff["add"] == {"3"}
    assert diff["delete"] == {"1"}
    assert diff["unchanged"] == {"2"}


def test_register_and_lookup(tmp_path: Path):
    vm = VersionManager(tmp_path / "versions.json")

    doc_hash = "abc123"
    chunks = {"c1": "h1", "c2": "h2"}

    vm.register(doc_hash, "file1.pdf", chunks)
    vm.save()

    loaded = VersionManager(tmp_path / "versions.json")

    assert loaded.get_by_hash(doc_hash)["chunk_count"] == 2
    assert loaded.get_by_source("file1.pdf")["chunk_count"] == 2


def test_add_source_same_content(tmp_path: Path):
    vm = VersionManager(tmp_path / "versions.json")

    vm.register("hash1", "a.pdf", {"c1": "h1"})
    vm.add_source("hash1", "b.pdf")

    assert vm.get_by_source("a.pdf")
    assert vm.get_by_source("b.pdf")

    assert len(vm.get_by_hash("hash1")["sources"]) == 2
