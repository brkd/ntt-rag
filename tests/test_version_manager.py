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

    document_id = "sr_2015"
    document_hash = "abc123"
    source = "sr_2015_20150301_v01.pdf"
    chunks = {"c1": "h1", "c2": "h2"}

    vm.register_version(document_id, document_hash, source, chunks)
    vm.save()

    loaded = VersionManager(tmp_path / "versions.json")

    version = loaded.get_version(document_id, document_hash)

    assert version is not None
    assert version["sources"] == [source]


def test_add_source_same_content(tmp_path: Path):
    vm = VersionManager(tmp_path / "versions.json")

    document_id = "sr_2015"
    document_hash = "hash1"

    vm.register_version(
        document_id=document_id,
        document_hash=document_hash,
        source="a.pdf",
        chunk_hashes={"c1": "h1"},
    )

    vm.add_source(
        document_id=document_id,
        document_hash=document_hash,
        source="b.pdf",
    )

    version = vm.get_version(document_id, document_hash)

    assert version is not None
    assert set(version["sources"]) == {"a.pdf", "b.pdf"}
    assert len(version["sources"]) == 2
