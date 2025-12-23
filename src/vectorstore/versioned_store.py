from typing import List
from langchain_core.documents import Document

from ingestion.version_manager import (
    VersionManager,
    hash_chunk_content,
    hash_document_chunks,
    diff_chunks,
)
from vectorstore.vectorstore import VectorStoreBuilder


class VersionedVectorStore:
    def __init__(self, store: VectorStoreBuilder, versions: VersionManager):
        self.store = store
        self.versions = versions

    def ingest(self, document_id: str, source: str, chunks: List[Document]):
        chunk_hashes = {
            c.metadata["chunk_id"]: hash_chunk_content(c.page_content)
            for c in chunks
        }
        document_hash = hash_document_chunks(chunks)

        doc = self.versions.get_document(document_id)

        if doc and doc["current_hash"] == document_hash:
            self.versions.add_source(document_id, document_hash, source)
            self.versions.save()
            return {"skipped": True, "reason": "content unchanged"}


        if not doc:
            self.store.add(chunks)
            self.versions.register_version(
                document_id,
                document_hash,
                source,
                chunk_hashes,
            )
            self.versions.save()
            return {"added": len(chunks)}
        
        old_hash = doc["current_hash"]
        old_version = doc["versions"][old_hash]

        diff = diff_chunks(old_version["chunk_hashes"], chunk_hashes)

        if diff["delete"]:
            self.store.vector_store.delete(ids=list(diff["delete"]))


        chunks_to_add = [
            c for c in chunks
            if c.metadata["chunk_id"] in diff["add"]
        ]

        if chunks_to_add:
            self.store.add(chunks_to_add)


        self.versions.register_version(
            document_id,
            document_hash,
            source,
            chunk_hashes,
        )
        self.versions.save()

        return {
            "added": len(chunks_to_add),
            "deleted": len(diff["delete"]),
        }



        # existing = self.versions.get_by_hash(document_hash)

        # if existing:
        #     self.versions.add_source(document_hash, source)
        #     self.versions.save()
        #     return {"skipped": True, "reason": "content unchanged"}

        old = self.versions.get_by_source(source)

        # First-time content
        if not old:
            self.store.add(chunks)
            self.versions.register(document_hash, source, chunk_hashes)
            self.versions.save()
            return {"added": len(chunks)}

        diff = diff_chunks(old["chunk_hashes"], chunk_hashes)

        chunks_to_add = [
            c for c in chunks
            if c.metadata["chunk_id"] in diff["add"]
        ]

        if diff["delete"]:
            self.store.vector_store.delete(ids=list(diff["delete"]))

        if chunks_to_add:
            self.store.add(chunks_to_add)

        self.versions.register(document_hash, source, chunk_hashes)
        self.versions.save()

        return {
            "added": len(chunks_to_add),
            "deleted": len(diff["delete"]),
        }

