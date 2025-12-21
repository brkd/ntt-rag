import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from langchain_core.documents import Document


def hash_chunk_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_document_chunks(chunks: List[Document]) -> str:
    h = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda c: c.metadata["chunk_id"]):
        h.update(chunk.page_content.encode("utf-8"))
    return h.hexdigest()


def diff_chunks(old: Dict[str, str], new: Dict[str, str]) -> Dict[str, set]:
    old_ids = set(old.keys())
    new_ids = set(new.keys())

    added = new_ids - old_ids
    removed = old_ids - new_ids
    common = old_ids & new_ids

    modified = {
        cid for cid in common
        if old[cid] != new[cid]
    }

    return {
        "add": added | modified,
        "delete": removed | modified,
        "unchanged": common - modified,
    }


class VersionManager:
    def __init__(self, version_file: Path):
        self.version_file = version_file
        self.state = self._load()
        self.source_to_doc = {
            src: doc_hash
            for doc_hash, data in self.state.items()
            for src in data.get("sources", [])
        }


    def _load(self) -> Dict:
        if self.version_file.exists():
            return json.loads(self.version_file.read_text())
        return {}

    def save(self):
        self.version_file.write_text(
            json.dumps(self.state, indent=2, sort_keys=True)
        )

    def get_by_hash(self, document_hash: str):
        return self.state.get(document_hash)

    def get_by_source(self, source: str):
        doc_hash = self.source_to_doc.get(source)
        if not doc_hash:
            return None
        return self.state.get(doc_hash)


    def register(
        self,
        document_hash: str,
        source: str,
        chunk_hashes: Dict[str, str],
    ):
        self.state[document_hash] = {
            "chunk_hashes": chunk_hashes,
            "chunk_count": len(chunk_hashes),
            "sources": [source],
            "last_indexed": datetime.now().isoformat(),
        }
        self.source_to_doc[source] = document_hash

    def add_source(self, document_hash: str, source: str):
        entry = self.state[document_hash]
        if source not in entry["sources"]:
            entry["sources"].append(source)
        self.source_to_doc[source] = document_hash
