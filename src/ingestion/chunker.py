import hashlib

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )


    def chunk(self, documents: List[Document]) -> List[Document]:
        chunks = self.chunker.split_documents(documents)

        chunk_counters = {}

        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", -1)

            key = (source, page)
            chunk_index = chunk_counters.get(key, 0)
            chunk_counters[key] = chunk_index + 1

            content_fingerprint = chunk.page_content.strip()

            raw_id = f"{page}::{chunk_index}::{content_fingerprint}"
            chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

            chunk.metadata.update({
                "file_name": source.split("/")[-1],
                "page": page,
                "chunk_index": chunk_index,
                "chunk_id": chunk_id
            })
            
        return chunks