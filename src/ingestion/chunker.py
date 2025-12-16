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
        
        return self.chunker.split_documents(documents)