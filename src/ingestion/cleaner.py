import re
import unicodedata
from typing import List

from langchain_core.documents import Document


class Cleaner:
    """
    Cleans PDF-extracted documents for RAG consumption.
    """

    def clean(self, documents: List[Document]) -> List[Document]:
        return [
            Document(
                page_content=self._clean_text(doc.page_content),
                metadata=dict(doc.metadata),
            )
            for doc in documents
        ]

    def _clean_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)

        # Remove Private Use Area glyphs
        text = "".join(
            " " if 0xE000 <= ord(ch) <= 0xF8FF else ch
            for ch in text
        )

        # Fix hyphenation across line breaks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Normalize line breaks
        text = text.replace("\r\n", "\n")

        # Collapse excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces
        text = re.sub(r"[ \t]{2,}", " ", text)

        return text.strip()
