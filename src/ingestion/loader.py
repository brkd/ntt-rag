from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_core.documents import Document

from typing import List


class PDFLoader:
    def __init__(self, input_directory: str):
        self.input_directory = input_directory
        self.loader = None

    def load(self) -> List[Document]:
        if self.loader == None:
            self.loader = DirectoryLoader(
                path = self.input_directory,
                glob = "**/*.pdf",
                loader_cls = PyMuPDFLoader
            )

        return self.loader.load()


