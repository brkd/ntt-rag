from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStoreBuilder:
    def __init__(self, collection_name: str, host: str, port: int, embedding_model: str):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.embedding_model = HuggingFaceEmbeddings(model=embedding_model, model_kwargs={"device": "cpu"}, encode_kwargs={"batch_size": 64})
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            host=host,
            port=port
        )

    def add(self, documents: List[Document]):
        ids = [doc.metadata["chunk_id"] for doc in documents]
        self.vector_store.add_documents(documents=documents, ids=ids)

    async def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        search_result = await self.vector_store.asimilarity_search_with_score(query, k=k)
        return search_result