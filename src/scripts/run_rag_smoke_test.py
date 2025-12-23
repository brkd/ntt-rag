import asyncio

from ingestion.loader import PDFLoader
from ingestion.cleaner import Cleaner
from ingestion.chunker import Chunker
from ingestion.version_manager import VersionManager

from vectorstore.versioned_store import VersionedVectorStore
from vectorstore.vectorstore import VectorStoreBuilder

from rag.llm import LLMInterface
from rag.pipeline import RAGPipeline

from api.app import derive_document_id

from pathlib import Path

from config.config import AppConfig

VERSION_FILE = Path(".document_versions.json")

async def main():
    app_config = AppConfig()

    loader = PDFLoader(app_config.PDF_LOCATION)
    documents = loader.load()

    if not documents:
        print("No documents loaded")
        return
    
    print(f"Loaded {len(documents)} documents")

    cleaner = Cleaner()
    cleaned_documents = cleaner.clean(documents)
    print(f"Cleaned {len(cleaned_documents)} documents")

    chunker = Chunker(chunk_size=app_config.CHUNK_SIZE, chunk_overlap=app_config.CHUNK_OVERLAP)
    chunks = chunker.chunk(cleaned_documents)
    print(f"Created {len(chunks)} chunks")

    vectorstore = VectorStoreBuilder(
        collection_name=app_config.CHROMA_COLLECTION,
        host=app_config.CHROMA_HOST,
        port=app_config.CHROMA_PORT,
        embedding_model=app_config.EMBEDDING_MODEL,
    )

    version_manager = VersionManager(VERSION_FILE)
    versioned_store = VersionedVectorStore(
        store=vectorstore,
        versions=version_manager,
    )

    collection = vectorstore.vector_store._collection

    count_before = collection.count()
    print(f"Chunks before ingestion: {count_before}")

    # Group chunks by source document
    from collections import defaultdict
    chunks_by_source = defaultdict(list)

    for chunk in chunks:
        chunks_by_source[chunk.metadata["source"]].append(chunk)

    for source, source_chunks in chunks_by_source.items():
        document_id = derive_document_id(source)
        result = versioned_store.ingest(
            document_id=document_id,
            source=source,
            chunks=source_chunks,
        )
        print(f"Ingestion result for {source}: {result}")

    # vectorstore.add(chunks)
    print("Documents added to vector store")

    count_after = collection.count()
    print(f"Chunks after ingestion: {count_after}")

    llm = LLMInterface(
        model=app_config.LLM_MODEL,
        inference_server_url=app_config.INFERENCE_SERVER_URL,
        temperature=app_config.LLM_TEMPERATURE,
        max_tokens=app_config.LLM_MAX_TOKENS
    )

    rag = RAGPipeline(vectorstore=vectorstore, llm=llm)

    question = "What information is in the documents related to 2014?"
    print(f"\nQuestion: {question}")

    result = await rag.ask(question, k=20)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for src in result["sources"]:
        print("-", src)


if __name__ == "__main__":
    asyncio.run(main())