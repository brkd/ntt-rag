import asyncio

from ingestion.loader import PDFLoader
from ingestion.cleaner import Cleaner
from ingestion.chunker import Chunker

from vectorstore.vectorstore import VectorStoreBuilder
from rag.llm import LLMInterface
from rag.pipeline import RAGPipeline

from config.config import BaseConfig

async def main():
    base_config = BaseConfig()

    loader = PDFLoader(base_config.PDF_LOCATION)
    documents = loader.load()

    if not documents:
        print("No documents loaded")
        return
    
    print(f"Loaded {len(documents)} documents")

    cleaner = Cleaner()
    cleaned_documents = cleaner.clean(documents)
    print(f"Cleaned {len(cleaned_documents)} documents")

    chunker = Chunker(chunk_size=base_config.CHUNK_SIZE, chunk_overlap=base_config.CHUNK_OVERLAP)
    chunks = chunker.chunk(cleaned_documents)
    print(f"Created {len(chunks)} chunks")

    vectorstore = VectorStoreBuilder(
        collection_name=base_config.CHROMA_COLLECTION,
        host=base_config.CHROMA_HOST,
        port=base_config.CHROMA_PORT,
        embedding_model=base_config.EMBEDDING_MODEL,
    )

    vectorstore.add(chunks)
    print("Documents added to vector store")

    llm = LLMInterface(
        model=base_config.LLM_MODEL,
        inference_server_url=base_config.INFERENCE_SERVER_URL,
        temperature=base_config.LLM_TEMPERATURE,
        max_tokens=base_config.LLM_MAX_TOKENS
    )

    rag = RAGPipeline(vectorstore=vectorstore, llm=llm)

    question = "What did Berkay Demireller do for the CULTURATI project?"
    print(f"\nQuestion: {question}")

    result = await rag.ask(question, k=3)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for src in result["sources"]:
        print("-", src)


if __name__ == "__main__":
    asyncio.run(main())