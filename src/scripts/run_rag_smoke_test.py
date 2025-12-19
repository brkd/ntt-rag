import asyncio

from ingestion.loader import PDFLoader
from ingestion.cleaner import Cleaner
from ingestion.chunker import Chunker

from vectorstore.vectorstore import VectorStoreBuilder
from rag.llm import LLMInterface
from rag.pipeline import RAGPipeline

async def main():

    loader = PDFLoader("/home/brkd/Desktop/repos/ntt-rag/data/raw")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    if not documents:
        print("No documents loaded")
        return
    
    cleaner = Cleaner()
    cleaned_documents = cleaner.clean(documents)
    print(f"Cleaned {len(cleaned_documents)} documents")

    chunker = Chunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk(cleaned_documents)
    print(f"Created {len(chunks)} chunks")

    vectorstore = VectorStoreBuilder(
        collection_name="demo_collection",
        host="localhost",
        port=8000,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
    )

    vectorstore.add(chunks)
    print("Documents added to vector store")

    llm = LLMInterface(
        model="llama3.2:3b",
        inference_server_url="http://localhost:11434/v1",
        temperature=0.0,
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