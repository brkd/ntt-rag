from ingestion.loader import PDFLoader
from ingestion.cleaner import Cleaner
from ingestion.chunker import Chunker

from rag.pipeline import RAGPipeline
from vectorstore.vectorstore import VectorStoreBuilder
from rag.llm import LLMInterface

from config.config import AppConfig

from fastapi import Depends

def get_config() -> AppConfig:
    return AppConfig()


def get_vectorstore(config: AppConfig = Depends(get_config)) -> VectorStoreBuilder:
    return VectorStoreBuilder(
        collection_name=config.CHROMA_COLLECTION,
        host=config.CHROMA_HOST,
        port=config.CHROMA_PORT,
        embedding_model=config.EMBEDDING_MODEL,
    )


def get_llm(config: AppConfig = Depends(get_config)) -> LLMInterface:
    return LLMInterface(
        model=config.LLM_MODEL,
        inference_server_url=config.INFERENCE_SERVER_URL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )


def get_ingestion_components(config: AppConfig = Depends(get_config)):
    return (
        PDFLoader(config.PDF_LOCATION),
        Cleaner(),
        Chunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        ),
    )

async def get_rag_pipeline(vectorstore: VectorStoreBuilder = Depends(get_vectorstore), llm: LLMInterface = Depends(get_llm)):
    return RAGPipeline(vectorstore=vectorstore, llm=llm)
