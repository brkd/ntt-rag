from contextlib import asynccontextmanager

from fastapi import FastAPI

from ingestion.loader import PDFLoader
from ingestion.cleaner import Cleaner
from ingestion.chunker import Chunker

from vectorstore.vectorstore import VectorStoreBuilder
from rag.llm import LLMInterface
from rag.pipeline import RAGPipeline

from config.config import AppConfig

app_config = AppConfig()

loader = PDFLoader(app_config.PDF_LOCATION)
cleaner = Cleaner()
chunker = Chunker(chunk_size=app_config.CHUNK_SIZE, chunk_overlap=app_config.CHUNK_OVERLAP)

vectorstore = VectorStoreBuilder(
    collection_name=app_config.CHROMA_COLLECTION,
    host=app_config.CHROMA_HOST,
    port=app_config.CHROMA_PORT,
    embedding_model=app_config.EMBEDDING_MODEL,
)

llm = LLMInterface(
    model=app_config.LLM_MODEL,
    inference_server_url=app_config.INFERENCE_SERVER_URL,
    temperature=app_config.LLM_TEMPERATURE,
    max_tokens=app_config.LLM_MAX_TOKENS
)


# Async context manager for startup operations
@asynccontextmanager
async def lifespan(app: FastAPI):
    documents = loader.load()
    cleaned_documents = cleaner.clean(documents)
    chunks = chunker.chunk(cleaned_documents)

    vectorstore.add(chunks)

    yield


from api.router import api_router

app = FastAPI()
app.include_router(api_router)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run("app:app", host=app_config.API_HOST, port=app_config.API_PORT, reload=False)