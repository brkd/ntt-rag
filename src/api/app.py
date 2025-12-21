from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.app_services import get_ingestion_components, get_config, get_vectorstore


# Async context manager for startup operations
@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()

    loader, cleaner, chunker = get_ingestion_components(config)
    vectorstore = get_vectorstore(config)

    documents = loader.load()
    cleaned_documents = cleaner.clean(documents)
    chunks = chunker.chunk(cleaned_documents)

    vectorstore.add(chunks)

    yield


from api.router import api_router

def create_app(enable_ingestion: bool = True) -> FastAPI:
    app = FastAPI(lifespan=lifespan if enable_ingestion else None)
    app.include_router(api_router)
    return app


app = create_app()


if __name__ == '__main__':
    import uvicorn

    config = get_config()

    uvicorn.run("app:app", host=config.API_HOST, port=config.API_PORT, reload=False)