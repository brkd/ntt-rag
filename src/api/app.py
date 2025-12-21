from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.app_services import get_ingestion_components, get_config, get_versioned_store, get_vectorstore, get_version_manager


# Async context manager for startup operations
@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()

    loader, cleaner, chunker = get_ingestion_components(config)

    vector_store = get_vectorstore(config)
    version_manager = get_version_manager(config)

    versioned_store= get_versioned_store(vector_store, version_manager)

    documents = loader.load()
    cleaned_documents = cleaner.clean(documents)
    chunks = chunker.chunk(cleaned_documents)

    from collections import defaultdict
    chunks_by_source = defaultdict(list)

    for chunk in chunks:
        chunks_by_source[chunk.metadata["source"]].append(chunk)

    for source, source_chunks in chunks_by_source.items():
        result = versioned_store.ingest(
            source=source,
            chunks=source_chunks,
        )
        print(f"Ingestion result for {source}: {result}")


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