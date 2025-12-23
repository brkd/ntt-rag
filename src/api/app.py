import re

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from api.app_services import get_ingestion_components, get_config, get_versioned_store, get_vectorstore, get_version_manager


def derive_document_id(source: str) -> str:
    """
    sr_2015_yyyymmdd_v01.pdf  -> sr_2015
    sr_2019_20200101_v03.pdf -> sr_2019
    """
    stem = Path(source).stem

    # Remove _YYYYMMDD_vNN
    stem = re.sub(r"_\d{8}_v\d+$", "", stem)

    return stem.lower()

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
        document_id = derive_document_id(source)
        result = versioned_store.ingest(
            document_id=document_id,
            source=source,
            chunks=source_chunks,
        )
        
        print(
            f"Ingestion result | doc_id={document_id} | source={source}: {result}"
        )


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