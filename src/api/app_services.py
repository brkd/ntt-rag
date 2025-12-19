from app import (
    vectorstore,
    llm,
)

from rag.pipeline import RAGPipeline
from vectorstore.vectorstore import VectorStoreBuilder
from rag.llm import LLMInterface

from fastapi import Depends


async def get_vectorstore():
    return vectorstore

async def get_llm():
    return llm

async def get_rag_pipeline(vectorstore: VectorStoreBuilder = Depends(get_vectorstore), llm: LLMInterface = Depends(get_llm)):
    return RAGPipeline(vectorstore=vectorstore, llm=llm)
