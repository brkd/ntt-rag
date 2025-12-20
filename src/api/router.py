from fastapi import APIRouter, Depends, HTTPException

from rag.pipeline import RAGPipeline

from api.schemas import HealthCheck, LLMQuestion, LLMAnswer
from api.app_services import get_rag_pipeline

api_router = APIRouter()

@api_router.get('/health', response_model=HealthCheck)
def check_health_status() -> HealthCheck:
    return HealthCheck(status="OK")


@api_router.post('/ask', response_model=LLMAnswer)
async def ask_question(payload: LLMQuestion, rag: RAGPipeline = Depends(get_rag_pipeline)) -> LLMAnswer:
    try:
        return await rag.ask(payload.question)
    except Exception as e:
        raise HTTPException(status_code=500)