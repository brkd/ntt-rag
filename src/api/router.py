from fastapi import APIRouter, Depends

from rag.pipeline import RAGPipeline

from schemas import HealthCheck
from app_services import get_rag_pipeline

api_router = APIRouter()

@api_router.get('/health')
async def check_health_status() -> HealthCheck:
    return HealthCheck(status="OK")