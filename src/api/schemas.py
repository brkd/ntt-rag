from pydantic import BaseModel
from typing import List

class HealthCheck(BaseModel):

    status: str = "OK"


class Source(BaseModel):
    source: str
    file_name: str
    page: int


class LLMAnswer(BaseModel):
    answer: str
    sources: List[Source]


class LLMQuestion(BaseModel):
    question: str