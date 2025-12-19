import pytest
from unittest.mock import AsyncMock, MagicMock

from rag.pipeline import RAGPipeline
from langchain_core.documents import Document

from langchain_core.messages import SystemMessage, HumanMessage


def test_create_rag_messages_structure():
    pipeline = RAGPipeline(
        vectorstore=MagicMock(),
        llm=MagicMock()
    )

    context = "Some retrieved context"
    question = "Some question"

    messages = pipeline.create_rag_messages(context=context, question=question)

    assert len(messages) == 2

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    assert context in messages[1].content
    assert question in messages[1].content


@pytest.mark.asyncio
async def test_rag_pipeline_returns_answers_and_sources():
    fake_results = [
        (
            Document(
                page_content="NTT Data promotes sustainability.",
                metadata={"source": "doc1.pdf", "file_name": "doc1.pdf", "page": 0},
            ),
            0.01,
        )
    ]

    mock_vectorstore = MagicMock()
    mock_vectorstore.search = AsyncMock(return_value=fake_results)

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="NTT Data promotes sustainability.")

    pipeline = RAGPipeline(vectorstore=mock_vectorstore, llm=mock_llm)

    result = await pipeline.ask("What does NTT Data promote?")

    assert "NTT Data promotes sustainability." in result["answer"]
    assert result["sources"] == [{"source": "doc1.pdf", "file_name": "doc1.pdf", "page": 0}]