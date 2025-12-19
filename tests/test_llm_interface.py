import pytest
from unittest.mock import AsyncMock, MagicMock

from rag.llm import LLMInterface


@pytest.mark.asyncio
async def test_llm_interface_generate_returns_text():
    mock_llm = MagicMock()
    mock_llm.agenerate = AsyncMock(
        return_value=MagicMock(
            generations=[[MagicMock(text="hello world")]]
        )
    )

    llm_interface = LLMInterface.__new__(LLMInterface)
    llm_interface.llm = mock_llm

    result = await llm_interface.generate("test message")

    mock_llm.agenerate.assert_called_once()
    assert result == "hello world"