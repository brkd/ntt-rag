from langchain_openai import ChatOpenAI


class LLMInterface:
    def __init__(self, model: str, inference_server_url: str, max_tokens: int = 512, temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key="EMPTY",
            base_url=inference_server_url,
            max_tokens=max_tokens,
            temperature=temperature
        )

    async def generate(self, messages: str) -> str:
        result = await self.llm.agenerate([messages])
        return result.generations[0][0].text