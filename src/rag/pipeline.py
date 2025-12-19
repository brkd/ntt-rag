from typing import Dict, Any, List

from rag.llm import LLMInterface
from vectorstore.vectorstore import VectorStoreBuilder

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.base import BaseMessage


class RAGPipeline:
    def __init__(self, vectorstore: VectorStoreBuilder, llm: LLMInterface):
        self.vectorstore = vectorstore
        self.llm = llm

    
    def create_rag_messages(self, context: str, question: str) -> List[BaseMessage]:
        system_message = SystemMessage(content="""You are an assistant answering questions using ONLY the provided context.
                                       
        Instructions:
        - Base your answer strictly on the context above
        - If the answer is not present, say "I don't know based on the provided documents."
        - Be concise and factual""")

        human_message = HumanMessage(content=f"""Context:
        {context}      

        Question:
        {question}

        Answer:""")

        return [system_message, human_message]

    
    async def ask(self, question: str, k: int = 3) -> Dict[str, Any]:
        # Retrieve relevant documents
        results = await self.vectorstore.search(query=question, k=k)

        for result in results:
            print("----------------------")
            print(result[0].page_content)
            print("----------------------")


        documents = [doc for doc, _ in results]

        # Build context
        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in documents
        )

        messages = self.create_rag_messages(context, question)

        # Generate answer
        answer = await self.llm.generate(messages)

        # Collect sources
        sources = list(
            {doc.metadata.get("source", "unknown") for doc in documents}
        )

        return {
            "answer": answer.strip(),
            "sources": sources,
        }