from langchain_core.tools import BaseTool, Tool
from pydantic.v1 import BaseModel, Field

from app.services.rag.vector_store_service import VectorStoreService

class RetrieverInput(BaseModel):
    """Input schema for the retriever tool."""
    query: str = Field(description="The query to search for in the knowledge base.")

def create_rag_tool(vector_store_service: VectorStoreService, collection_name: str) -> BaseTool:
    """
    Creates a retriever tool for the RAG agent.

    This tool allows the agent to search the vector store for relevant documents
    based on the user's query. The retriever is configured to return the top 5
    results from a specific document source.
    """
    retriever = vector_store_service.as_retriever(
        collection_name=collection_name,
        search_kwargs={"k": 5}
    )
    
    return Tool(
        name="knowledge_base_retriever",
        description="This is your primary tool. You **MUST** use it to search the Geniats knowledge base for specific details about the e-learning programs, pricing, and curriculum to answer client questions. Use this to find the exact information you need before responding.",
        func=retriever.invoke,
        args_schema=RetrieverInput,
    ) 