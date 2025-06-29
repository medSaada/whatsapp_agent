from langchain_core.tools import BaseTool, create_retriever_tool

from app.services.rag.vector_store_service import VectorStoreService

def create_rag_tool(vector_store_service: VectorStoreService, collection_name: str) -> BaseTool:
    """
    Creates a retriever tool for the RAG agent.

    This tool allows the agent to search the vector store for relevant documents
    based on the user's query. The retriever is configured to return the top 5
    results from a specific document source.
    """
    retriever = vector_store_service.as_retriever(
        collection_name=collection_name,
        search_kwargs={
            "k": 5,
            # This filter can be parameterized if needed
           #"filter": {"source": "H:\\projects\\pojects_codes\\Personal_Projects\\whatsapp_agent_poc\\data\\documents\\datagenerated_assistant.txt"
           }
        
    )
    
    retriever_tool = create_retriever_tool(
        retriever,
        "knowledge_base_retriever",
        "Searches and returns relevant information from the knowledge base to answer user questions.",
    )
    return retriever_tool 