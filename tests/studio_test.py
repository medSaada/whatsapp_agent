"""
Simple test script for LangGraph Studio to debug the existing GraphBuilder
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import existing modules
from app.services.rag.graph.builder import GraphBuilder
from app.services.rag.generation_service import GenerationService
from app.services.rag.vector_store_service import VectorStoreService
from app.services.rag.graph.tools import create_rag_tool
from langchain_openai import ChatOpenAI


def create_test_graph():
    """Create test graph using existing modules"""
    
    # Use a simple mock LLM to avoid API calls
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        api_key="test-key"  # Won't be used since we'll mock responses
    )
    
    # Create generation service
    generation_service = GenerationService(llm)
    
    # Create vector store service (will use mock data)
    vector_store_service = VectorStoreService(
        vector_store_path="data/vector_store",
        embedding_model="text-embedding-3-small"
    )
    
    # Create tools
    tools = [create_rag_tool(vector_store_service, "test_collection")]
    
    # Create and build graph
    graph_builder = GraphBuilder(
        generation_service=generation_service,
        tools=tools,
        memory_threshold=3
    )
    
    # Build the graph
    db_path = "data/sqlite/studio_test.db"
    return graph_builder.build(db_path)


# For LangGraph Studio
workflow = create_test_graph()


if __name__ == "__main__":
    print("Creating test graph...")
    graph = create_test_graph()
    print("Graph created successfully!") 