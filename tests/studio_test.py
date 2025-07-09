"""
Simple test script for LangGraph Studio to debug the existing GraphBuilder
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.rag.graph.builder import GraphBuilder
from app.services.rag.generation_service import GenerationService
from app.services.rag.vector_store_service import VectorStoreService
from app.services.rag.graph.tools import create_rag_tool
from langchain_openai import ChatOpenAI


def create_test_graph():
    """Create test graph using existing modules"""
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        api_key="test-key"
    )
    
    generation_service = GenerationService(llm)
    
    vector_store_service = VectorStoreService(
        vector_store_path="data/vector_store",
        embedding_model="text-embedding-3-small"
    )
    
    tools = [create_rag_tool(vector_store_service, "test_collection")]
    
    graph_builder = GraphBuilder(
        generation_service=generation_service,
        tools=tools,
        memory_threshold=3
    )
    
    db_path = "data/sqlite/studio_test.db"
    return graph_builder.build(db_path)

if __name__ == "__main__":
    print("Creating test graph...")
    graph = create_test_graph()
    print("Graph created successfully!") 