#!/usr/bin/env python3
"""
Test script to demonstrate memory management functionality.
This script simulates multiple interactions to trigger the memory threshold.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app.services.rag.orchestrator import RAGOrchestrator

# Configure logging to see the memory management logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_memory_management():
    """Test the memory management functionality"""
    
    print("🧪 Testing Memory Management Functionality")
    print("=" * 50)
    
    # Initialize the orchestrator with a low threshold for testing
    orchestrator = RAGOrchestrator(
        vector_store_path="data/vector_store",
        collection_name="production_collection",
        model_name="gpt-4.1",
        temperature=0.2,
        memory_threshold=3  # Low threshold for testing
    )
    
    if not orchestrator.is_ready():
        print("❌ RAG system not ready. Please ensure the vector store is populated.")
        return
    
    # Test conversation ID
    conversation_id = "test_memory_management_001"
    
    # Simulate multiple interactions
    test_questions = [
        "Hello, what is Geniats?",
        "Tell me about your programs for kids",
        "What are the prices?",
        "How long are the courses?",
        "Do you have online classes?",
        "What age groups do you serve?",
        "Can you tell me more about the curriculum?",
        "How do I enroll my child?"
    ]
    
    print(f"📝 Testing with {len(test_questions)} questions (threshold: 3)")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Question {i}: {question}")
        print("-" * 30)
        
        try:
            response = orchestrator.answer_question(question, conversation_id)
            print(f"✅ Response: {response[:100]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print(f"📊 Interaction {i} completed")
    
    print("\n" + "=" * 50)
    print("🏁 Memory management test completed!")
    print("Check the logs above to see the memory management in action.")

if __name__ == "__main__":
    test_memory_management() 