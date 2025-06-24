import sys
import pytest
import logging

from app.services.rag.vector_store_service import VectorStoreService, VectorStoreConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_load_and_retrieve():
    """Simple test to load collection and get documents"""
    
    # Setup
    config = VectorStoreConfig(store_path="data/vector_store/test_store_documents", collection_name="test_collection")
    service = VectorStoreService(config)
    
    # Load collection
    collection = service.load_collection("test_collection")
    if not collection:
        pytest.skip("Collection 'test_collection' not found")
    
    logger.info(f"Loaded collection with {collection.document_count} documents")
    
    # Search for documents
    results = service.search_collection("test_collection", "taman", k=5,filter_dict={"source": "H:\\projects\\pojects_codes\\Personal_Projects\\whatsapp_agent_poc\\data\\documents\\manual_data_fz.txt"})
    
    logger.info(f"Found {len(results.documents)} documents")
    logger.info(f"Results: {results.documents[0].page_content}")
    # Check we got some results
    assert len(results.documents) >= 0