import sys
import logging
import pytest
from langchain_openai import OpenAIEmbeddings
from app.services.rag.vector_store_service import VectorStoreService, VectorStoreConfig
from app.core.config import get_settings
from app.services.rag.chunking_service import ChunkingService

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def setup_services():
    """
    A pytest fixture to set up the necessary services and data for the test module.
    This runs once per test module, making tests more efficient.
    """
    settings = get_settings()
    embeddings = OpenAIEmbeddings()
    text_paths = settings.DOCUMENT_PATHS
    
    # Setup VectorStoreService
    collection_name = "test_retrieval_collection"
    config = VectorStoreConfig(
        store_path="data/vector_store/test_store_retrieval",
        collection_name=collection_name
    )
    vector_store_service = VectorStoreService(config, embedding_model=embeddings)

    # Clean up any old collections before starting
    if vector_store_service.collection_exists(collection_name):
        logger.info(f"Deleting pre-existing test collection: {collection_name}")
        vector_store_service.delete_collection(collection_name)
    
    yield settings, embeddings, text_paths, vector_store_service, collection_name
    
    # --- Teardown ---
    # This code runs after all tests in the module have completed
    logger.info(f"Cleaning up test collection: {collection_name}")
    vector_store_service.delete_collection(collection_name)
    vector_store_service.cleanup()

def test_chunking_and_ingestion(setup_services):
    """
    Tests the document loading, chunking, and ingestion process.
    """
    settings, embeddings, text_paths, vector_store_service, collection_name = setup_services
    
    # --- Chunking ---
    chunker = ChunkingService(text_paths)
    success = chunker.load_documents()
    assert success, "Documents should be loaded successfully"
    
    chunks = chunker.separator_based_chunking(separator="\\n", chunk_size=1000, chunk_overlap=200)
    assert chunks, "Chunks should be created successfully"
    logger.info(f"Created {len(chunks)} chunks.")

    # --- Ingestion ---
    logger.info(f"Creating collection '{collection_name}' for ingestion test.")
    collection_info = vector_store_service.create_collection(collection_name, chunks)
    assert collection_info is not None, "Collection should be created successfully"
    assert collection_info.document_count == len(chunks), "Document count should match number of chunks"
    
    # Verify the collection now exists
    assert vector_store_service.collection_exists(collection_name), "Collection should exist after creation"

def test_retrieval_from_collection(setup_services):
    """
    Tests the search and retrieval functionality from an existing collection.
    This test depends on the successful completion of the ingestion test.
    """
    settings, embeddings, text_paths, vector_store_service, collection_name = setup_services

    # Ensure the collection exists before searching
    if not vector_store_service.collection_exists(collection_name):
        pytest.skip("Skipping retrieval test because the collection was not created in the previous step.")

    # --- Retrieval ---
    query = "what are the prices?"
    logger.info(f"Testing retrieval with query: '{query}'")
    search_result = vector_store_service.search_collection(collection_name, query, k=3)
    
    assert search_result is not None, "Search should return a result object"
    assert search_result.total_results > 0, "Search should find at least one relevant document"
    logger.info(f"Retrieved {search_result.total_results} results.")
    
    # Log the content of the top result for manual inspection
    if search_result.documents:
        logger.info(f"Top result content: {search_result.documents[0].page_content[:200]}...")

