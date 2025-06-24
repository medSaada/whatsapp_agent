import sys
import logging
from langchain_openai import OpenAIEmbeddings
from app.services.rag import test_basic_chunking, test_combined_chunking, test_semantic_chunking, test_separator_chunking
from app.services.rag.vector_store_service import VectorStoreService, VectorStoreConfig
from app.core.config import get_settings
from app.services.rag.chunking_service import ChunkingService
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def test_chunking_retrieval():
    #get the settings
    settings = get_settings()
    #embeddings
    embeddings = OpenAIEmbeddings()
    #get the text paths
    text_paths = settings.DOCUMENT_PATHS
    logger.info("Using paths from settings")
    #get the chunker
    chunker = ChunkingService(text_paths)
    #load the documents
    success = chunker.load_documents()
    assert success, "Documents should be loaded successfully"
    #get the text and metadata
    text, metadata = chunker.get_text_and_metadata()
    assert text is not None, "Text should be returned successfully"
    assert metadata is not None, "Metadata should be returned successfully"
    chunks = chunker.semantic_with_separator_chunking(
        separator="--",
        breakpoint_threshold_type="percentile",
        embeddings=embeddings
    )
    assert chunks is not None, "Chunks should be returned successfully"
    logger.info(f"Chunks: {chunks[0].page_content}")

    #create a vector store service
    vector_store_service = VectorStoreService(
        VectorStoreConfig(
            store_path="data/vector_store/test_store_documents",
            collection_name="test_collection"
        ),
        embedding_model=embeddings
    )
    #create a collection
    collection_name = "test_collection"
    vector_store_service.create_collection(collection_name, chunks)
    #load the collection
    collection = vector_store_service.load_collection(collection_name)
    assert collection is not None, "Collection should be loaded successfully"

    #search the collection
    query = "prix du service?"
    results = vector_store_service.search_collection(collection_name, query)
    assert results.total_results > 0, "Search results should be returned successfully"
    logger.info(f"Results: {results.documents[0].page_content}")
    

    #clean up
    vector_store_service.cleanup()

