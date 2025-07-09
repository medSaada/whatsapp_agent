import logging
from app.services.rag.chunking_service import ChunkingService
from app.services.rag.vector_store_service import VectorStoreService, VectorStoreConfig
from app.core.config import get_settings
from app.core.logging import get_logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = get_logger()

def main():
    """
    Main function to run the data ingestion process.
    """
    logger.info("Starting data ingestion process...")
    
    try:
        settings = get_settings()
        
        vector_store_path = "data/vector_store"
        collection_name = "production_collection"
        
        config = VectorStoreConfig(
            store_path=vector_store_path,
            collection_name=collection_name
        )
        vector_store_service = VectorStoreService(config)

        if vector_store_service.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists. Deleting it to ensure a fresh start.")
            vector_store_service.delete_collection(collection_name)
        
        logger.info(f"Loading documents from paths: {settings.DOCUMENT_PATHS}")
        chunker = ChunkingService(settings.DOCUMENT_PATHS)
        
        if not chunker.load_documents():
            logger.error("Failed to load documents. Aborting ingestion.")
            return

        chunks = chunker.separator_based_chunking(separator="\n", chunk_size=1000, chunk_overlap=200)
        if not chunks:
            logger.error("No chunks were created from the documents. Aborting ingestion.")
            return
            
        logger.info(f"Successfully created {len(chunks)} chunks.")
        logger.info(f"Documents loaded: {chunks}")

        logger.info(f"Creating collection '{collection_name}' and ingesting {len(chunks)} chunks.")
        collection_info = vector_store_service.create_collection(collection_name, chunks)
        
        if collection_info:
            logger.info(f"Successfully created collection '{collection_name}' with {collection_info.document_count} documents.")
        else:
            logger.error(f"Failed to create collection '{collection_name}'.")

    except Exception as e:
        logger.error(f"An error occurred during the ingestion process: {e}", exc_info=True)
    finally:
        logger.info("Data ingestion process finished.")
        if 'vector_store_service' in locals() and vector_store_service:
            vector_store_service.cleanup()


if __name__ == "__main__":
    main() 