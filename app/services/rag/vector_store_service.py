from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any, Tuple, FrozenSet
from dataclasses import dataclass
from pathlib import Path
import os
import logging
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class VectorStoreConfig:
    """Immutable configuration for vector store"""
    store_path: str
    collection_name: str
    embedding_model_name: str = "text-embedding-ada-002"
    max_documents_per_collection: int = 10000
    
    def __post_init__(self):
        """Fail Fast validation"""
        if not self.store_path or not self.store_path.strip():
            raise ValueError("store_path cannot be empty")
        
        if not self.collection_name or not self.collection_name.strip():
            raise ValueError("collection_name cannot be empty")
        
        if not self.collection_name.replace('_', '').replace('-', '').isalnum():
            raise ValueError("collection_name must contain only alphanumeric characters, hyphens, and underscores")
        
        if self.max_documents_per_collection <= 0:
            raise ValueError("max_documents_per_collection must be positive")
    
    def with_collection_name(self, new_name: str) -> 'VectorStoreConfig':
        """Create new config with different collection name"""
        return VectorStoreConfig(
            store_path=self.store_path,
            collection_name=new_name,
            embedding_model_name=self.embedding_model_name,
            max_documents_per_collection=self.max_documents_per_collection
        )

@dataclass(frozen=True)
class SearchResult:
    """Immutable search result"""
    documents: Tuple[Document, ...]
    query: str
    collection_name: str
    search_time: datetime
    total_results: int
    
    def __post_init__(self):
        """Fail Fast validation"""
        if not self.query.strip():
            raise ValueError("Query cannot be empty")
        
        if not self.collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        
        if self.total_results < 0:
            raise ValueError("total_results cannot be negative")
    
    def filter_by_score(self, min_score: float) -> 'SearchResult':
        """Return new SearchResult with filtered documents"""
        # Note: This assumes documents have a score attribute
        filtered_docs = tuple(
            doc for doc in self.documents 
            if hasattr(doc, 'metadata') and doc.metadata.get('score', 0) >= min_score
        )
        
        return SearchResult(
            documents=filtered_docs,
            query=self.query,
            collection_name=self.collection_name,
            search_time=self.search_time,
            total_results=len(filtered_docs)
        )

@dataclass(frozen=True)
class CollectionInfo:
    """Immutable collection information"""
    name: str
    document_count: int
    created_at: datetime
    last_updated: datetime
    embedding_model: str
    
    def __post_init__(self):
        """Fail Fast validation"""
        if not self.name.strip():
            raise ValueError("Collection name cannot be empty")
        
        if self.document_count < 0:
            raise ValueError("Document count cannot be negative")

class VectorStoreService:
    """Vector store service with improved error handling and immutable results"""
    
    def __init__(self, 
                 config: VectorStoreConfig,
                 embedding_model: Optional[OpenAIEmbeddings] = None):
        """
        Initialize vector store service
        
        Args:
            config: Immutable configuration object
            embedding_model: Optional embedding model (will create default if None)
        """
        # Fail Fast validation
        if not isinstance(config, VectorStoreConfig):
            raise TypeError("config must be a VectorStoreConfig instance")
        
        self._config = config
        self._embedding_model = embedding_model or OpenAIEmbeddings(
            model=config.embedding_model_name
        )
        self._vector_stores: Dict[str, Chroma] = {}
        self._collections_info: Dict[str, CollectionInfo] = {}
        
        # Ensure store directory exists
        store_path = Path(config.store_path)
        store_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VectorStoreService initialized with config: {config}")
    
    @property
    def config(self) -> VectorStoreConfig:
        """Get immutable configuration"""
        return self._config
    
    @property
    def available_collections(self) -> FrozenSet[str]:
        """Get available collection names"""
        return frozenset(self._collections_info.keys())
    
    def create_collection(self, 
                         collection_name: str,
                         documents: List[Document]) -> CollectionInfo:
        """
        Create a new collection with documents
        
        Args:
            collection_name: Name for the collection
            documents: List of documents to add
            
        Returns:
            Immutable CollectionInfo object
            
        Raises:
            ValueError: If validation fails
            RuntimeError: If creation fails
        """
        # Fail Fast validation
        if not collection_name or not collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        
        if not collection_name.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Collection name must contain only alphanumeric characters, hyphens, and underscores")
        
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        if len(documents) > self._config.max_documents_per_collection:
            raise ValueError(f"Too many documents. Max allowed: {self._config.max_documents_per_collection}")
        
        if collection_name in self._vector_stores:
            raise ValueError(f"Collection '{collection_name}' already exists")
        
        # Validate document format
        for i, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise TypeError(f"Document at index {i} is not a Document instance")
            
            if not hasattr(doc, 'page_content') or not doc.page_content.strip():
                raise ValueError(f"Document at index {i} has empty content")
        
        logger.info(f"Creating collection '{collection_name}' with {len(documents)} documents")
        
        try:
            # Create collection-specific path
            collection_path = Path(self._config.store_path) / collection_name
            collection_path.mkdir(exist_ok=True)
            
            # Create vector store
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self._embedding_model,
                persist_directory=str(collection_path)
            )
            
            # Persist immediately
            vector_store.persist()
            
            # Store references
            self._vector_stores[collection_name] = vector_store
            
            # Create collection info
            now = datetime.now()
            collection_info = CollectionInfo(
                name=collection_name,
                document_count=len(documents),
                created_at=now,
                last_updated=now,
                embedding_model=self._config.embedding_model_name
            )
            
            self._collections_info[collection_name] = collection_info
            
            logger.info(f"Collection '{collection_name}' created successfully")
            return collection_info
            
        except Exception as e:
            logger.error(f"Error creating collection '{collection_name}': {e}")
            # Cleanup on failure
            collection_path = Path(self._config.store_path) / collection_name
            if collection_path.exists():
                shutil.rmtree(collection_path, ignore_errors=True)
            raise RuntimeError(f"Failed to create collection '{collection_name}': {e}")
    
    def load_collection(self, collection_name: str) -> Optional[CollectionInfo]:
        """
        Load an existing collection from disk
        
        Args:
            collection_name: Name of collection to load
            
        Returns:
            CollectionInfo if successful, None if not found
        """
        # Fail Fast validation
        if not collection_name or not collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        
        collection_path = Path(self._config.store_path) / collection_name
        
        if not collection_path.exists():
            logger.warning(f"Collection '{collection_name}' not found at {collection_path}")
            return None
        
        try:
            logger.info(f"Loading collection '{collection_name}'")
            
            # Load vector store
            vector_store = Chroma(
                persist_directory=str(collection_path),
                embedding_function=self._embedding_model
            )
            
            # Get document count
            try:
                doc_count = vector_store._collection.count()
            except:
                doc_count = 0
            
            # Store references
            self._vector_stores[collection_name] = vector_store
            
            # Create collection info (we don't have creation date from disk)
            now = datetime.now()
            collection_info = CollectionInfo(
                name=collection_name,
                document_count=doc_count,
                created_at=now,  # Approximation
                last_updated=now,
                embedding_model=self._config.embedding_model_name
            )
            
            self._collections_info[collection_name] = collection_info
            
            logger.info(f"Collection '{collection_name}' loaded successfully with {doc_count} documents")
            return collection_info
            
        except Exception as e:
            logger.error(f"Error loading collection '{collection_name}': {e}")
            return None
    
    def add_documents_to_collection(self, 
                                   collection_name: str,
                                   documents: List[Document]) -> CollectionInfo:
        """
        Add documents to existing collection
        
        Args:
            collection_name: Name of existing collection
            documents: Documents to add
            
        Returns:
            Updated CollectionInfo
        """
        # Fail Fast validation
        if not collection_name or not collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        if collection_name not in self._vector_stores:
            raise ValueError(f"Collection '{collection_name}' does not exist. Create it first.")
        
        # Check document limit
        current_info = self._collections_info[collection_name]
        total_docs = current_info.document_count + len(documents)
        
        if total_docs > self._config.max_documents_per_collection:
            raise ValueError(f"Adding {len(documents)} documents would exceed limit of {self._config.max_documents_per_collection}")
        
        # Validate documents
        for i, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise TypeError(f"Document at index {i} is not a Document instance")
            
            if not hasattr(doc, 'page_content') or not doc.page_content.strip():
                raise ValueError(f"Document at index {i} has empty content")
        
        logger.info(f"Adding {len(documents)} documents to collection '{collection_name}'")
        
        try:
            vector_store = self._vector_stores[collection_name]
            
            # Add documents
            vector_store.add_documents(documents)
            vector_store.persist()
            
            # Update collection info
            updated_info = CollectionInfo(
                name=collection_name,
                document_count=total_docs,
                created_at=current_info.created_at,
                last_updated=datetime.now(),
                embedding_model=current_info.embedding_model
            )
            
            self._collections_info[collection_name] = updated_info
            
            logger.info(f"Successfully added {len(documents)} documents to '{collection_name}'")
            return updated_info
            
        except Exception as e:
            logger.error(f"Error adding documents to collection '{collection_name}': {e}")
            raise RuntimeError(f"Failed to add documents: {e}")
    
    def search_collection(self, 
                         collection_name: str,
                         query: str,
                         k: int = 5,
                         filter_dict: Optional[Dict[str, Any]] = None) -> SearchResult:
        """
        Search in a specific collection
        
        Args:
            collection_name: Name of collection to search
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            Immutable SearchResult object
        """
        # Fail Fast validation
        if not collection_name or not collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if k <= 0:
            raise ValueError("k must be positive")
        
        if k > 100:
            raise ValueError("k cannot exceed 100")
        
        if collection_name not in self._vector_stores:
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        logger.info(f"Searching collection '{collection_name}' for: '{query}' (k={k})")
        
        try:
            vector_store = self._vector_stores[collection_name]
            search_start = datetime.now()
            
            # Perform search
            results = vector_store.similarity_search(
                query=query.strip(),
                k=k,
                filter=filter_dict
            )
            
            # Create immutable result
            search_result = SearchResult(
                documents=tuple(results),
                query=query.strip(),
                collection_name=collection_name,
                search_time=search_start,
                total_results=len(results)
            )
            
            logger.info(f"Search completed. Found {len(results)} results")
            return search_result
            
        except Exception as e:
            logger.error(f"Error searching collection '{collection_name}': {e}")
            raise RuntimeError(f"Search failed: {e}")
    
    def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """
        Get information about a collection
        
        Args:
            collection_name: Name of collection
            
        Returns:
            CollectionInfo if exists, None otherwise
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        
        return self._collections_info.get(collection_name)
    
    def list_collections(self) -> Tuple[CollectionInfo, ...]:
        """
        List all available collections
        
        Returns:
            Tuple of CollectionInfo objects
        """
        return tuple(self._collections_info.values())
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection and all its data
        
        Args:
            collection_name: Name of collection to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Fail Fast validation
        if not collection_name or not collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        
        if collection_name not in self._vector_stores:
            logger.warning(f"Collection '{collection_name}' does not exist")
            return False
        
        logger.info(f"Deleting collection '{collection_name}'")
        
        try:
            # Remove from memory
            del self._vector_stores[collection_name]
            del self._collections_info[collection_name]
            
            # Remove from disk
            collection_path = Path(self._config.store_path) / collection_name
            if collection_path.exists():
                shutil.rmtree(collection_path)
            
            logger.info(f"Collection '{collection_name}' deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}")
            return False
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get service statistics
        
        Returns:
            Dictionary with service statistics
        """
        total_documents = sum(info.document_count for info in self._collections_info.values())
        
        return {
            "total_collections": len(self._collections_info),
            "total_documents": total_documents,
            "embedding_model": self._config.embedding_model_name,
            "store_path": self._config.store_path,
            "collections": [
                {
                    "name": info.name,
                    "document_count": info.document_count,
                    "created_at": info.created_at.isoformat(),
                    "last_updated": info.last_updated.isoformat()
                }
                for info in self._collections_info.values()
            ]
        }
    
    def cleanup(self):
        """Clean up resources"""
        for name, store in self._vector_stores.items():
            try:
                store.persist()
                # Close Chroma client
                if name in self._chroma_clients:
                    self._chroma_clients[name].reset()  # Reset client
                    del self._chroma_clients[name]
            except Exception as e:
                logger.warning(f"Cleanup warning for {name}: {e}")
        
        self._vector_stores.clear()
        self._collections_info.clear()
        
        logger.info("VectorStoreService cleanup completed")

