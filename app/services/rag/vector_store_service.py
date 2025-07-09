from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any, Tuple, FrozenSet
from dataclasses import dataclass
from pathlib import Path
from app.core.logging import get_logger
from datetime import datetime

# Configure logging
logger = get_logger()

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
    """Vector store service using Qdrant with improved error handling."""
    
    def __init__(self, 
                 config: VectorStoreConfig,
                 embedding_model: Optional[OpenAIEmbeddings] = None):
        """Initialize Qdrant vector store service"""
        if not isinstance(config, VectorStoreConfig):
            raise TypeError("config must be a VectorStoreConfig instance")
        
        self._config = config
        self._embedding_model = embedding_model or OpenAIEmbeddings(
            model=config.embedding_model_name
        )
        
        store_path = Path(config.store_path)
        store_path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(store_path))
        
        self._vector_stores: Dict[str, Qdrant] = {}
        self._collections_info: Dict[str, CollectionInfo] = {}
        
        logger.info(f"VectorStoreService (Qdrant) initialized at path: {config.store_path}")
    
    @property
    def config(self) -> VectorStoreConfig:
        return self._config
    
    @property
    def available_collections(self) -> FrozenSet[str]:
        return frozenset(c.name for c in self._client.get_collections().collections)
    
    def create_collection(self, 
                         collection_name: str,
                         documents: List[Document]) -> CollectionInfo:
        """
        Creates a new Qdrant collection and indexes documents using the service's primary client.
        This method avoids creating a new, conflicting client instance.
        """
        if self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' already exists")
        
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        logger.info(f"Creating and populating Qdrant collection '{collection_name}' with {len(documents)} documents")
        
        try:
            vector_size = len(self._embedding_model.embed_query("test"))
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
            
            vector_store = Qdrant(
                client=self._client,
                collection_name=collection_name,
                embeddings=self._embedding_model
            )

            vector_store.add_documents(documents, wait=True)
            self._vector_stores[collection_name] = vector_store
            
            now = datetime.now()
            doc_count = self._client.count(collection_name=collection_name, exact=True).count
            collection_info = CollectionInfo(
                name=collection_name,
                document_count=doc_count,
                created_at=now,
                last_updated=now,
                embedding_model=self._config.embedding_model_name
            )
            self._collections_info[collection_name] = collection_info
            
            logger.info(f"Collection '{collection_name}' created successfully in Qdrant")
            return collection_info
            
        except Exception as e:
            logger.error(f"Error creating Qdrant collection '{collection_name}': {e}", exc_info=True)
            if self.collection_exists(collection_name):
                self._client.delete_collection(collection_name=collection_name)
            raise RuntimeError(f"Failed to create Qdrant collection '{collection_name}': {e}")
    
    def load_collection(self, collection_name: str) -> Optional[CollectionInfo]:
        """Loads an existing Qdrant collection from disk."""
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' not found in Qdrant.")
            return None
        
        if collection_name in self._vector_stores:
            return self._collections_info.get(collection_name)

        try:
            logger.info(f"Loading collection '{collection_name}' from Qdrant")
            vector_store = Qdrant(
                client=self._client,
                collection_name=collection_name,
                embeddings=self._embedding_model,
            )
            
            doc_count = self._client.count(collection_name=collection_name, exact=True).count
            self._vector_stores[collection_name] = vector_store
            
            now = datetime.now()
            collection_info = CollectionInfo(
                name=collection_name, document_count=doc_count, created_at=now,
                last_updated=now, embedding_model=self._config.embedding_model_name
            )
            self._collections_info[collection_name] = collection_info
            
            logger.info(f"Collection '{collection_name}' loaded successfully with {doc_count} documents")
            return collection_info
            
        except Exception as e:
            logger.error(f"Error loading Qdrant collection '{collection_name}': {e}", exc_info=True)
            return None
    
    def add_documents_to_collection(self, 
                                   collection_name: str,
                                   documents: List[Document]) -> CollectionInfo:
        """Add documents to an existing collection."""
        if not self.load_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist. Create it first.")
        
        vector_store = self._vector_stores[collection_name]
        
        try:
            vector_store.add_documents(documents, wait=True)
            
            current_info = self._collections_info[collection_name]
            new_doc_count = self._client.count(collection_name=collection_name, exact=True).count
            updated_info = CollectionInfo(
                name=collection_name, document_count=new_doc_count, created_at=current_info.created_at,
                last_updated=datetime.now(), embedding_model=current_info.embedding_model
            )
            self._collections_info[collection_name] = updated_info
            return updated_info
        except Exception as e:
            logger.error(f"Error adding documents to collection '{collection_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to add documents: {e}")
    
    def search_collection(self, 
                         collection_name: str,
                         query: str,
                         k: int = 5,
                         filter_dict: Optional[Dict[str, Any]] = None) -> SearchResult:
        """Search in a specific collection, converting dict filter to Qdrant filter."""
        if not self.load_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
        vector_store = self._vector_stores[collection_name]
        
        try:
            qdrant_filter = self._create_qdrant_filter(filter_dict)
            
            search_start = datetime.now()
            results = vector_store.similarity_search(query=query, k=k, filter=qdrant_filter)
            
            return SearchResult(
                documents=tuple(results), query=query, collection_name=collection_name,
                search_time=search_start, total_results=len(results)
            )
        except Exception as e:
            logger.error(f"Error searching collection '{collection_name}': {e}", exc_info=True)
            raise RuntimeError(f"Search failed: {e}")

    def _create_qdrant_filter(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        if not filter_dict:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
                for key, value in filter_dict.items()
            ]
        )
    
    def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        if not self.collection_exists(collection_name):
            return None
        return self.load_collection(collection_name)
    
    def list_collections(self) -> Tuple[CollectionInfo, ...]:
        return tuple(info for info in (self.get_collection_info(c.name) for c in self._client.get_collections().collections) if info)
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection and all its data."""
        if not self.collection_exists(collection_name):
            logger.warning(f"Attempted to delete non-existent collection '{collection_name}'")
            return False
        
        try:
            result = self._client.delete_collection(collection_name=collection_name)
            if result:
                self._vector_stores.pop(collection_name, None)
                self._collections_info.pop(collection_name, None)
            return result
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}", exc_info=True)
            return False
    
    def get_service_stats(self) -> Dict[str, Any]:
        collections = self.list_collections()
        return {
            "total_collections": len(collections),
            "total_documents": sum(c.document_count for c in collections),
            "embedding_model": self._config.embedding_model_name,
            "store_path": self._config.store_path,
            "collections": [{"name": c.name, "document_count": c.document_count} for c in collections]
        }
    
    def cleanup(self):
        """Clean up resources."""
        self._client.close()
        self._vector_stores.clear()
        self._collections_info.clear()
        logger.info("VectorStoreService (Qdrant) resources cleaned up.")

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in Qdrant."""
        return collection_name in {c.name for c in self._client.get_collections().collections}
        
    def as_retriever(self, collection_name: str, **kwargs) -> 'VectorStoreRetriever':
        """Returns a LangChain retriever for a specific Qdrant collection."""
        if not self.load_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist or could not be loaded.")
        
        vector_store = self._vector_stores[collection_name]
        
        if 'search_kwargs' in kwargs and 'filter' in kwargs['search_kwargs']:
            filter_dict = kwargs['search_kwargs'].get('filter')
            kwargs['search_kwargs']['filter'] = self._create_qdrant_filter(filter_dict)

        return vector_store.as_retriever(**kwargs)

