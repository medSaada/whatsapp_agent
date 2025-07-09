import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from app.core.logging import get_logger
from app.core.config import Settings, get_settings
import warnings

logger = get_logger()

def find_project_root():
    """Find the project root directory by looking for main.py"""
    current_path = Path(__file__).resolve()
    
    for parent in current_path.parents:
        if (parent / "main.py").exists():
            return parent
    
    return current_path.parent.parent.parent

def setup_python_path():
    """Setup Python path to allow imports from project root"""
    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

PROJECT_ROOT = setup_python_path()

try:
    from app.core.config import Settings, get_settings
    _has_settings = True
except ImportError:
    _has_settings = False

logger = get_logger()

class ChunkingService:
    """
    Service for text chunking and document processing.
    
    Supports various text chunking strategies including recursive character splitting
    and semantic chunking with Arabic text handling capabilities.
    """
    
    def __init__(self, embedding_model=None, settings: Optional[Settings] = None):
        """
        Initialize the chunking service with an optional embedding model.
        
        Args:
            embedding_model: Optional embedding model for semantic chunking
            settings: Optional settings object for configuration
        """
        self.embedding_model = embedding_model
        self.settings = settings or (get_settings() if _has_settings else None)
        
        logger.info(f"ChunkingService initialized - Embeddings: {'Available' if embedding_model else 'Not provided'}")
    
    def load_document_from_file(self, file_path: str) -> Optional[Document]:
        """
        Load a single document from a file path.
        
        Args:
            file_path: Path to the text file to load
            
        Returns:
            Document object or None if loading fails
        """
        try:
            if not os.path.isabs(file_path):
                file_path = os.path.join(PROJECT_ROOT, file_path)
            
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            if content.strip():
                return Document(
                    page_content=content,
                    metadata={"source": file_path, "type": "text_file"}
                )
            else:
                logger.warning(f"File is empty: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading document from {file_path}: {e}")
            return None
    
    def load_documents_from_paths(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple documents from a list of file paths.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for file_path in file_paths:
            doc = self.load_document_from_file(file_path)
            if doc:
                documents.append(doc)
            else:
                logger.warning(f"Skipping file: {file_path}")
        
        logger.info(f"Loaded {len(documents)} documents from {len(file_paths)} file paths")
        return documents
    
    def load_documents_from_text(self, texts: List[str], sources: Optional[List[str]] = None) -> List[Document]:
        """
        Create Document objects from text strings.
        
        Args:
            texts: List of text strings
            sources: Optional list of source identifiers
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for i, text in enumerate(texts):
            if isinstance(text, list):
                for j, subtext in enumerate(text):
                    source = sources[i] if sources and i < len(sources) else f"text_{i}_{j}"
                    documents.append(Document(page_content=subtext, metadata={"source": source}))
            else:
                source = sources[i] if sources and i < len(sources) else f"text_{i}"
                documents.append(Document(page_content=text, metadata={"source": source}))
        
        logger.info(f"Created {len(documents)} documents from text input")
        return documents
    
    def _detect_arabic_text(self, text: str) -> bool:
        """Check if text contains Arabic characters"""
        arabic_chars = any('\u0600' <= char <= '\u06FF' for char in text)
        return arabic_chars
    
    def _handle_arabic_text(self, text: str):
        """Handle Arabic text display and processing"""
        try:
            from arabic_reshaper import reshape
            reshaped_text = reshape(text)
            
            try:
                from bidi.algorithm import get_display
                arabic_text = get_display(reshaped_text)
                
                logger.info("Properly shaped Arabic text:")
                logger.info(arabic_text)
                
            except ImportError:
                try:
                    from bidi.algorithm import get_display
                    arabic_text = get_display(text)
                    logger.info("Bidi-processed text:")
                    logger.info(arabic_text)
                except ImportError:
                    logger.info("Raw text (libraries not available):")
                    logger.info(text)
        except ImportError:
            try:
                from bidi.algorithm import get_display
                arabic_text = get_display(text)
                logger.info("Bidi-processed text:")
                logger.info(arabic_text)
            except ImportError:
                logger.info("Raw text (libraries not available):")
                logger.info(text)
    
    def process_documents(self, 
                         documents: Optional[List[Document]] = None,
                         chunks: Optional[List[Document]] = None) -> List[Document]:
        """
        Process documents or chunks by handling Arabic text and logging content.
        
        Args:
            documents: Optional list of documents to process
            chunks: Optional list of chunks to process
            
        Returns:
            List of processed documents/chunks
        """
        items_to_process = documents if documents is not None else chunks
        
        if not items_to_process:
            logger.warning("No documents or chunks provided for processing")
            return []
        
        logger.info(f"Processing {len(items_to_process)} items")
        
        for i, item in enumerate(items_to_process):
            text_content = item.page_content
            
            logger.info(f"--- Item {i+1}/{len(items_to_process)} ---")
            logger.info(f"Source: {item.metadata.get('source', 'Unknown')}")
            logger.info(f"Content length: {len(text_content)} characters")
            
            if self._detect_arabic_text(text_content):
                logger.info("Arabic text detected - applying special handling")
                self._handle_arabic_text(text_content[:200])
            else:
                logger.info("Non-Arabic text detected")
                logger.info(f"Preview: {text_content[:200]}...")
        
        return items_to_process
    
    def chunk_documents(self, 
                       documents: List[Document], 
                       method: str = "recursive",
                       chunk_size: int = 1000, 
                       chunk_overlap: int = 200) -> List[Document]:
        """
        Chunk documents using specified method.
        
        Args:
            documents: List of documents to chunk
            method: Chunking method ("recursive" or "semantic")
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunked documents
        """
        if not documents:
            logger.warning("No documents provided for chunking")
            return []
        
        logger.info(f"Starting {method} chunking of {len(documents)} documents")
        
        if method == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = splitter.split_documents(documents)
            
        elif method == "semantic" and self.embedding_model:
            semantic_splitter = SemanticChunker(
                embeddings=self.embedding_model,
                breakpoint_threshold_type="percentile"
            )
            chunks = semantic_splitter.split_documents(documents)
            
        else:
            logger.warning(f"Invalid method '{method}' or no embedding model for semantic chunking")
            return []
        
        logger.info(f"Created {len(chunks)} chunks using {method} method")
        return chunks
    
    def hybrid_chunk(self, 
                    documents: List[Document],
                    recursive_chunk_size: int = 2000,
                    semantic_threshold: str = "percentile") -> List[Document]:
        """
        Apply hybrid chunking: recursive first, then semantic.
        
        Args:
            documents: Documents to chunk
            recursive_chunk_size: Initial chunk size for recursive splitting
            semantic_threshold: Threshold type for semantic splitting
            
        Returns:
            List of hybrid-chunked documents
        """
        if not self.embedding_model:
            logger.warning("No embedding model available for hybrid chunking, falling back to recursive")
            return self.chunk_documents(documents, "recursive")
        
        logger.info(f"Starting hybrid chunking of {len(documents)} documents")
        
        recursive_chunks = self.chunk_documents(
            documents, 
            method="recursive", 
            chunk_size=recursive_chunk_size
        )
        
        semantic_splitter = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type=semantic_threshold
        )
        
        final_chunks = semantic_splitter.split_documents(recursive_chunks)
        
        logger.info(f"Hybrid chunking complete: {len(final_chunks)} final chunks")
        return final_chunks

def test_basic_processing():
    """Test basic document processing functionality"""
    logger.info("Running basic processing test...")
    
    project_root = setup_python_path()
    
    test_texts = [
        "This is a test document for chunking.",
        "Another test document with different content.",
        "مرحبا بكم في اختبار النص العربي"
    ]
    
    chunker = ChunkingService()
    
    documents = chunker.load_documents_from_text(
        test_texts,
        sources=["test1.txt", "test2.txt", "test3.txt"]
    )
    
    logger.info(f"Created {len(documents)} test documents")
    
    processed_docs = chunker.process_documents(documents=documents)
    
    chunks = chunker.chunk_documents(processed_docs, method="recursive", chunk_size=100)
    
    logger.info(f"Basic processing test completed: {len(chunks)} chunks created")
    return chunks

def test_file_loading():
    """Test loading documents from actual files"""
    logger.info("Running file loading test...")
    
    project_root = setup_python_path()
    
    settings = None
    if _has_settings:
        try:
            settings = get_settings()
        except Exception as e:
            logger.warning(f"Could not load settings: {e}")
    
    test_paths = []
    if settings and hasattr(settings, 'DOCUMENT_PATHS'):
        test_paths = settings.DOCUMENT_PATHS
    else:
        test_paths = [
            "data/documents/manual_data_fz.txt",
            "data/documents/datagenerated_assistant.txt"
        ]
    
    chunker = ChunkingService(settings=settings)
    
    for path in test_paths:
        doc = chunker.load_document_from_file(path)
        if doc:
            logger.info(f"Successfully loaded: {path}")
            logger.info(f"Content length: {len(doc.page_content)} characters")
        else:
            logger.warning(f"Failed to load: {path}")
    
    documents = chunker.load_documents_from_paths(test_paths)
    logger.info(f"File loading test completed: {len(documents)} documents loaded")
    
    return documents

def test_semantic_chunking():
    """Test semantic chunking with OpenAI embeddings"""
    logger.info("Running semantic chunking test...")
    
    project_root = setup_python_path()
    
    settings = None
    if _has_settings:
        try:
            settings = get_settings()
            
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                api_key=settings.OPENAI_API_KEY
            ) if settings else None
        except Exception as e:
            logger.warning(f"Could not create embeddings with settings: {e}")
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    chunker = ChunkingService(embedding_model=embeddings, settings=settings)
    
    test_texts = [
        """Geniats Academy offers coding courses for children aged 6-15. 
        Our interactive sessions help kids learn programming fundamentals through fun projects.
        We use modern teaching methods and provide personalized attention to each student.""",
        
        """Our pricing is competitive at 490 MAD per month. 
        We offer various packages and discounts for siblings. 
        Payment can be made monthly or with special offers for longer commitments."""
    ]
    
    documents = chunker.load_documents_from_text(test_texts)
    
    semantic_chunks = chunker.chunk_documents(documents, method="semantic")
    hybrid_chunks = chunker.hybrid_chunk(documents)
    
    logger.info(f"Semantic chunking test completed:")
    logger.info(f"- Semantic chunks: {len(semantic_chunks)}")
    logger.info(f"- Hybrid chunks: {len(hybrid_chunks)}")
    
    return semantic_chunks, hybrid_chunks

if __name__ == "__main__":
    logger.info("Starting ChunkingService tests...")
    
    try:
        test_basic_processing()
        test_file_loading()
        test_semantic_chunking()
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

    
    
