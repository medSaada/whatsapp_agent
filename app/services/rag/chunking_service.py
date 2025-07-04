# Importing the dependencies 
from langchain_openai import ChatOpenAI
import langchain_core.prompts 
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from typing import Optional, List, Tuple
import os
import sys
import locale
from app.core.logging import get_logger

def setup_test_environment():
    """Setup Python path for testing"""
    import sys
    import os
    
    # Get the project root (where main.py is located)
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    
    # Add project root to Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return project_root

# Setup project paths for imports
try:
    # Try to import from project root
    from setup_paths import resolve_path, PROJECT_ROOT
except ImportError:
    # Fallback if setup_paths is not available
    def get_project_root():
        """Get the project root directory regardless of where the script is executed from"""
        # Method 1: Look for a marker file (like main.py) in parent directories
        current_dir = os.path.abspath(os.path.dirname(__file__))
        
        while current_dir != os.path.dirname(current_dir):  # Stop at root
            if os.path.exists(os.path.join(current_dir, 'main.py')):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        
        # Method 2: Fallback - assume we're in app/services/rag/ and go up 4 levels
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
        return project_root

    def resolve_path(relative_path: str) -> str:
        """Convert relative path to absolute path from project root"""
        project_root = get_project_root()
        return os.path.join(project_root, relative_path)
    
    PROJECT_ROOT = get_project_root()

# Configure logging
logger = get_logger()

class ArabicTextHandler:
    """Handles Arabic text display and encoding issues"""
    
    @staticmethod
    def setup_arabic_console():
        """Setup console for proper Arabic text display"""
        try:
            if sys.platform.startswith('win'):
                os.system('chcp 65001 > nul')
                os.system('reg add "HKCU\\Console" /v "FaceName" /t REG_SZ /d "Consolas" /f > nul 2>&1')
                
            try:
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            except locale.Error:
                pass
                
        except Exception as e:
            logger.warning(f"Could not setup Arabic console: {e}")

    @staticmethod
    def print_arabic_properly(text: str, encoding: str = 'utf-8'):
        """Print Arabic text with proper formatting"""
        try:
            if isinstance(text, str):
                try:
                    import arabic_reshaper
                    from bidi.algorithm import get_display
                    
                    reshaped_text = arabic_reshaper.reshape(text)
                    arabic_text = get_display(reshaped_text)
                    
                    logger.info("🔤 Properly shaped Arabic text:")
                    logger.info(arabic_text)
                    
                except ImportError:
                    try:
                        from bidi.algorithm import get_display
                        arabic_text = get_display(text)
                        logger.info("📝 Bidi-processed text:")
                        logger.info(arabic_text)
                    except ImportError:
                        logger.info("📄 Raw text (libraries not available):")
                        logger.info(text)
            else:
                ArabicTextHandler.print_arabic_properly(str(text), encoding)
                
        except Exception as e:
            logger.error(f"Error printing Arabic text: {e}")
            logger.info(repr(text))

class DocumentLoader:
    """Handles document loading and encoding issues"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        logger.info(f"DocumentLoader initialized with project root: {self.project_root}")
    
    def load_single_document(self, text_path: str) -> Optional[List]:
        """Load a single document with proper error handling"""
        try:
            # Handle both absolute and relative paths
            if os.path.isabs(text_path):
                full_path = text_path
            else:
                # Convert relative path to absolute path from project root
                full_path = resolve_path(text_path)
            
            logger.info(f"Loading document from: {full_path}")
            
            # Check if file exists
            if not os.path.exists(full_path):
                logger.error(f"File does not exist: {full_path}")
                return None
            
            # Try to load with UTF-8 encoding
            loader = TextLoader(full_path, encoding='utf-8')
            documents = loader.load()
            
            logger.info(f"Successfully loaded {len(documents)} document(s) from {text_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document from {text_path}: {e}")
            return None
    
    def load_multiple_documents(self, text_paths: List[str]) -> List:
        """Load multiple documents and return a flat list"""
        all_documents = []
        
        for text_path in text_paths:
            documents = self.load_single_document(text_path)
            if documents:
                all_documents.extend(documents)
            else:
                logger.warning(f"Skipping {text_path} due to loading error")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents

class DocumentProcessor:
    """Handles document processing and metadata extraction"""
    
    @staticmethod
    def extract_text_and_metadata(documents: List) -> Tuple[List[str], List[dict]]:
        """Extract text and metadata from documents"""
        text_list = []
        metadata_list = []
        
        for doc in documents:
            try:
                if isinstance(doc, list):
                    # Handle nested document lists
                    for page in doc:
                        text_list.append(page.page_content)
                        metadata_list.append(page.metadata)
                else:
                    # Handle single documents
                    text_list.append(doc.page_content)
                    metadata_list.append(doc.metadata)
                    
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                continue
        
        logger.info(f"Extracted {len(text_list)} text segments and {len(metadata_list)} metadata entries")
        return text_list, metadata_list

class ChunkingService:
    """Main chunking service with improved structure and error handling"""
    
    def __init__(self, text_paths: List[str]):
        self.text_paths = text_paths
        self.document_loader = DocumentLoader()
        self.document_processor = DocumentProcessor()
        self.documents = None
        
        # Setup Arabic text handling
        ArabicTextHandler.setup_arabic_console()
    
    def load_documents(self) -> bool:
        """Load all documents and store them"""
        logger.info("Loading documents...")
        self.documents = self.document_loader.load_multiple_documents(self.text_paths)
        
        if not self.documents:
            logger.error("No documents were loaded successfully")
            return False
        
        logger.info(f"Successfully loaded {len(self.documents)} documents")
        return True
    
    def get_text_and_metadata(self) -> Tuple[Optional[List[str]], Optional[List[dict]]]:
        """Get text and metadata from loaded documents"""
        if not self.documents:
            logger.error("No documents loaded. Call load_documents() first.")
            return None, None
        
        return self.document_processor.extract_text_and_metadata(self.documents)
    
    def separator_based_chunking(self, separator: str, chunk_size: int, chunk_overlap: int) -> Optional[List]:
        """Create chunks using separator-based splitting"""
        logger.info(f"Starting separator-based chunking with separator: '{separator}'")
        
        text_list, metadata_list = self.get_text_and_metadata()
        if not text_list or not metadata_list:
            logger.error("No text or metadata available for chunking")
            return None
        
        try:
            text_splitter = CharacterTextSplitter(
                separator=separator, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.create_documents(text_list, metadata_list)
            
            logger.info(f"Created {len(chunks)} chunks using separator-based chunking")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in separator-based chunking: {e}")
            return None
    
    def semantic_chunking(self, breakpoint_threshold_type: str, embeddings: OpenAIEmbeddings, 
                         input_chunks: Optional[List] = None) -> Optional[List]:
        """Create chunks using semantic analysis"""
        logger.info(f"Starting semantic chunking with threshold type: {breakpoint_threshold_type}")
        
        if input_chunks is None:
            # Use loaded documents
            text_list, metadata_list = self.get_text_and_metadata()
        else:
            # Use provided chunks
            text_list, metadata_list = self.document_processor.extract_text_and_metadata(input_chunks)
        
        if not text_list or not metadata_list:
            logger.error("No text or metadata available for semantic chunking")
            return None
        
        try:
            semantic_chunker = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=breakpoint_threshold_type
            )
            chunks = semantic_chunker.create_documents(text_list, metadata_list)
            
            logger.info(f"Created {len(chunks)} chunks using semantic chunking")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            return None
    
    def semantic_with_separator_chunking(self, separator: str, breakpoint_threshold_type: str, 
                                       embeddings: OpenAIEmbeddings) -> Optional[List]:
        """Combine separator-based and semantic chunking"""
        logger.info("Starting combined separator + semantic chunking")
        
        # First, do separator-based chunking
        separator_chunks = self.separator_based_chunking(
            separator=separator, 
            chunk_size=100, 
            chunk_overlap=20
        )
        
        if not separator_chunks:
            logger.error("Separator-based chunking failed")
            return None
        
        # Then apply semantic chunking to the results
        final_chunks = self.semantic_chunking(
            breakpoint_threshold_type=breakpoint_threshold_type,
            embeddings=embeddings,
            input_chunks=separator_chunks
        )
        
        if final_chunks:
            logger.info(f"Combined chunking completed. Final chunks: {len(final_chunks)}")
        
        return final_chunks
    
    def debug_documents(self):
        """Debug method to inspect loaded documents"""
        if not self.documents:
            logger.error("No documents loaded")
            return
        
        logger.info(f"=== DEBUG: Document Information ===")
        logger.info(f"Total documents: {len(self.documents)}")
        
        for i, doc in enumerate(self.documents[:3]):  # Show first 3 documents
            logger.info(f"Document {i+1}:")
            logger.info(f"  Type: {type(doc)}")
            logger.info(f"  Content preview: {str(doc.page_content)[:100]}...")
            logger.info(f"  Metadata: {doc.metadata}")
            logger.info("---")

# Test functions for individual testing
def test_basic_chunking():
    """Test basic chunking functionality with in-memory data"""
    # Setup Python path for imports
    setup_test_environment()
    
    logger.info("=== Testing Basic Chunking with In-Memory Data ===")
    
    # Create test data in memory
    test_texts = [
        "This is the first document. It contains some text about artificial intelligence and machine learning.",
        "This is the second document. It discusses natural language processing and chatbots.",
        "This is the third document. It covers deep learning and neural networks."
    ]
    
    # Create a simple test chunker that works with in-memory data
    try:
        from langchain_core.documents import Document
        
        # Create documents from test texts
        documents = [Document(page_content=text, metadata={"source": f"test_{i}"}) for i, text in enumerate(test_texts)]
        
        logger.info(f"Created {len(documents)} test documents")
        
        # Test basic text processing
        text_list = [doc.page_content for doc in documents]
        metadata_list = [doc.metadata for doc in documents]
        
        logger.info(f"Extracted {len(text_list)} text segments")
        logger.info(f"Sample text: {text_list[0][:50]}...")
        
        # Test simple chunking
        from langchain.text_splitter import CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = text_splitter.create_documents(text_list, metadata_list)
        
        logger.info(f"Created {len(chunks)} chunks")
        logger.info(f"First chunk: {chunks[0].page_content[:50]}...")
        
        logger.info("✅ Basic chunking test successful!")
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Basic chunking test failed: {e}")
        return None

def test_document_loading():
    """Test document loading functionality"""
    # Setup Python path for imports
    setup_test_environment()
    
    logger.info("=== Testing Document Loading ===")
    
    try:
        # Try to get paths from settings
        from app.core.config import get_settings
        settings = get_settings()
        text_paths = settings.DOCUMENT_PATHS
        logger.info("Using paths from settings")
    except Exception as e:
        logger.warning(f"Could not load settings: {e}")
        logger.info("Using hardcoded paths for testing")
        # Fallback to hardcoded paths
        text_paths = ["data/datagenerated_assistant.txt", "data/manual_data_fz.txt"]
    
    chunker = ChunkingService(text_paths)
    
    success = chunker.load_documents()
    if success:
        chunker.debug_documents()
    else:
        logger.error("Document loading failed")
    
    return chunker

def test_separator_chunking():
    """Test separator-based chunking"""
    logger.info("=== Testing Separator-Based Chunking ===")
    
    chunker = test_document_loading()
    if not chunker:
        return None
    
    chunks = chunker.separator_based_chunking(
        separator="--", 
        chunk_size=100, 
        chunk_overlap=20
    )
    
    if chunks:
        logger.info(f"Separator chunking successful: {len(chunks)} chunks")
        logger.info(f"First chunk preview: {str(chunks[0].page_content)[:100]}...")
    else:
        logger.error("Separator chunking failed")
    
    return chunks

def test_semantic_chunking(settings=None):
    """Test semantic chunking"""
    logger.info("=== Testing Semantic Chunking ===")
    
    chunker = test_document_loading()
    if not chunker:
        return None
    
    # Create embeddings with settings if provided
    if settings:
        embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    else:
        # Fallback for testing without settings
        embeddings = OpenAIEmbeddings()
        
    chunks = chunker.semantic_chunking(
        breakpoint_threshold_type="percentile",
        embeddings=embeddings
    )
    
    if chunks:
        logger.info(f"Semantic chunking successful: {len(chunks)} chunks")
        logger.info(f"First chunk preview: {str(chunks[0].page_content)[:100]}...")
    else:
        logger.error("Semantic chunking failed")
    
    return chunks

def test_combined_chunking(settings=None):
    """Test combined separator + semantic chunking"""
    logger.info("=== Testing Combined Chunking ===")
    
    chunker = test_document_loading()
    if not chunker:
        return None
    
    # Create embeddings with settings if provided
    if settings:
        embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    else:
        # Fallback for testing without settings
        embeddings = OpenAIEmbeddings()
        
    chunks = chunker.semantic_with_separator_chunking(
        separator="--",
        breakpoint_threshold_type="percentile",
        embeddings=embeddings
    )
    
    if chunks:
        logger.info(f"Combined chunking successful: {len(chunks)} chunks")
        logger.info(f"First chunk preview: {str(chunks[0].page_content)[:100]}...")
    else:
        logger.error("Combined chunking failed")
    
    return chunks

    
    
