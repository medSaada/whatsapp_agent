from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings
import logging
from pathlib import Path

from app.services.rag.vector_store_service import VectorStoreService, VectorStoreConfig
from app.services.rag.generation_service import GenerationService

logger = logging.getLogger(__name__)

# Use the same prompt template from test_generation.py
QNA_TEMPLATE = "\n".join([
    # Persona & Goal
    "You are **Fatima-Zahra**, a **client support expert** at **Geniats**, an **e-learning coding academy** for **Moroccan kids aged 6–15**.",
    "Your mission is to **respond to client messages in Moroccan Darija** (or in French only when necessary), with the goal of **convincing them to join and purchase our offer**.",
    "You must sound like a **real Moroccan person**, **not an AI**—friendly, respectful, helpful and professional.",

    # Language Rules
    "## ⚠️ Language Rules",
    "1. **Darija lines** use Arabic script and punctuation: comma `،`, question mark `؟`, exclamation `!`.",
    "2. **French lines** use Latin script and punctuation: `, . ? ! : ;`.",
    "3. **One line = one language**. To switch, end the line, insert a blank line, then continue in the other language.",
    "4. **Never guess a Darija word**. If unsure, first check `document-conversation.pdf` or `data_caption.pdf`; if still unsure, reply in French or \"I don't know.\"",
    "5. **If you choose to respond in Darija, you must write entirely in Arabic letters**—no Latin transliteration.",

    # Reasoning Process
    "## 🧩 Reasoning Process (Internal Steps)",
    "1. **Comprehend** the client's question: identify their needs, doubts, and what they need to know before buying.",
    "3. If you find an example, **adapt** it with a soft sales mindset: highlight benefits, address pain points, and guide them toward next steps.",
    "4. **Compose** your answer in clear, correct Darija (in Arabic letters) or French if necessary.",
    "5. **Verify** punctuation, script directionality, and no mixed-language lines.",

    # Output Rules
    "## 💬 Output Rules",
    "- Deliver **one complete message**—no lists or step-by-step breakdowns.",
    "- Tone: **warm, respectful, professional**, with natural Darija (and French where needed).",
    "- Length: **as short or long as necessary** to fully answer the question.",
    "- **If the client flirts** and you can tell it's a man, gently remind him of professional boundaries; otherwise respond kindly.",

    # Conversation placeholders
    "### Context:",
    "{context}",
    "",
    "### Client Message:",
    "{question}",
    "",
    "### Answer:",
])

class RAGOrchestrator:
    """Simple RAG orchestrator for WhatsApp integration"""
    
    def __init__(self, 
                 vector_store_path: str = "data/vector_store/test_store_documents",
                 collection_name: str = "test_collection",
                 model_name: str = "gpt-4",
                 temperature: float = 0.2):
        """
        Initialize RAG Orchestrator with existing vector store
        """
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        
        # Initialize services
        self._initialize_services(model_name, temperature)
        
        logger.info(f"RAG Orchestrator initialized with collection: {collection_name}")
    
    def _initialize_services(self, model_name: str, temperature: float):
        """Initialize vector store and generation services"""
        # Vector store configuration
        config = VectorStoreConfig(
            store_path=self.vector_store_path,
            collection_name=self.collection_name
        )
        
        # Initialize services
        self.vector_store_service = VectorStoreService(config)
        self.generation_service = GenerationService(
            model_name=model_name,
            temperature=temperature,
            prompt_template=QNA_TEMPLATE
        )
    
    def answer_question(self, question: str, k: int = 5) -> str:
        """
        Answer a question using RAG (same approach as test_generation.py)
        """
        try:
            # Load collection
            collection = self.vector_store_service.load_collection(self.collection_name)
            if not collection:
                logger.warning(f"Collection '{self.collection_name}' not found")
                return "Je ne trouve pas d'information pertinente pour répondre à votre question."
            
            logger.info(f"Loaded collection with {collection.document_count} documents")
            
            # Search for relevant documents
            search_result = self.vector_store_service.search_collection(
                collection_name=self.collection_name,
                query=question,
                k=k
            )
            
            if not search_result.documents:
                return "Je ne trouve pas d'information pertinente pour répondre à votre question."
            
            # Generate response using the same approach as test_generation.py
            answer = self.generation_service.generate_response_with_documents(
                question=question,
                documents=list(search_result.documents)
            )
            
            logger.info(f"Generated answer for question: '{question[:50]}...'")
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Désolé, j'ai rencontré une erreur: {str(e)}"
    
    def is_ready(self) -> bool:
        """Check if RAG system is ready"""
        try:
            collection = self.vector_store_service.load_collection(self.collection_name)
            return collection is not None and collection.document_count > 0
        except Exception:
            return False
