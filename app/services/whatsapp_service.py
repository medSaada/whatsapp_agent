from app.schemas.whatsapp import WebhookPayload
from app.services.meta_api_client import MetaAPIClient
from app.services.rag.orchestrator import RAGOrchestrator
from app.core.config import Settings, get_settings
from fastapi import Depends
import logging

logger = logging.getLogger(__name__)

class WhatsAppService:
    def __init__(self, settings: Settings = Depends(get_settings)):
        """
        Initializes the service with settings and a Meta API client.
        FastAPI's dependency injection will provide the settings.
        """
        self.settings = settings
        self.meta_api_client = MetaAPIClient(settings=self.settings)
        
        # Initialize RAG Orchestrator with existing vector store
        self.rag_orchestrator = RAGOrchestrator(
            vector_store_path="data/vector_store/test_store_documents",
            collection_name="test_collection",
            model_name="gpt-4",
            temperature=0.2
        )
        
        logger.info("WhatsAppService initialized with RAG capabilities")

    def process_message(self, payload: WebhookPayload):
        """Processes an incoming webhook payload."""
        for entry in payload.entry:
            for change in entry.changes:
                # Handle incoming messages
                if change.value.messages:
                    message_data = change.value.messages[0]
                    if message_data['type'] == 'text':
                        sender_id = message_data['from']
                        message_text = message_data['text']['body']
                        logging.info(f"Received message from {sender_id}: '{message_text}'")
                        
                        # Generate intelligent response using RAG
                        reply_message = self._generate_rag_response(message_text)
                        self.meta_api_client.send_text_message(sender_id, reply_message)
                
                # Handle message status updates
                elif change.value.statuses:
                    status_data = change.value.statuses[0]
                    status = status_data['status']
                    recipient_id = status_data['recipient_id']
                    message_id = status_data['id']
                    logging.info(f"Message {message_id} to {recipient_id} has status: {status}")

                else:
                    logging.warning(f"Unhandled change value type: {change.value}")
    
    def _generate_rag_response(self, user_message: str) -> str:
        """
        Generate intelligent response using RAG (same approach as test_generation.py)
        """
        try:
            # Check if RAG system is ready
            if not self.rag_orchestrator.is_ready():
                logger.warning("RAG system not ready, using fallback response")
                return self._get_fallback_response(user_message)
            
            # Generate RAG response using the same approach as test_generation.py
            response = self.rag_orchestrator.answer_question(user_message, k=5)
            
            # Log the interaction
            logger.info(f"Generated RAG response for: '{user_message[:50]}...'")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return self._get_fallback_response(user_message)
    
    def _get_fallback_response(self, user_message: str) -> str:
        """
        Get fallback response when RAG is not available
        """
        # Simple keyword-based responses as fallback
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['bonjour', 'hello', 'salut']):
            return "Bonjour ! Je suis Fatima-Zahra, votre assistante Geniats. Comment puis-je vous aider aujourd'hui ?"
        
        elif any(word in message_lower for word in ['prix', 'tarif', 'coût', 'combien']):
            return "Je serais ravie de vous parler de nos tarifs ! Pouvez-vous me dire l'âge de votre enfant pour que je vous propose l'offre la plus adaptée ?"
        
        elif any(word in message_lower for word in ['cours', 'programme', 'formation']):
            return "Nos cours sont spécialement conçus pour les enfants de 6 à 15 ans. Que souhaitez-vous savoir exactement sur nos programmes ?"
        
        elif any(word in message_lower for word in ['inscription', 'inscrire', 'inscrit']):
            return "Parfait ! Pour l'inscription, j'aurais besoin de quelques informations. Quel est l'âge de votre enfant ?"
        
        else:
            return "Merci pour votre message ! Je suis là pour vous aider avec toutes vos questions sur Geniats. Que souhaitez-vous savoir ?" 