from app.schemas.whatsapp import WebhookPayload
from app.services.meta_api_client import MetaAPIClient
from app.services.rag.orchestrator import RAGOrchestrator
from app.core.config import Settings, get_settings
from fastapi import Depends
import logging

logger = logging.getLogger(__name__)

class WhatsAppService:
    def __init__(self, rag_orchestrator: RAGOrchestrator, settings: Settings):
        """
        Initializes the service with a shared RAG orchestrator and settings.
        This follows the dependency injection pattern for better maintainability.
        """
        self.settings = settings
        self.meta_api_client = MetaAPIClient(settings=self.settings)
        self.rag_orchestrator = rag_orchestrator
        
        logger.info("WhatsAppService initialized with a shared RAG orchestrator")

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
                        
                        # Generate intelligent response using the stateful RAG agent
                        reply_message = self._generate_rag_response(sender_id, message_text)
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
    
    def _generate_rag_response(self, sender_id: str, user_message: str) -> str:
        """
        Generates an intelligent response using the stateful RAG agent.
        The sender_id is used to maintain conversation history.
        """
        try:
            # Check if RAG system is ready
            if not self.rag_orchestrator.is_ready():
                logger.warning("RAG system not ready, using fallback response")
                return self._get_fallback_response(user_message)
            
            # Generate a response using the agent, which maintains state via conversation_id
            response = self.rag_orchestrator.answer_question(
                question=user_message,
                conversation_id=sender_id 
            )
            
            # Log the interaction
            logger.info(f"Generated RAG response for conversation '{sender_id}': '{user_message[:50]}...'")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating RAG response for conversation '{sender_id}': {e}", exc_info=True)
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