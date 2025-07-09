from app.schemas.whatsapp import WebhookPayload
from app.services.meta_api_client import MetaAPIClient
from app.services.rag.orchestrator import RAGOrchestrator
from app.core.config import Settings, get_settings
from fastapi import Depends
from app.core.logging import get_logger
from typing import List, Dict, Optional

logger = get_logger()

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
                        logger.info(f"Received message from {sender_id}: '{message_text}'")
                        
                        # Check if this is a specific trigger for template messages
                        if self._should_send_template(message_text):
                            self._send_appropriate_template(sender_id, message_text)
                        else:
                            # Generate intelligent response using the stateful RAG agent
                            reply_message = self._generate_rag_response(sender_id, message_text)
                            self.meta_api_client.send_text_message(sender_id, reply_message)
                
                # Handle message status updates
                elif change.value.statuses:
                    status_data = change.value.statuses[0]
                    status = status_data['status']
                    recipient_id = status_data['recipient_id']
                    message_id = status_data['id']
                    logger.info(f"Message {message_id} to {recipient_id} has status: {status}")

                else:
                    logger.warning(f"Unhandled change value type: {change.value}")
    
    def _should_send_template(self, message_text: str) -> bool:
        """
        Determines if a template message should be sent based on the user's message.
        You can customize this logic based on your business needs.
        """
        message_lower = message_text.lower()
        
        # Keywords that trigger template messages
        template_triggers = [
            'sessions', 'جلسات', 'how sessions look', 'comment se déroulent les sessions',
            'template', 'send template', 'show template'
        ]
        
        return any(trigger in message_lower for trigger in template_triggers)
    
    def _send_appropriate_template(self, sender_id: str, message_text: str):
        """
        Sends the appropriate template based on the user's message.
        """
        message_lower = message_text.lower()
        
        if any(word in message_lower for word in ['sessions', 'جلسات', 'how sessions look']):
            # Send your Arabic template about sessions
            self.send_sessions_template(sender_id)
        else:
            # Default fallback
            self.send_welcome_template(sender_id)
    
    def send_sessions_template(self, recipient_phone: str):
        """
        Sends your specific 'how_the_sessions_look_like' template with video from Facebook Media Library.
        This template has a header with video and a body with a variable placeholder.
        """
        try:
            media_id = "1232963834726789"
            template_parameter = "اذا عندك اي سؤال مرحبا"
            
            response = self.meta_api_client.send_template_message(
                to=recipient_phone,
                template_name="how_the_sessions_look_like",
                language_code="ar_MA",
                components=[
                    {
                        "type": "header",
                        "parameters": [
                            {
                                "type": "video",
                                "video": {
                                    "id": media_id
                                   # "link":"https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4"
                                }
                            }
                        ]
                    },
                    {
                "type": "body",
                "parameters": [
                    {
                        "type": "text",
                        "parameter_name":"var",
                        "text":template_parameter,
                       
                        
                    }
                ]
            },
            
                
                ]
            )
            
            if response:
                logger.info(f"Sessions template with video sent successfully to {recipient_phone}")
                return response
            else:
                logger.error(f"Failed to send sessions template to {recipient_phone}")
                # Send fallback text message
                self.meta_api_client.send_text_message(
                    recipient_phone, 
                    "سأرسل لك معلومات عن كيفية إجراء الجلسات قريباً"
                )
                
        except Exception as e:
            logger.error(f"Error sending sessions template: {e}", exc_info=True)
            # Send fallback text message
            self.meta_api_client.send_text_message(
                recipient_phone, 
                "عذراً، حدث خطأ. سأرسل لك المعلومات المطلوبة قريباً"
            )

    def send_welcome_template(self, recipient_phone: str):
        """
        Sends a welcome template message. This is a fallback for when no specific template is matched.
        You can customize this to use any approved template you have.
        """
        try:
            # For now, send a simple text message as fallback
            # You can replace this with an actual template when you have one approved
            self.meta_api_client.send_text_message(
                recipient_phone, 
                "مرحباً بك! أنا هنا لمساعدتك. يمكنك كتابة 'sessions' أو 'جلسات' لرؤية كيف تبدو جلساتنا."
            )
            logger.info(f"Welcome message sent to {recipient_phone}")
            
        except Exception as e:
            logger.error(f"Error sending welcome message: {e}", exc_info=True)
            # Final fallback
            self.meta_api_client.send_text_message(
                recipient_phone, 
                "مرحباً! كيف يمكنني مساعدتك اليوم؟"
            )
    
    def _generate_rag_response(self, sender_id: str, user_message: str) -> str:
        """
        Generates an intelligent response using the stateful RAG agent with langgraph.
        The sender_id is used to maintain conversation history.
        """
        try:

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
   
    # Not used anymore
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