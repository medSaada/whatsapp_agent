from app.schemas.whatsapp import WebhookPayload
from app.services.meta_api_client import MetaAPIClient
from app.services.rag.orchestrator import RAGOrchestrator
from app.core.config import Settings, get_settings
from fastapi import Depends
from app.core.logging import get_logger
from typing import List, Dict, Optional
from app.schemas.whatsapp import Entry

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

    async def process_message(self, entry: Entry):
        """Process incoming WhatsApp entry"""
        for change in entry.changes:
            if change.field == "messages":
                for message in change.value.messages:
                    sender_id = message.from_
                    message_text = message.text.body if message.text else None
                    message_id = message.id
                    
                    logger.info(f"Received message from {sender_id}: '{message_text}'")
                    
                    if message_text in ["Hello!", "Testing"]:
                        await self._handle_template_triggers(message_text, sender_id)
                    else:
                        await self._generate_rag_response(message_text, sender_id, message_id)
                
                for status in change.value.statuses:
                    message_id = status.id
                    recipient_id = status.recipient_id
                    status_value = status.status
                    
                    logger.info(f"Message {message_id} to {recipient_id} has status: {status_value}")
            else:
                logger.warning(f"Unhandled change value type: {change.value}")

    async def _handle_template_triggers(self, message_text: str, sender_id: str):
        """Handle specific messages that should trigger template responses"""
        trigger_keywords = [
            "Hello!", "Testing", "sessions", "جلسات", "دروس", "حصص"
        ]
        
        user_phone = f"+{sender_id}"
        
        if any(keyword in message_text for keyword in trigger_keywords):
            try:
                await self.send_sessions_template_with_video(user_phone)
            except Exception as e:
                await self.send_welcome_message(user_phone)
        
    async def send_sessions_template_with_video(self, recipient_phone: str):
        """Send template message with video header"""
        
        video_component = {
            "type": "header",
            "parameters": [
                {
                    "type": "video",
                    "video": {
                        "id": "1217067246113775",
                    }
                }
            ]
        }
        
        button_component = {
            "type": "button",
            "sub_type": "quick_reply",
            "index": "0",
            "parameters": [
                {
                    "type": "text",
                    "text": "نعم، أريد معرفة المزيد"
                }
            ]
        }
        
        try:
            response = await self.meta_api_client.send_template_message(
                to_phone=recipient_phone,
                template_name="sessions", 
                template_language="ar",
                components=[video_component, button_component]
            )
            
            if response and response.get('messages'):
                logger.info(f"Sessions template with video sent successfully to {recipient_phone}")
            else:
                logger.error(f"Failed to send sessions template to {recipient_phone}")
                await self.send_welcome_message(recipient_phone)
                
        except Exception as e:
            logger.error(f"Error sending sessions template: {e}", exc_info=True)
            await self.send_welcome_message(recipient_phone)

    async def send_welcome_message(self, recipient_phone: str):
        """Send a simple welcome text message as fallback"""
        welcome_text = """
        مرحباً بكم في أكاديمية Geniats! 
        
        نحن نقدم دورات تعليمية في البرمجة للأطفال من سن 6 إلى 15 سنة.
        
        يمكنكم التواصل معنا لمعرفة المزيد عن برامجنا التعليمية.
        """
        
        try:
            await self.meta_api_client.send_text_message(recipient_phone, welcome_text.strip())
            logger.info(f"Welcome message sent to {recipient_phone}")
        except Exception as e:
            logger.error(f"Error sending welcome message: {e}", exc_info=True)

    async def _generate_rag_response(self, user_message: str, sender_id: str, message_id: str):
        """Generate intelligent response using the stateful RAG agent"""
        
        try:
            if not self.rag_orchestrator or not self.rag_orchestrator.is_ready():
                logger.warning("RAG system not ready, using fallback response")
                
                fallback_response = self._get_fallback_response(user_message)
                user_phone = f"+{sender_id}"
                await self.meta_api_client.send_text_message(user_phone, fallback_response)
                return

            conversation_id = sender_id
            response = await self.rag_orchestrator.generate_response_async(
                user_message=user_message,
                conversation_id=conversation_id
            )
            
            logger.info(f"Generated RAG response for conversation '{sender_id}': '{user_message[:50]}...'")
            
            user_phone = f"+{sender_id}"
            await self.meta_api_client.send_text_message(user_phone, response)
            
        except Exception as e:
            logger.error(f"Error generating RAG response for conversation '{sender_id}': {e}", exc_info=True)
            
            fallback_response = self._get_fallback_response(user_message)
            user_phone = f"+{sender_id}"
            await self.meta_api_client.send_text_message(user_phone, fallback_response)

    def send_sessions_template(self, recipient_phone: str):
        """
        Sends the Arabic sessions template with video to the user.
        This is a synchronous version for backward compatibility.
        """
        video_component = {
            "type": "header",
            "parameters": [
                {
                    "type": "video",
                    "video": {
                        "id": "1217067246113775",
                    }
                }
            ]
        }
        
        button_component = {
            "type": "button",
            "sub_type": "quick_reply",
            "index": "0",
            "parameters": [
                {
                    "type": "text",
                    "text": "نعم، أريد معرفة المزيد"
                }
            ]
        }
        
        try:
            response = self.meta_api_client.send_template_message(
                to_phone=recipient_phone,
                template_name="sessions", 
                template_language="ar",
                components=[video_component, button_component]
            )
            
            if response and response.get('messages'):
                logger.info(f"Sessions template with video sent successfully to {recipient_phone}")
            else:
                logger.error(f"Failed to send sessions template to {recipient_phone}")
                self.send_welcome_template(recipient_phone)
                
        except Exception as e:
            logger.error(f"Error sending sessions template: {e}", exc_info=True)
            self.send_welcome_template(recipient_phone)

    def send_welcome_template(self, recipient_phone: str):
        """Send a simple welcome text message as fallback"""
        welcome_text = """
        مرحباً بكم في أكاديمية Geniats! 
        
        نحن نقدم دورات تعليمية في البرمجة للأطفال من سن 6 إلى 15 سنة.
        
        يمكنكم التواصل معنا لمعرفة المزيد عن برامجنا التعليمية.
        """
        
        try:
            self.meta_api_client.send_text_message(recipient_phone, welcome_text.strip())
            logger.info(f"Welcome message sent to {recipient_phone}")
        except Exception as e:
            logger.error(f"Error sending welcome message: {e}", exc_info=True)

    def _get_fallback_response(self, user_message: str) -> str:
        """Simple keyword-based responses as fallback"""
        user_message_lower = user_message.lower()
        
        if any(word in user_message_lower for word in ["price", "سعر", "tarif", "coût"]):
            return "أسعارنا 490 درهم شهرياً، مع عروض خاصة متاحة. تواصل معنا لمعرفة المزيد!"
        elif any(word in user_message_lower for word in ["schedule", "horaire", "وقت", "جدول"]):
            return "لدينا جلسات أسبوعية مباشرة مع مهندسين متخصصين. تواصل معنا لمعرفة المواعيد المتاحة."
        else:
            return "مرحباً! نحن هنا لمساعدتكم. تواصلوا معنا للحصول على معلومات أكثر عن دوراتنا في البرمجة." 