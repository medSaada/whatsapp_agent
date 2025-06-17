from app.schemas.whatsapp import WebhookPayload
from app.services.meta_api_client import meta_api_client
import logging

class WhatsAppService:
    def process_message(self, payload: WebhookPayload):
        """Processes an incoming webhook payload."""
        for entry in payload.entry:
            for change in entry.changes:
                if change.field == "messages":
                    message_data = change.value.messages[0]
                    sender_id = message_data['from']
                    
                    if message_data['type'] == 'text':
                        message_text = message_data['text']['body']
                        logging.info(f"Received message from {sender_id}: '{message_text}'")
                        
                        # --- Your agent logic goes here ---
                        # For now, we just echo the message back.
                        reply_message = f"You said: {message_text}"
                        meta_api_client.send_text_message(sender_id, reply_message)
                        
whatsapp_service = WhatsAppService() 