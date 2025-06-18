from app.schemas.whatsapp import WebhookPayload
from app.services.meta_api_client import MetaAPIClient
from app.core.config import Settings, get_settings
from fastapi import Depends
import logging

class WhatsAppService:
    def __init__(self, settings: Settings = Depends(get_settings)):
        """
        Initializes the service with settings and a Meta API client.
        FastAPI's dependency injection will provide the settings.
        """
        self.settings = settings
        self.meta_api_client = MetaAPIClient(settings=self.settings)

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
                        
                        # --- Your agent logic goes here ---
                        reply_message = f"You said: {message_text}"
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