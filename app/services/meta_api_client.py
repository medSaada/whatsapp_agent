import requests
from app.core.config import Settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetaAPIClient:
    def __init__(self, settings: Settings):
        """Initializes the client with the application settings."""
        self.base_url = settings.GRAPH_API_URL
        self.access_token = settings.META_ACCESS_TOKEN
        self.sender_phone_id = settings.META_PHONE_NUMBER_ID

    def send_text_message(self, to: str, message: str):
        """Sends a simple text message to a WhatsApp user."""
        url = f"{self.base_url}/{self.sender_phone_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": message},
        }
        
        logger.info(f"Sending message to {to}: '{message}'")
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            logger.info(f"Message sent successfully to {to}. Response: {response.json()}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending message to {to}: {e}")
            logger.error(f"Response body: {e.response.text if e.response else 'No response'}")
            return None 