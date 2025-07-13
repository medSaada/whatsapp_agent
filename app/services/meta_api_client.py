import requests
from app.core.config import Settings
from app.core.logging import get_logger
from typing import List, Dict, Optional
import mimetypes
from pathlib import Path
import httpx
from typing import List, Dict, Any, Optional

logger = get_logger()

class MetaAPIClient:
    def __init__(self, settings: 'Settings'):
        self.settings = settings
        # The base URL now correctly includes the phone number ID
        self.base_url = f"{settings.GRAPH_API_URL}/{settings.META_PHONE_NUMBER_ID}"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.META_ACCESS_TOKEN}",
        }
        # The client no longer needs base_url or headers, as they are passed per-request
        self.async_client = httpx.AsyncClient()

    async def send_text_message(self, recipient_phone: str, message: str) -> Optional[Dict[str, Any]]:
        """Sends a simple text message to a WhatsApp user."""
        # Corrected URL: no longer appends the phone number ID
        url = f"{self.base_url}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_phone,
            "type": "text",
            "text": {"body": message},
        }
        
        logger.info(f"Sending message to {recipient_phone}: '{message}'")
        try:
            # Pass headers directly in the request
            response = await self.async_client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Message sent successfully to {recipient_phone}. Response: {response.json()}")
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error sending message to {recipient_phone}: {e}")
            logger.error(f"Response body: {e.response.text if e.response else 'No response'}")
            return None

    async def upload_media(self, file_path: str, media_type: str = None) -> Optional[Dict]:
        """
        Upload media file to WhatsApp Cloud API
        
        Args:
            file_path: Path to the local media file
            media_type: Optional media type (image, video, document, audio). Auto-detected if not provided.
            
        Returns:
            Dict with upload results including Media ID, or None if failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        # Auto-detect media type if not provided
        if not media_type:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type:
                if mime_type.startswith('image/'):
                    media_type = 'image'
                elif mime_type.startswith('video/'):
                    media_type = 'video'
                elif mime_type.startswith('audio/'):
                    media_type = 'audio'
                else:
                    media_type = 'document'
            else:
                logger.error(f"Could not determine media type for {file_path}")
                return None
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            logger.error(f"Could not determine MIME type for {file_path}")
            return None
        
        # Corrected URL: no longer appends the phone number ID
        url = f"{self.base_url}/media"
        # Corrected Authorization header to use the proper token
        headers = {
            "Authorization": f"Bearer {self.settings.META_ACCESS_TOKEN}"
        }
        
        try:
            with open(file_path, 'rb') as file:
                files = {
                    'file': (file_path.name, file, mime_type)
                }
                data = {
                    'type': media_type,
                    'messaging_product': 'whatsapp'
                }
                
                logger.info(f"Uploading {media_type} file: {file_path.name}")
                
                # Pass headers directly in the request
                response = await self.async_client.post(url, files=files, data=data, headers=headers)
                response.raise_for_status()
                
                result = response.json()
                media_id = result.get('id')
                
                if media_id:
                    logger.info(f"Successfully uploaded {file_path.name}. Media ID: {media_id}")
                    return {
                        'id': media_id,
                        'media_type': media_type,
                        'file_name': file_path.name,
                        'api_response': result
                    }
                else:
                    logger.error(f"No media ID returned from API: {result}")
                    return None
                    
        except httpx.RequestError as e:
            logger.error(f"Error uploading media {file_path}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    logger.error(f"Detailed error response: {error_details}")
                except:
                    logger.error(f"Response body: {e.response.text}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error uploading media {file_path}: {e}")
            return None

    async def send_template_message(self, to: str, template_name: str, language_code: str, components: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Sends a template message to a WhatsApp user.
        
        Args:
            to: Recipient phone number (with country code, without +)
            template_name: Name of the approved template
            language_code: Language code (e.g., 'en_US', 'ar')
            components: Optional list of template components with parameters
        """
        # Corrected URL: no longer appends the phone number ID
        url = f"{self.base_url}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {
                    "code": language_code
                }
            }
        }
        
        # Add components if provided
        if components:
            payload["template"]["components"] = components
            
        logger.info(f"Sending template '{template_name}' to {to}")
        logger.debug(f"Template payload: {payload}")  # Debug log to see the exact payload
        
        try:
            # Pass headers directly in the request
            response = await self.async_client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Template message sent successfully to {to}. Response: {response.json()}")
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error sending template message to {to}: {e}")
            
            # Get detailed error information
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    logger.error(f"Detailed error response: {error_details}")
                except:
                    logger.error(f"Response body: {e.response.text}")
            else:
                logger.error("No response available")
            
            return None

    async def send_template_with_media(self, to: str, template_name: str, language_code: str, 
                                media_type: str, media_url: str, body_parameters: Optional[List[str]] = None):
        """
        Sends a template message with media (image, video, or document).
        
        Args:
            to: Recipient phone number
            template_name: Name of the approved template
            language_code: Language code
            media_type: Type of media ('image', 'video', 'document')
            media_url: URL of the media file
            body_parameters: Optional list of parameters for the template body
        """
        components = [
            {
                "type": "header",
                "parameters": [
                    {
                        "type": media_type,
                        media_type: {
                            "link": media_url
                        }
                    }
                ]
            }
        ]
        
        # Add body parameters if provided
        if body_parameters:
            body_component = {
                "type": "body",
                "parameters": [
                    {
                        "type": "text",
                        "text": param
                    } for param in body_parameters
                ]
            }
            components.append(body_component)
            
        return await self.send_template_message(to, template_name, language_code, components)

    async def send_template_with_buttons(self, to: str, template_name: str, language_code: str,
                                  body_parameters: Optional[List[str]] = None,
                                  button_parameters: Optional[List[Dict]] = None):
        """
        Sends a template message with interactive buttons.
        
        Args:
            to: Recipient phone number
            template_name: Name of the approved template
            language_code: Language code
            body_parameters: Optional list of parameters for the template body
            button_parameters: Optional list of button parameters
        """
        components = []
        
        # Add body parameters if provided
        if body_parameters:
            body_component = {
                "type": "body",
                "parameters": [
                    {
                        "type": "text",
                        "text": param
                    } for param in body_parameters
                ]
            }
            components.append(body_component)
            
        # Add button parameters if provided
        if button_parameters:
            for button_param in button_parameters:
                button_component = {
                    "type": "button",
                    "sub_type": button_param.get("sub_type", "quick_reply"),
                    "index": button_param.get("index", 0),
                    "parameters": [
                        {
                            "type": "payload",
                            "payload": button_param.get("payload", "")
                        }
                    ]
                }
                components.append(button_component)
                
        return await self.send_template_message(to, template_name, language_code, components)

    async def send_simple_template(self, to: str, template_name: str, language_code: str = "ar_MA"):
        """
        Sends a simple template message without any parameters.
        Perfect for templates that don't require variable substitution.
        """
        return await self.send_template_message(to, template_name, language_code) 