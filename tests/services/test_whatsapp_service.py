from app.services.whatsapp_service import WhatsAppService
from app.schemas.whatsapp import WebhookPayload
from app.core.config import Settings
from unittest.mock import MagicMock, patch

def test_process_message_sends_echo():
    """
    Test that the WhatsAppService correctly processes a text message
    and calls the MetaAPIClient to send an echo response.
    """
    # 1. Arrange
    # Create mock settings
    mock_settings = Settings(
        META_ACCESS_TOKEN="fake", META_VERIFY_TOKEN="fake", 
        META_WABA_ID="fake", META_PHONE_NUMBER_ID="fake"
    )

    # Mock the MetaAPIClient class that the service depends on
    with patch("app.services.whatsapp_service.MetaAPIClient") as MockMetaAPIClient:
        # Create an instance of the service, which will now use the mocked client
        service = WhatsAppService(settings=mock_settings)
        
        # This is the instance of the client *inside* the service
        mock_api_client_instance = MockMetaAPIClient.return_value

        # Create a sample payload
        test_payload = WebhookPayload.model_validate({
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "123",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {},
                        "contacts": [{"wa_id": "1234567890", "profile": {"name": "test"}}],
                        "messages": [{"from": "1234567890", "type": "text", "text": {"body": "hello"}}]
                    },
                    "field": "messages"
                }]
            }]
        })

        # 2. Act
        service.process_message(test_payload)

        # 3. Assert
        # Check that the send_text_message method was called correctly
        mock_api_client_instance.send_text_message.assert_called_once_with(
            "1234567890",       # 'to' number
            "You said: hello"   # expected echo message
        ) 