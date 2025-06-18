from httpx import AsyncClient
import pytest
from unittest.mock import MagicMock

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

async def test_verify_webhook_success(client: AsyncClient):
    """
    Test successful webhook verification (GET request).
    """
    test_settings = client.app.dependency_overrides[client.app.router.dependencies[0].dependency]()
    verify_token = test_settings.META_VERIFY_TOKEN
    
    response = await client.get(
        f"/api/v1/whatsapp/webhook?hub.mode=subscribe&hub.challenge=12345&hub.verify_token={verify_token}"
    )
    
    assert response.status_code == 200
    assert response.text == "12345"

async def test_verify_webhook_failure(client: AsyncClient):
    """
    Test failed webhook verification due to incorrect token.
    """
    response = await client.get(
        "/api/v1/whatsapp/webhook?hub.mode=subscribe&hub.challenge=12345&hub.verify_token=wrong_token"
    )
    
    assert response.status_code == 403

async def test_process_webhook_message(client: AsyncClient, mocker):
    """
    Test successful processing of a text message (POST request).
    """
    # Mock the WhatsAppService to prevent real processing
    mock_service_instance = MagicMock()
    mocker.patch(
        "app.api.v1.endpoints.whatsapp.WhatsAppService",
        return_value=mock_service_instance
    )
    
    # A sample payload matching the Pydantic schema
    test_payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "123",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {},
                    "contacts": [],
                    "messages": [{"from": "1234567890", "type": "text", "text": {"body": "hello"}}]
                },
                "field": "messages"
            }]
        }]
    }

    response = await client.post("/api/v1/whatsapp/webhook", json=test_payload)
    
    # Assert that the endpoint returns 200 OK
    assert response.status_code == 200
    
    # Assert that the process_message method of the service was called once
    mock_service_instance.process_message.assert_called_once() 