from fastapi import APIRouter, Request, Response, HTTPException, status, Depends
from app.schemas.whatsapp import WebhookPayload
from app.services.whatsapp_service import WhatsAppService
from app.core.config import Settings, get_settings
import logging

router = APIRouter()

@router.get("/webhook")
def verify_webhook(
    request: Request, settings: Settings = Depends(get_settings)
):
    """
    Handles the webhook verification request from Meta.
    It uses the injected settings to access the verify token.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == settings.META_VERIFY_TOKEN:
        logging.info("Webhook verified successfully!")
        return Response(content=challenge, status_code=status.HTTP_200_OK)
    else:
        logging.error("Webhook verification failed.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Failed to verify webhook token.",
        )

@router.post("/webhook")
async def process_webhook(
    payload: WebhookPayload, service: WhatsAppService = Depends()
):
    """
    Handles incoming messages and other events from WhatsApp.
    It uses the injected WhatsAppService to process the payload.
    """
    try:
        service.process_message(payload)
        return Response(status_code=status.HTTP_200_OK)
    except Exception as e:
        logging.error(f"Error processing webhook: {e}", exc_info=True)
        # We return a 200 OK to Meta even if processing fails to prevent
        # them from resending the webhook repeatedly. The error is logged for debugging.
        return Response(status_code=status.HTTP_200_OK) 