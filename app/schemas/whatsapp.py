from pydantic import BaseModel, Field
from typing import List, Optional

# Models for incoming webhook payload from Meta
# Based on: https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/components

class WebhookChangeValue(BaseModel):
    messaging_product: str
    metadata: dict
    contacts: List[dict]
    messages: List[dict]

class WebhookChange(BaseModel):
    value: WebhookChangeValue
    field: str

class WebhookEntry(BaseModel):
    id: str
    changes: List[WebhookChange]

class WebhookPayload(BaseModel):
    object: str
    entry: List[WebhookEntry] 