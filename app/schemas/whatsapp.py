from pydantic import BaseModel, Field
from typing import List, Optional

# Models for incoming webhook payload from Meta
# Based on: https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/components

class WebhookChangeValue(BaseModel):
    messaging_product: str
    metadata: dict
    contacts: Optional[List[dict]] = None
    messages: Optional[List[dict]] = None
    statuses: Optional[List[dict]] = None

class WebhookChange(BaseModel):
    value: WebhookChangeValue
    field: str

class WebhookEntry(BaseModel):
    id: str
    changes: List[WebhookChange]

class WebhookPayload(BaseModel):
    object: str
    entry: List[WebhookEntry] 