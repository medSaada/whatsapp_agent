import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from fastapi import FastAPI
from app.api.v1.api import api_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="WhatsApp Agent",
    description="A modular agent to interact with WhatsApp.",
    version="0.1.0",
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    return {"message": "Med Saada WhatsApp Agent is running"}

if __name__ == "__main__":
    # So we don't need to write uvicorn main:app --reload every time
    import uvicorn
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT) 
    # from app.services.rag import test_basic_chunking, test_combined_chunking, test_semantic_chunking, test_separator_chunking
    # test_basic_chunking()
    # test_combined_chunking()
    # test_semantic_chunking()
    # test_separator_chunking()
