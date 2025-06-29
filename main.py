import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app.api.v1.api import api_router
from app.core.config import get_settings
from app.services.rag.orchestrator import RAGOrchestrator

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    Initializes a single RAGOrchestrator instance to be shared across the application,
    preventing file-locking errors and improving maintainability.
    """
    app.state.rag_orchestrator = RAGOrchestrator(
        vector_store_path="data/vector_store",
        collection_name="production_collection",
        model_name="gpt-4.1",
        temperature=0.2
    )
    yield
    # Cleanup, like closing database connections, can happen here after `yield`
    app.state.rag_orchestrator.cleanup()

app = FastAPI(
    title="WhatsApp Agent",
    description="A modular agent to interact with WhatsApp.",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    return {"message": "Med Saada WhatsApp Agent is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT)
