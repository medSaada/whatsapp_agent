import os
import sys
from app.core.config import get_settings

# Get settings once at the application level
settings = get_settings()

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.v1.api import api_router
from app.services.rag.orchestrator import RAGOrchestrator
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    Initializes a single RAGOrchestrator instance to be shared across the application,
    preventing file-locking errors and improving maintainability.
    """
    # Create RAGOrchestrator with settings dependency injection
    app.state.rag_orchestrator = RAGOrchestrator(
        settings=settings,  # Pass settings as dependency
        vector_store_path="data/vector_store",
        collection_name="production_collection",
        model_name="gpt-4.1",
        temperature=0.2,
        memory_threshold=6  # Summarize and wipe memory every 6 interactions
    )
    yield
    # Cleanup, like closing database connections, can happen here after `yield`
    app.state.rag_orchestrator.cleanup()

app = FastAPI(
    title="WhatsApp Agent",
    description="WhatsApp agent with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Med Saada WhatsApp Agent is running"}

if __name__ == "__main__":
    import uvicorn
    # print(settings.LANGSMITH_PROJECT)
    # from langsmith import Client
    # client = Client()
    # url = next(client.list_runs(project_name="whatsapp-agent-project")).url
    # print(url)
    uvicorn.run(app, host="0.0.0.0", port=8000)
