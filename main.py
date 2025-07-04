import os
import sys
from app.core.config import get_settings
from app.core.logging import get_logger

# Get settings once at the application level
settings = get_settings()

logger = get_logger()

settings.ensure_langsmith_env_vars()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.v1.api import api_router
from app.services.rag.orchestrator import RAGOrchestrator
import os

# LangSmith imports and setup (now safe because env vars are set)
try:
    from langsmith import tracing_context, Client
    
    
    langsmith_client = Client()
    print(f"✅ LangSmith connected successfully to project: {settings.LANGSMITH_PROJECT}")
    
    # Optional: Get project URL for debugging
    try:
        runs = list(langsmith_client.list_runs(project_name=settings.LANGSMITH_PROJECT, limit=1))
        if runs:
            print(f" Latest run URL: {runs[0].url}")
        else:
            print("No runs found yet - this will be the first!")
    except Exception as url_error:
        print(f"ould not fetch run URL (this is normal for new projects): {url_error}")
        
except Exception as langsmith_error:
    print(f"LangSmith initialization failed: {langsmith_error}")
    print("Application will continue without LangSmith tracing")
    langsmith_client = None

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
   