import os
import sys
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger()

settings.ensure_langsmith_env_vars()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.v1.api import api_router
from app.services.rag.orchestrator import RAGOrchestrator
import os

try:
    from langsmith import tracing_context, Client
    
    langsmith_client = Client()
    logger.info(f"LangSmith connected successfully to project: {settings.LANGSMITH_PROJECT}")
    
    try:
        runs = list(langsmith_client.list_runs(project_name=settings.LANGSMITH_PROJECT, limit=1))
        if runs:
            logger.info(f"Latest run URL: {runs[0].url}")
        else:
            logger.info("No runs found yet - this will be the first!")
    except Exception as url_error:
        logger.warning(f"Could not fetch run URL (this is normal for new projects): {url_error}")
        
except Exception as langsmith_error:
    logger.warning(f"LangSmith initialization failed: {langsmith_error}")
    logger.info("Application will continue without LangSmith tracing")
    langsmith_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    Initializes a single RAGOrchestrator instance to be shared across the application,
    preventing file-locking errors and improving maintainability.
    """
    app.state.rag_orchestrator = RAGOrchestrator(
        settings=settings, 
        vector_store_path="data/vector_store",
        collection_name="production_collection",
        model_name="gpt-4.1",
        temperature=0.2,
        memory_threshold=6  
    )
    yield
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
   