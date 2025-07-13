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
from app.services.mcp_loader import initialize_mcp_client, close_mcp_client
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pathlib import Path
import os

# LangSmith imports and setup (now safe because env vars are set)
try:
    from langsmith import tracing_context, Client
    
    
    langsmith_client = Client()
    print(f"LangSmith connected successfully to project: {settings.LANGSMITH_PROJECT}")
    
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
    Initializes a single RAGOrchestrator instance and its resources to be shared across the application.
    """
    db_path = "data/sqlite/conversation_memory.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # The checkpointer's lifecycle is now managed by the application's lifespan
    checkpointer = AsyncSqliteSaver.from_conn_string(db_path)
    
    async with checkpointer as memory:
        # Initialize MCP client and tools on startup
        await initialize_mcp_client()

        # The orchestrator will now be created asynchronously with the checkpointer
        app.state.rag_orchestrator = await RAGOrchestrator.create(
            settings=settings, 
            checkpointer=memory,
            vector_store_path="data/vector_store",
            collection_name="production_collection",
            model_name="gpt-4.1",
            temperature=0.2,
            memory_threshold=6
        )
        yield
    
    # Cleanup resources on shutdown
    # The 'async with' block handles the checkpointer connection closure
    app.state.rag_orchestrator.cleanup()
    await close_mcp_client()

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
   