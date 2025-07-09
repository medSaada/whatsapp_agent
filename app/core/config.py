import os
from pydantic_settings import BaseSettings
from typing import List
from app.core.logging import get_logger

logger = get_logger()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    META_ACCESS_TOKEN: str
    META_VERIFY_TOKEN: str
    META_WABA_ID: str
    META_PHONE_NUMBER_ID: str
    OPENAI_API_KEY: str
    HF_TOKEN: str = ""
    COHERE_API_KEY: str = ""
    
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "default"
    LANGCHAIN_TRACING_V2: bool = False
    
    GRAPH_API_URL: str = "https://graph.facebook.com/v21.0"
    
    DOCUMENT_PATH_1: str = "data/documents/manual_data_fz.txt"
    DOCUMENT_PATH_2: str = "data/documents/datagenerated_assistant.txt"
    
    @property
    def DOCUMENT_PATHS(self) -> List[str]:
        """Return a list of all document paths"""
        paths = []
        if self.DOCUMENT_PATH_1:
            paths.append(self.DOCUMENT_PATH_1)
        if self.DOCUMENT_PATH_2:
            paths.append(self.DOCUMENT_PATH_2)
        return paths
    
    def ensure_langsmith_env_vars(self):
        """Set LangSmith environment variables if they're configured"""
        if self.LANGSMITH_TRACING and self.LANGSMITH_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = str(self.LANGCHAIN_TRACING_V2)
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGSMITH_ENDPOINT
            os.environ["LANGCHAIN_API_KEY"] = self.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGSMITH_PROJECT
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings() -> Settings:
    return Settings()