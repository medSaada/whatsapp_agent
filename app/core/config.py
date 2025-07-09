import os
from pydantic_settings import BaseSettings
from app.core.logging import get_logger

logger = get_logger()

class Settings(BaseSettings):
    """
    Settings for the application
    """
    META_ACCESS_TOKEN: str
    META_VERIFY_TOKEN: str
    META_WABA_ID: str
    META_PHONE_NUMBER_ID: str
    OPENAI_API_KEY: str
    HF_TOKEN: str
    COHERE_API_KEY: str
    LANGSMITH_TRACING: str
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str
    LANGCHAIN_TRACING_V2: str
    # Application Settings
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    API_V1_STR: str = "/api/v1"
    
    # Graph API
    GRAPH_API_VERSION: str = "v23.0"
    
    # Document paths for RAG
    DOCUMENT_PATH_1: str 
    DOCUMENT_PATH_2: str
    
    @property
    def GRAPH_API_URL(self) -> str:
        return f"https://graph.facebook.com/{self.GRAPH_API_VERSION}"
    
    @property
    def DOCUMENT_PATHS(self) -> list[str]:
        """Get list of document paths for RAG"""
        return [self.DOCUMENT_PATH_1, self.DOCUMENT_PATH_2]
    
    def ensure_langsmith_env_vars(self) -> None:
        """
        Ensure LangSmith environment variables are available in os.environ.
        
        LangSmith libraries expect these variables to be in os.environ, not just in our settings.
        This method bridges our centralized settings approach with LangSmith's expectations.
        
        This is called once during application startup to ensure compatibility.
        """
        langsmith_vars = {
            "LANGCHAIN_TRACING_V2": self.LANGCHAIN_TRACING_V2,
            "LANGSMITH_API_KEY": self.LANGSMITH_API_KEY,
            "LANGSMITH_PROJECT": self.LANGSMITH_PROJECT,
            "LANGSMITH_ENDPOINT": self.LANGSMITH_ENDPOINT,
        }
        
        for var_name, var_value in langsmith_vars.items():
            if var_value:  # Only set if value exists
                os.environ[var_name] = var_value
                
       
    
    class Config:     
        env_file = ".env"


def get_settings():
    #this function will be used to get the settings creating a one instance of the settings class
    return Settings()