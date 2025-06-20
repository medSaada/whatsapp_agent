import os
from pydantic_settings import BaseSettings

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
    LANGSMITH_API_KEY: str
    COHERE_API_KEY: str


    # Application Settings
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    API_V1_STR: str = "/api/v1"
    
    # Graph API
    GRAPH_API_VERSION: str = "v20.0"
    
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
    
    class Config:     
        env_file = ".env"


def get_settings():
    #this function will be used to get the settings creating a one instance of the settings class
    return Settings()