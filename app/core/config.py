import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Meta WhatsApp API Credentials
    META_ACCESS_TOKEN: str
    META_VERIFY_TOKEN: str
    META_WABA_ID: str
    META_PHONE_NUMBER_ID: str

    # Application Settings
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    API_V1_STR: str = "/api/v1"
    
    # Graph API
    GRAPH_API_VERSION: str = "v20.0"
    
    @property
    def GRAPH_API_URL(self) -> str:
        return f"https://graph.facebook.com/{self.GRAPH_API_VERSION}"

settings = Settings() 