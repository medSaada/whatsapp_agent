import pytest
import asyncio
from httpx import AsyncClient
from typing import AsyncGenerator

from main import app
from app.core.config import get_settings, Settings

# Override settings for testing
def get_test_settings() -> Settings:
    return Settings(
        META_ACCESS_TOKEN="test_access_token",
        META_VERIFY_TOKEN="test_verify_token",
        META_WABA_ID="test_waba_id",
        META_PHONE_NUMBER_ID="test_phone_id",
    )

# Override the dependency for the application
app.dependency_overrides[get_settings] = get_test_settings

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def client() -> AsyncGenerator[AsyncClient, None]:
    """
    Create an AsyncClient for making requests to the app in tests.
    """
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c 