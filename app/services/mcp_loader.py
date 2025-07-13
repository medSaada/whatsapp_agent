import asyncio
import json
from pathlib import Path
from typing import List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.logging import get_logger

logger = get_logger()

_mcp_client: Optional[MultiServerMCPClient] = None
_mcp_tools: Optional[List] = None


async def initialize_mcp_client(config_path: str = "mcp_config.json"):
    """
    Initializes the MCP client and fetches tools asynchronously.
    This should be called once during application startup.
    """
    global _mcp_client, _mcp_tools
    if _mcp_client:
        return

    try:
        logger.info("Initializing MCP client...")
        config = json.loads(Path(config_path).read_text())
        
        _mcp_client = MultiServerMCPClient(connections=config["mcpServers"])
        _mcp_tools = await _mcp_client.get_tools()
        
        logger.info(f"MCP client initialized successfully with {_mcp_tools and len(_mcp_tools) or 0} tools.")

        # --- Debugging Start ---
        if _mcp_tools:
            logger.info("--- Loaded MCP Tools Details ---")
            for i, tool in enumerate(_mcp_tools):
                logger.info(f"Tool {i+1}:")
                logger.info(f"  Name: {tool.name}")
                logger.info(f"  Description: {tool.description}")
                logger.info(f"  Args: {tool.args}")
                # For Google Calendar, we might want to check for auth-related attributes.
                # This is speculative, but let's see if it has any helpful attributes.
                if 'google' in tool.name.lower() or 'calendar' in tool.name.lower():
                    logger.info("  [Google Tool Detected] Checking for auth attributes...")
                    # The following attributes are just guesses based on common library patterns
                    auth_related_attrs = ['credentials', 'creds', 'api_key', 'client_secret']
                    for attr in auth_related_attrs:
                        if hasattr(tool, attr):
                            value = getattr(tool, attr)
                            logger.info(f"    - Found attribute '{attr}': {'PRESENT' if value else 'MISSING'}")
                        else:
                            logger.info(f"    - Attribute '{attr}' not found.")
            logger.info("---------------------------------")
        # --- Debugging End ---

    except FileNotFoundError:
        logger.warning(f"MCP config file not found at '{config_path}'. No MCP tools will be loaded.")
        _mcp_tools = []
    except Exception as e:
        logger.error(f"Failed to initialize MCP client: {e}", exc_info=True)
        _mcp_tools = []


def get_mcp_tools() -> List:
    """
    Returns the cached list of MCP tools.
    Assumes initialize_mcp_client has already been called.
    """
    return _mcp_tools if _mcp_tools is not None else []

async def close_mcp_client():
    """Closes the MCP client and its subprocesses."""
    if _mcp_client:
        logger.info("Closing MCP client...")
        await _mcp_client.aclose()
        logger.info("MCP client closed.") 