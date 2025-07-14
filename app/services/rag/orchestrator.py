from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import logging
from pathlib import Path
from langchain.globals import set_debug
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.graph import StateGraph
from app.services.mcp_loader import get_mcp_tools
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.services.rag.vector_store_service import VectorStoreService, VectorStoreConfig
from app.services.rag.generation_service import GenerationService
from app.services.rag.graph.builder import GraphBuilder
from app.services.rag.graph.tools import create_rag_tool
from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger()

# Enable for deep debugging of LangGraph execution
# set_debug(True)

# Note: No specific prompt is needed here anymore, as the GenerationService
# is self-contained.

class RAGOrchestrator:
    """
    High-level orchestrator for the RAG agent.

    It initializes the necessary services and builds the agent graph on-demand
    once the required data collection is available.
    """
    
    # Make __init__ lightweight and synchronous
    def __init__(self, settings: Settings, graph: StateGraph, vector_store_service: VectorStoreService):
        self.settings = settings
        self._graph = graph
        self.vector_store_service = vector_store_service
        self.collection_name = vector_store_service.config.collection_name
        logger.info(f"RAG Orchestrator initialized for collection: {self.collection_name}")

    @classmethod
    async def create(
        cls,
        settings: Settings,
        checkpointer: AsyncSqliteSaver,
        vector_store_path: str,
        collection_name: str,
        planner_model_name: str,
        generator_model_name: str,
        temperature: float,
    ) -> "RAGOrchestrator":
        """Asynchronously creates and initializes the RAGOrchestrator."""
        
        planner_llm = ChatOpenAI(model=planner_model_name, temperature=temperature, api_key=settings.OPENAI_API_KEY)
        generator_llm = ChatOpenAI(model=generator_model_name, temperature=temperature, api_key=settings.OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=settings.OPENAI_API_KEY)
        
        vector_store_service = VectorStoreService(
            VectorStoreConfig(store_path=vector_store_path, collection_name=collection_name),
            embedding_model=embeddings
        )
        generation_service = GenerationService(planner_llm=planner_llm, generator_llm=generator_llm)

        # Fetch all tools
        rag_tool = create_rag_tool(vector_store_service, collection_name)
        mcp_tools = get_mcp_tools()  # This is now a simple sync call to get cached tools
        all_tools = [rag_tool] + mcp_tools

        logger.info(f"Building Async-LangGraph agent with {len(all_tools)} tools...")
        builder = GraphBuilder(
            generation_service=generation_service,
            tools=all_tools,
            settings=settings
        )
        
        # The graph is now built asynchronously using the provided checkpointer
        graph = await builder.build(checkpointer)
        logger.info("Async-LangGraph agent built successfully.")
        
        return cls(settings, graph, vector_store_service)

    async def answer_question(self, question: str, conversation_id: str) -> str:
        """
        Answer a question using the LangGraph agent, with detailed logging.
        The graph is built on the first call if it hasn't been already.
        """
        if not self._graph:
            logger.warning("RAG system not ready, using fallback response.")
            return "Je ne suis pas encore prêt à répondre aux questions. Veuillez réessayer dans un instant."

        try:
            config = {"configurable": {"thread_id": conversation_id}}
            input_data = {
                "messages": [HumanMessage(content=question)],
                "context": "", # Kept for compatibility, but no longer primary
                "rag_context": "",
                "interaction_count": 0
            }
            
            logger.info(f"--- Starting new ASYNC RAG flow for conversation '{conversation_id}' ---")
            logger.info(f"[Memory Management] Processing interaction for conversation: {conversation_id}")
            
            # Use ainvoke for the async graph
            final_state = await self._graph.ainvoke(input_data, config=config)
            
            # The final response is the last message in the list.
            final_response = final_state["messages"][-1].content
            #logger.info(f"Final response: {final_state['messages']}")
            
            # Log the current interaction count
            interaction_count = final_state.get("interaction_count")
            logger.info(f"[Memory Management] Current interaction count for '{conversation_id}': {interaction_count}")
            
            # Determine if tools were used by inspecting the message history
            tool_used = any(isinstance(msg, ToolMessage) for msg in final_state["messages"])
            if tool_used:
                logger.info("[Tool Usage] Agent used tools to generate the response.")
            else:
                logger.info("[Tool Usage] Agent did not use any tools and answered directly.")

            if final_response:
                logger.info(f"--- RAG flow finished for conversation '{conversation_id}' ---")
                return final_response
            else:
                # This path should ideally not be reached with the new logic.
                logger.warning(f"Could not extract final response for conversation '{conversation_id}'")
                return "Je n'ai pas pu générer de réponse."

        except Exception as e:
            logger.error(f"Error in async RAG flow for '{conversation_id}': {e}", exc_info=True)
            # Per user request, do not send a message on error, just log it.
            return ""
    
    def is_ready(self) -> bool:
        """Check if RAG system is ready by checking if the collection exists."""
        try:
            logger.info(f"Checking if collection '{self.collection_name}' exists... result: {self.vector_store_service.collection_exists(self.collection_name)}")
            return self.vector_store_service.collection_exists(self.collection_name)
        except Exception as e:
            logger.error(f"Error checking if RAG is ready: {e}", exc_info=True)
            return False

    def cleanup(self):
        """
        Cleans up resources managed by the orchestrator's services.
        This should be called during application shutdown.
        """
        logger.info("Cleaning up RAG Orchestrator resources...")
        if self.vector_store_service:
            self.vector_store_service.cleanup()
        logger.info("RAG Orchestrator cleanup completed.")
