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
    
    def __init__(self, 
                 settings: Settings,  
                 vector_store_path: str = "data/vector_store",
                 collection_name: str = "production_collection",
                 model_name: str = "gpt-4.1",
                 temperature: float = 0.2,
                 db_path: str = "data/sqlite/conversation_memory.db",
                 memory_threshold: int = 6):
        """
        Initialize the RAG Orchestrator services.
        The agent graph is not built here, but on-demand.
        """
        self.settings = settings
        self.collection_name = collection_name
        self.db_path = db_path
        self._graph = None # The graph will be built lazily
        self.memory_threshold = memory_threshold
        
        # 1. Initialize core services with settings
        llm = ChatOpenAI(
            model=model_name, 
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY
        )
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=settings.OPENAI_API_KEY
        )
        
        self.vector_store_service = self._init_vector_store(vector_store_path, collection_name, embeddings)
        self.generation_service = self._init_generation_service(llm)
        
        logger.info(f"RAG Orchestrator initialized for collection: {collection_name}")
        logger.info(f"[Memory Management] Memory threshold set to: {self.memory_threshold} interactions")
    
    def _get_or_build_graph(self):
        """
        Builds the LangGraph agent on-demand.
        
        If the graph is already built, it returns the cached instance.
        If not, it checks if the required collection exists and then builds it.
        Returns None if the system is not ready.
        """
        if self._graph:
            return self._graph

        if not self.is_ready():
            logger.warning(f"Cannot build graph: Collection '{self.collection_name}' not found or is empty.")
            return None
        
        logger.info("Building LangGraph agent...")
        rag_tool = create_rag_tool(self.vector_store_service, self.collection_name)
        builder = GraphBuilder(
            generation_service=self.generation_service, 
            tools=[rag_tool], 
            memory_threshold=self.memory_threshold,
            settings=self.settings  # Pass settings to GraphBuilder
        )
        self._graph = builder.build(self.db_path)
        logger.info("LangGraph agent built successfully.")
        
        return self._graph

    def _init_vector_store(self, store_path: str, collection_name: str, embeddings: OpenAIEmbeddings) -> VectorStoreService:
        config = VectorStoreConfig(store_path=store_path, collection_name=collection_name)
        return VectorStoreService(config, embedding_model=embeddings)

    def _init_generation_service(
        self, llm: ChatOpenAI
    ) -> GenerationService:
        return GenerationService(llm=llm)

    def answer_question(self, question: str, conversation_id: str) -> str:
        """
        Answer a question using the LangGraph agent, with detailed logging.
        The graph is built on the first call if it hasn't been already.
        """
        graph = self._get_or_build_graph()
        
        if not graph:
            logger.warning("RAG system not ready, using fallback response.")
            return "Je ne suis pas encore prêt à répondre aux questions. Veuillez réessayer dans un instant."

        try:
            config = {"configurable": {"thread_id": conversation_id}}
            input_data = {
                "messages": [HumanMessage(content=question)],
                "context": "",
                "interaction_count": 0
            }
            
            logger.info(f"--- Starting new RAG flow for conversation '{conversation_id}' ---")
            logger.info(f"[Memory Management] Processing interaction for conversation: {conversation_id}")
            
            # Invoke the graph and wait for the final state. This is more robust
            # than streaming and trying to interpret intermediate steps.
            final_state = graph.invoke(input_data, config=config)
            
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
            logger.error(f"Error answering question for conversation '{conversation_id}': {e}", exc_info=True)
            return f"Désolé, j'ai rencontré une erreur: {str(e)}"
    
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
