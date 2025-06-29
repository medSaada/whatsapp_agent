from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import logging
from pathlib import Path

from app.services.rag.vector_store_service import VectorStoreService, VectorStoreConfig
from app.services.rag.generation_service import GenerationService
from app.core.prompt import  PERSONA_PROMPT
from app.services.rag.graph.builder import GraphBuilder
from app.services.rag.graph.tools import create_rag_tool

logger = logging.getLogger(__name__)

# Use the same prompt template from test_generation.py


class RAGOrchestrator:
    """
    High-level orchestrator for the RAG agent.

    It initializes the necessary services and builds the agent graph on-demand
    once the required data collection is available.
    """
    
    def __init__(self, 
                 vector_store_path: str = "data/vector_store",
                 collection_name: str = "production_collection",
                 model_name: str = "gpt-4.1",
                 temperature: float = 0.2,
                 db_path: str = "data/sqlite/conversation_memory.db",
                 persona_prompt: Optional[str] = None):
        """
        Initialize the RAG Orchestrator services.
        The agent graph is not built here, but on-demand.
        """
        self.collection_name = collection_name
        self.db_path = db_path
        self._graph = None # The graph will be built lazily
        
        # 1. Initialize core services
        prompt = persona_prompt if persona_prompt else PERSONA_PROMPT
        llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.vector_store_service = self._init_vector_store(vector_store_path, collection_name)
        self.generation_service = self._init_generation_service(llm, prompt)
        
        logger.info(f"RAG Orchestrator initialized for collection: {collection_name}")
    
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
        builder = GraphBuilder(self.generation_service, [rag_tool])
        self._graph = builder.build(self.db_path)
        logger.info("LangGraph agent built successfully.")
        
        return self._graph

    def _init_vector_store(self, store_path: str, collection_name: str) -> VectorStoreService:
        config = VectorStoreConfig(store_path=store_path, collection_name=collection_name)
        return VectorStoreService(config)

    def _init_generation_service(
        self, llm: ChatOpenAI, persona_prompt: str
    ) -> GenerationService:
        return GenerationService(
            llm=llm,
            persona_prompt=persona_prompt,
        )

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
            input_data = {"messages": [HumanMessage(content=question)]}
            
            final_response = None
            logger.info(f"--- Starting new RAG flow for conversation '{conversation_id}' ---")
            
            # Use the graph to stream events and find the final agent response
            for event in graph.stream(input_data, config=config):
                # Log every event type and its content for deep analysis
                # logger.debug(f"Event: {event}")

                if "call_tool" in event:
                    tool_call = event["call_tool"]
                    tool_name = tool_call.get('name')
                    tool_input = tool_call.get('args')
                    logger.info(f"[Tool Call] Agent called '{tool_name}' with input: {tool_input}")
                
                # Check for messages, especially tool outputs
                if "messages" in event:
                    last_message = event["messages"][-1]
                    if last_message.type == "tool":
                        logger.info(f"[Tool Output] Retrieved {len(last_message.content)} documents.")
                        # To see the actual content, you can uncomment the next line, but it can be very verbose.
                        # logger.debug(f"Retrieved content: {last_message.content}")

                if "agent" in event:
                    if messages := event["agent"].get("messages"):
                        last_message = messages[-1]
                        if not last_message.tool_calls and last_message.type == 'ai':
                            final_response = last_message.content
                            # The agent has made its final decision
                            logger.info(f"[Final Answer] Agent generated final response.")
            
            if final_response:
                logger.info(f"--- RAG flow finished for conversation '{conversation_id}' ---")
                return final_response
            else:
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
