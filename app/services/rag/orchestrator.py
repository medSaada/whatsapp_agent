from app.services.rag.generation_service import GenerationService
from app.services.rag.vector_store_service import VectorStoreService, VectorStoreConfig
from app.services.rag.graph.builder import GraphBuilder
from app.services.rag.graph.tools import create_rag_tool
from app.core.logging import get_logger
from app.core.config import Settings
from langchain_core.messages import HumanMessage
from langgraph.graph.graph import CompiledGraph
from typing import Dict, Any
import uuid
from pathlib import Path
import asyncio
import functools

logger = get_logger()

class RAGOrchestrator:
    """
    Main orchestrator for RAG operations using LangGraph.
    
    This class integrates all RAG components (vector store, generation, tools, and graph)
    into a unified service that maintains conversation state and generates responses.
    """
    
    def __init__(self,
                 settings: Settings,
                 vector_store_path: str = "data/vector_store",
                 collection_name: str = "production_collection",
                 model_name: str = "gpt-4.1",
                 temperature: float = 0.2,
                 memory_threshold: int = 6):
        
        self.settings = settings
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self.model_name = model_name
        self.temperature = temperature
        self.memory_threshold = memory_threshold
        
        self._ready = False
        self._graph: CompiledGraph = None
        self._setup_services()
        self._ready = True
        
        logger.info(f"RAGOrchestrator initialized with model: {model_name}, temp: {temperature}")
    
    def _setup_services(self):
        """Initialize all services and build the graph"""
        
        vector_config = VectorStoreConfig(
            store_path=self.vector_store_path,
            collection_name=self.collection_name
        )
        self.vector_store_service = VectorStoreService(vector_config)
        
        self.generation_service = GenerationService(
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        tools = [create_rag_tool(self.vector_store_service, self.collection_name)]
        
        graph_builder = GraphBuilder(
            generation_service=self.generation_service,
            tools=tools,
            memory_threshold=self.memory_threshold,
            settings=self.settings
        )
        
        db_path = "data/sqlite/conversations.db"
        self._graph = graph_builder.build(db_path)
        
        logger.info("RAG orchestrator setup completed successfully")
    
    def is_ready(self) -> bool:
        """Check if the orchestrator is ready to process requests"""
        return self._ready and self._graph is not None
    
    def generate_response(self, user_message: str, conversation_id: str = None) -> str:
        """
        Generate a response using the LangGraph workflow.
        
        Args:
            user_message: The user's input message
            conversation_id: Optional conversation ID for maintaining state
            
        Returns:
            Generated response as a string
        """
        if not self.is_ready():
            raise RuntimeError("RAGOrchestrator is not ready. Call setup() first.")
        
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        config = {"configurable": {"thread_id": conversation_id}}
        
        input_message = HumanMessage(content=user_message)
        initial_state = {
            "messages": [input_message],
            "context": "",
            "interaction_count": 0
        }
        
        final_state = self._graph.invoke(initial_state, config=config)
        
        final_response = final_state['messages'][-1].content
        
        logger.info(f"[Conversation: {conversation_id}] Response generated successfully")
        
        interaction_count = final_state.get('interaction_count', 0)
        logger.info(f"Current interaction count: {interaction_count}")
        
        tool_used = any(
            hasattr(msg, 'tool_calls') and msg.tool_calls 
            for msg in final_state['messages'] 
            if hasattr(msg, 'tool_calls')
        )
        
        if tool_used:
            logger.info(f"[Conversation: {conversation_id}] RAG tool was used for this response")
        else:
            logger.info(f"[Conversation: {conversation_id}] Response generated without RAG tool")
            
        return final_response
    
    async def generate_response_async(self, user_message: str, conversation_id: str = None) -> str:
        """Async wrapper for generate_response"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            functools.partial(self.generate_response, user_message, conversation_id)
        )
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'vector_store_service') and self.vector_store_service:
            self.vector_store_service.cleanup()
        logger.info("RAGOrchestrator cleanup completed")
