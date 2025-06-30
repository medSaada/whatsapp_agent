from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from typing import List, Optional
import logging
from langchain_core.output_parsers import JsonOutputParser
from app.core.prompt import PLANNER_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT, SUMMARIZER_PROMPT
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
logger = logging.getLogger(__name__)

"""
Responsibilities:
LLM model management
Response generation
Answer quality control

"""

class GenerationService:
    """Service for generating responses using LLM with RAG context"""
    
    def __init__(self, llm: ChatOpenAI):
        """
        Initialize GenerationService
        
        Args:
            llm: An initialized LangChain ChatOpenAI instance.
        """
        if not llm:
            raise ValueError("llm instance cannot be None")
        
        self.llm = llm
        logger.info(f"GenerationService initialized with model: {self.llm.model_name}")

    def get_planner_chain(self, tools: List) -> 'Runnable':
        """
        Creates a chain for the Planner node.
        This chain decides the next action but does not generate the final response.
        """
        llm_with_tools = self.llm.bind_tools(tools)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        return prompt | llm_with_tools

    def get_generator_chain(self) -> 'Runnable':
        """
        Creates a chain for the Generator node.
        This chain crafts the final user-facing response.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", GENERATOR_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages")
        ])
        return prompt | self.llm

    def get_summarizer_chain(self) -> 'Runnable':
        """
        Creates a chain for summarizing conversation history.
        """
        prompt = ChatPromptTemplate.from_template(SUMMARIZER_PROMPT)
        return prompt | self.llm

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.llm.model_name,
            "temperature": self.llm.temperature,
        }
    


