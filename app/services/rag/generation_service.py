import json
from datetime import datetime, timezone
from typing import Any, List, Optional
from app.core.logging import get_logger
from langchain_core.output_parsers import JsonOutputParser
from app.core.prompt import PLANNER_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage

logger = get_logger()

"""
Responsibilities:
LLM model management
Response generation
Answer quality control

"""

class GenerationService:
    """Service for generating responses using LLM with RAG context"""
    
    def __init__(self, planner_llm: 'ChatOpenAI', generator_llm: 'ChatOpenAI'):
        """
        Initialize GenerationService with separate models for planning and generation.
        
        Args:
            planner_llm: An initialized LangChain ChatOpenAI instance for the planner.
            generator_llm: An initialized LangChain ChatOpenAI instance for the generator.
        """
        if not planner_llm or not generator_llm:
            raise ValueError("Both planner_llm and generator_llm instances cannot be None")
        
        self.planner_llm = planner_llm
        self.generator_llm = generator_llm
        logger.info(f"GenerationService initialized with Planner: {self.planner_llm.model_name} and Generator: {self.generator_llm.model_name}")

    def get_planner_chain(self, tools: List[Any]) -> 'Runnable':
        """
        Creates a chain for the Planner node for ASYNC execution.
        The planner LLM is bound to the tools.
        """
        llm_with_tools = self.planner_llm.bind_tools(tools)

        try:
            # Always use UTC for consistency
            now_utc = datetime.now(timezone.utc)
            time_context = now_utc.strftime("%A, %d %B %Y, %H:%M %Z")
            temporal_sentence = f"Current date and time: {time_context}."
        except Exception:
            # Fallback in case of any issue
            now_utc = datetime.now(timezone.utc)
            temporal_sentence = f"Current date and time (UTC): {now_utc.strftime('%Y-%m-%d %H:%M')} UTC."

        # The system prompt is now cleaner, letting bind_tools handle the heavy lifting.
        system_prompt_content = (
            f"{PLANNER_SYSTEM_PROMPT}\n\n"
            "# Current State Information\n"
            "You have access to the following state to help you make your decision. Use it to avoid repeating actions.\n"
            "- Database Schema: {database_schema_status}\n\n"
            f"{temporal_sentence}\n"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt_content),
            MessagesPlaceholder(variable_name="messages")
        ])

        # We must update the input variables to include our new state key
        prompt.input_variables.append("database_schema_status")

        return prompt | llm_with_tools

    def get_generator_chain(self) -> 'Runnable':
        """
        Creates a chain for the Generator node.
        This chain crafts the final user-facing response using the generator LLM.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", GENERATOR_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages")
        ])
        return prompt | self.generator_llm

    def get_model_info(self) -> dict:
        """Get model information for both planner and generator"""
        return {
            "planner_model": {
                "model_name": self.planner_llm.model_name,
                "temperature": self.planner_llm.temperature,
            },
            "generator_model": {
                "model_name": self.generator_llm.model_name,
                "temperature": self.generator_llm.temperature,
            }
        }
    


