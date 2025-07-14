import json
from datetime import datetime, timezone
from typing import Any, List, Optional
from app.core.logging import get_logger
from langchain_core.output_parsers import JsonOutputParser
from app.core.prompt import PLANNER_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT, SUMMARIZER_PROMPT
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
    
    def __init__(self, llm: 'ChatOpenAI'):
        """
        Initialize GenerationService
        
        Args:
            llm: An initialized LangChain ChatOpenAI instance.
        """
        if not llm:
            raise ValueError("llm instance cannot be None")
        
        self.llm = llm
        logger.info(f"GenerationService initialized with model: {self.llm.model_name}")

    def get_planner_chain(self, tools: List[Any]) -> 'Runnable':
        """
        Creates a chain for the Planner node for ASYNC execution.
        The LLM is bound to the tools, and the system prompt is simplified.
        """
        llm_with_tools = self.llm.bind_tools(tools)

        try:
            # Always use UTC for consistency
            now_utc = datetime.now(timezone.utc)
            time_context = now_utc.strftime("%A, %d %B %Y, %H:%M %Z")
            temporal_sentence = f"Current date and time: {time_context}."
        except Exception:
            # Fallback in case of any issue
            now_utc = datetime.now(timezone.utc)
            temporal_sentence = f"Current date and time (UTC): {now_utc.strftime('%Y-%m-%d %H:%M')} UTC."

        # Format tools into a JSON-like string for the prompt, as per the notebook example.
        tool_descriptions = [tool.model_dump_json(include=["name", "description"]) for tool in tools]
        tools_info_str = "\n".join(tool_descriptions)

        # The system prompt now includes tool descriptions, along with the standard binding.
        system_prompt_content = (
            f"{PLANNER_SYSTEM_PROMPT}\n\n"
            f"{temporal_sentence}\n\n"
            "You have access to the following tools. Use them if necessary to answer the user's request or to add a event to the calendar or to add a prospect to the database.\n\n"
            "<tools>\n"
            f"{tools_info_str}\n"
            "</tools>"
        )
        
        # By creating a SystemMessage directly, we avoid template parsing issues with JSON braces.
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt_content),
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
    


