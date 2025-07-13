import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, List, Optional
from app.core.logging import get_logger
from langchain_core.output_parsers import JsonOutputParser
from app.core.prompt import PLANNER_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT, SUMMARIZER_PROMPT
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
import timezone

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

    def _format_tools_for_prompt(self, tools: List[Any]) -> str:
        """
        Formats a list of tools into a string for the planner's system prompt.
        This includes the header and usage examples.
        """
        tool_descriptions = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_name = tool.get("name", "Unknown Tool")
                tool_description = tool.get("description", "No description available.")
                tool_examples = tool.get("examples", [])
                tool_info = f"- **{tool_name}**: {tool_description}"
                if tool_examples:
                    tool_info += f" (Examples: {', '.join(tool_examples)})"
                tool_descriptions.append(tool_info)
            elif isinstance(tool, str):
                tool_descriptions.append(f"- **{tool}**: No specific description or examples available.")
            else:
                tool_descriptions.append(f"- **{type(tool).__name__}**: No specific description or examples available.")

        header = (
            "TOOLS AVAILABLE:\n"
            "Here's a list of tools you can use to generate your response. "
            "Each tool has a name, a description, and examples if applicable.\n\n"
        )
        return header + "\n\n".join(tool_descriptions)

    def get_planner_chain(self, tools: List[Any]) -> 'Runnable':
        """
        Creates a chain for the Planner node for ASYNC execution.
        """
        llm_with_tools = self.llm.bind_tools(tools)

        # Re-introduce temporal context and tool formatting
        try:
            now_casa = datetime.now(ZoneInfo("Africa/Casablanca"))
            time_context = now_casa.strftime("%A, %d %B %Y, %H:%M (Africa/Casablanca)")
            temporal_sentence = f"Current date and time: {time_context}."
        except Exception:
            now_utc = datetime.now(timezone.utc)
            temporal_sentence = f"Current date and time (UTC): {now_utc.strftime('%Y-%m-%d %H:%M')} UTC."

        tool_info = self._format_tools_for_prompt(tools)

        # Create and escape the full system prompt
        system_prompt_raw = (
            f"{PLANNER_SYSTEM_PROMPT}\n\n"
            f"{temporal_sentence}\n\n"
            f"Here are the tools you have access to:\n{tool_info}"
        )
        system_prompt_with_tools = system_prompt_raw.replace("{", "{{").replace("}", "}}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_with_tools),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        # This chain will be awaited with .ainvoke()
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
    


