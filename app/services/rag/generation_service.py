from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from typing import List, Optional
import logging
from app.core.prompt import (
    PERSONA_PROMPT, 
    RAG_DATA_TEMPLATE, 
    AGENT_TOOL_INSTRUCTION
)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, BasePromptTemplate
logger = logging.getLogger(__name__)

"""
Responsibilities:
LLM model management
Response generation
Answer quality control

"""

class GenerationService:
    """Service for generating responses using LLM with RAG context"""
    
    def __init__(self, 
                 llm: ChatOpenAI,
                 persona_prompt: Optional[str] = None):
        """
        Initialize GenerationService
        
        Args:
            llm: An initialized LangChain ChatOpenAI instance.
            persona_prompt: A string defining the persona of the AI.
        """
        if not llm:
            raise ValueError("llm instance cannot be None")
        
        self.llm = llm
        
        # Default persona prompt if none provided
        self.persona_prompt = persona_prompt or self._get_default_persona_prompt()
        
        logger.info(f"GenerationService initialized with model: {self.llm.model_name}")
    
    def get_agent_runnable(self, tools: List) -> 'Runnable':
        """
        Creates and returns a LangChain runnable that binds the LLM with 
        the system prompt and the necessary tools for the agent.
        """
        llm_with_tools = self.llm.bind_tools(tools)
        
        # Compose the system prompt from the persona and the agent instructions.
        system_prompt_str = f"{self.persona_prompt}\n\n{AGENT_TOOL_INSTRUCTION}"

        # Create the agent's prompt structure
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_str),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        # Return the complete runnable chain
        return agent_prompt | llm_with_tools
    
    def _get_default_persona_prompt(self) -> str:
        """Get default RAG prompt template"""
        return PERSONA_PROMPT
    

    def generate_response(self, 
                         question: str, 
                         context: str,
                         max_tokens: Optional[int] = None) -> str:
        """
        Generate response using RAG
        
        Args:
            question: User question
            context: Retrieved context
            max_tokens: Maximum tokens for response
            
        Returns:
            Generated answer
        """
        # Fail Fast validation
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        
        try:
            # Compose the full RAG prompt from the persona and the data template.
            full_rag_prompt_str = f"{self.persona_prompt}\n\n{RAG_DATA_TEMPLATE}"
            prompt = ChatPromptTemplate.from_template(full_rag_prompt_str)
            
            # Format prompt
            formatted_prompt = prompt.format(
                context=context.strip(),
                question=question.strip()
            )
            
            # Generate response
            response = self.llm.invoke(formatted_prompt)
            
            logger.info(f"Generated response for question: '{question[:50]}...'")
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Désolé, j'ai rencontré une erreur: {str(e)}"
    
    def generate_response_with_documents(self, 
                                       question: str, 
                                       documents: List[Document],
                                       max_tokens: Optional[int] = None) -> str:
        """
        Generate response from list of documents
        
        Args:
            question: User question
            documents: List of retrieved documents
            max_tokens: Maximum tokens for response
            
        Returns:
            Generated answer
        """
        if not documents:
            return "Je ne trouve pas d'information pertinente pour répondre à votre question."
        
        # Combine documents into context
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}: {doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        return self.generate_response(question, context, max_tokens)
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.llm.model_name,
            "temperature": self.llm.temperature,
            "persona_prompt_length": len(self.persona_prompt)
        }
    
   
    
    
    


