from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from typing import List, Optional
import logging
from app.core.prompt import QNA_TEMPLATE_RAG
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
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 prompt_template: Optional[str] = None):
        """
        Initialize GenerationService
        
        Args:
            model_name: LLM model name
            temperature: Model temperature (0.0 to 1.0)
            prompt_template: Custom prompt template
        """
        # Fail Fast validation
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")
        
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
        # Default prompt template if none provided
        self.prompt_template = prompt_template or self._get_default_prompt()
        
        logger.info(f"GenerationService initialized with model: {model_name}")
    
    def get_agent_runnable(self, tools: List) -> 'Runnable':
        """
        Creates and returns a LangChain runnable that binds the LLM with 
        the system prompt and the necessary tools for the agent.
        """
        # Bind the tools to the LLM
        llm_with_tools = self.llm.bind_tools(tools)
        
        # Adapt the RAG prompt to a system prompt for the agent
        system_prompt_str = self.prompt_template.replace(
            "{context}", 
            "Use the documents from the 'knowledge_base_retriever' tool to answer the user's question."
        ).replace("{question}", "")

        # Create the agent's prompt structure
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_str),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        # Return the complete runnable chain
        return agent_prompt | llm_with_tools
    
    def _get_default_prompt(self) -> str:
        """Get default RAG prompt template"""
        return PROMPT_TELECOM_RAG
    

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
            # Create prompt template
            prompt = ChatPromptTemplate.from_template(self.prompt_template)
            
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
            "model_name": self.model_name,
            "temperature": self.temperature,
            "prompt_template_length": len(self.prompt_template)
        }
    
   
    
    
    


