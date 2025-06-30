from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from typing import List, Optional
import logging
from langchain_core.output_parsers import JsonOutputParser
from app.core.prompt import PLANNER_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT
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

    def get_agent_runnable(self, tools: List) -> 'Runnable':
        """
        Creates and returns a LangChain runnable that binds the LLM with 
        the system prompt and the necessary tools for the agent.
        """
        llm_with_tools = self.llm.bind_tools(tools)
        
        # Create the agent's prompt structure
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        # Return the complete runnable chain
        return agent_prompt | llm_with_tools

    def get_routing_chain(self):
        """
        Creates a chain specifically for routing the user's query.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_PROMPT),
            ("human", "{question}")
        ])
        
        return prompt | self.llm | JsonOutputParser()

    def get_generation_chain(self):
        """
        Creates a chain that generates the final response, potentially with context.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.system_prompt}\n\n{RAG_DATA_TEMPLATE}"),
            # MessagesPlaceholder(variable_name="messages")
        ])
        return prompt | self.llm

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.llm.model_name,
            "temperature": self.llm.temperature,
        }
    
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
            full_rag_prompt_str = f"{self.system_prompt}\n\n{RAG_DATA_TEMPLATE}"
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
    
   
    
    
    


