from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

"""
Responsibilities:
LLM model management
Prompt engineering
Response generation
Answer quality control

"""

class GenerationService:
    def __init__(self, model_name: str, temperature: float = 0.2,prompt_template:str =''):
        """
        Initialize the GenerationService with the given model name, temperature, and prompt template that is a string like "You are **Fatima-Zahra**, a **client support expert** at **Geniats**, an **e-learning coding academy** for **Moroccan kids aged 6–15",
     "Your job is to **respond to client messages in Moroccan Darija** (or French, indeed the same language used by the client), with the goal of **convincing them to join and purchase our offers",
     "Behave like a respectful professional sales girl that her ultimate goal is to convert the lead into a client targeting the lead pain points",
     "Keep answers short and concise, and always end with a question to keep the conversation going",
      "### Context:",
    "{context}",
    "",
    "### Question:",
    "{question}",
    "",
    "### Answer:".
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        self.prompt_template = prompt_template

    def generate_response_staff_chain(self,question: str,context: str) -> str:
        qna_prompt =PromptTemplate(template=self.prompt_template,input_variables=['context','question'])
        staff_chain=load_qa_chain(llm=self.llm,chain_type="stuff",prompt=qna_prompt)
        answer=staff_chain({
                    "input_documents": context,
                    "question":question,
                            }
                            )
        return answer['answer']
    
   
    
    
    


