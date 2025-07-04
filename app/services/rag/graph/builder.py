from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import sqlite3
from pathlib import Path
from langchain_core.tools import BaseTool
from typing import List
from langchain_core.messages import ToolMessage, AIMessage, SystemMessage
import logging
from langsmith import tracing_context,Client
from app.services.rag.generation_service import GenerationService
from app.services.rag.graph.state import AgentState
from langgraph.checkpoint.sqlite import SqliteSaver
from app.core.config import Settings

logger = logging.getLogger(__name__)


def should_continue(state: AgentState):
    """
    Determines the next step for the agent.

    If the last message is from the planner and contains tool calls, route to the tool executor.
    If the last message is a ToolMessage, it means the tool has run, so route to the generator.
    Otherwise, end the conversation.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info(f"Tool call detected. Routing to tool executor.")
        return "call_tool"
    else:
        logger.info(f"No tool call detected. Routing to generator.")
        return "generate"


class GraphBuilder:
    """
    Builds the stateful LangGraph agent with separate planner and generator nodes.
    """

    def __init__(self, generation_service: GenerationService, tools: List[BaseTool], memory_threshold: int = 6, settings: Settings = None):
        self.generation_service = generation_service
        self.tools = tools
        self.memory_threshold = memory_threshold
        self.settings = settings

    def _check_memory_threshold(self, state: AgentState) -> AgentState:
        """
        Check if memory threshold is reached and handle summarization if needed.
        Returns the updated state.
        """
        current_count = state.get("interaction_count")
        logger.info(f"Hona counts ::☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️☝🏿️: {current_count}")
        new_count = current_count + 1
        
        logger.info(f"[Memory Management] Interaction count: {current_count} -> {new_count} (threshold: {self.memory_threshold})")
        
        # Check if we need to summarize and wipe memory
        if current_count >= self.memory_threshold and len(state["messages"]) > 2:  # More than just system messages
            logger.info(f"[Memory Management] 🧹 THRESHOLD REACHED! Summarizing and wiping conversation history...")
            logger.info(f"[Memory Management] Messages to summarize: {len(state['messages'])}")
            
            try:
                # Get conversation history for summarization
                conversation_messages = [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]
                
                if conversation_messages:
                    logger.info(f"[Memory Management] Found {len(conversation_messages)} conversation messages to summarize")
                    
                    # Create a text representation of the conversation
                    conversation_text = "\n".join([
                        f"{'User' if hasattr(msg, 'type') and msg.type == 'human' else 'Assistant'}: {msg.content}"
                        for msg in conversation_messages
                    ])
                    
                    # Generate summary
                    summarizer_chain = self.generation_service.get_summarizer_chain()
                    summary = summarizer_chain.invoke({"history": conversation_text})
                    
                    logger.info(f"[Memory Management] 📝 SUMMARY CREATED: {summary}")
                    logger.info(f"[Memory Management] ✅ Memory wiped and reset. Starting fresh with summary.")
                    
                    # Create new state with summary as system message and reset messages
                    summary_message = SystemMessage(content=f"Previous conversation summary: {summary}")
                    
                    return {
                        "messages": [summary_message],
                        "context": state.get("context", ""),
                        "interaction_count": 1  # Reset counter
                    }
                else:
                    logger.warning("[Memory Management] No conversation messages to summarize")
                    return {
                        "messages": state["messages"],
                        "context": state.get("context", ""),
                        "interaction_count": 1
                    }
                    
            except Exception as e:
                logger.error(f"[Memory Management] Error during summarization: {e}", exc_info=True)
                # If summarization fails, just reset the counter and continue
                return {
                    "messages": state["messages"],
                    "context": state.get("context", ""),
                    "interaction_count": 1
                }
        else:
            # No threshold reached, just increment counter
            if current_count == 0:
                logger.info(f"[Memory Management] First interaction. Counter: {new_count}")
            else:
                logger.info(f"[Memory Management] Continuing conversation. Counter: {new_count}/{self.memory_threshold}")
            
            return {
                "messages": state["messages"],
                "context": state.get("context", ""),
                "interaction_count": 1
            }

    def _planner_node(self, state: AgentState):
        """The 'brain' of the agent. Decides the next action."""
        # Ensure state has all required keys
        if "context" not in state:
            state["context"] = ""
            
        # Check memory threshold before processing
        updated_state = self._check_memory_threshold(state)
        
        planner_chain = self.generation_service.get_planner_chain(self.tools)
        response = planner_chain.invoke({"messages": updated_state['messages']})
        return {"messages": [response], "interaction_count": 1}

 
    def _generator_node(self, state: AgentState):
        """The 'voice' of the agent. Generates the final response."""
        # The context is now the content of the last message (the tool output)
        context = state["messages"][-1].content
        #logger.info(f"Generator received context: {context[:200]}...")

        generator_chain = self.generation_service.get_generator_chain()
        response = generator_chain.invoke({
            "messages": state['messages'],
            "context": context
        })
        return {"messages": [response]}

    def build(self, db_path: str):
        """
        Builds and compiles the graph with a checkpointer for persistence.
        """
        workflow = StateGraph(AgentState)

        # Add the nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("call_tool", ToolNode(self.tools))
        workflow.add_node("generator", self._generator_node)

        # Define the entry point and the conditional routing
        workflow.set_entry_point("planner")

        workflow.add_conditional_edges(
            "planner",
            should_continue,
            {
                "call_tool": "call_tool",
                "generate": "generator",
            }
        )
        
        workflow.add_edge("call_tool", "generator")
        workflow.add_edge("generator", END)

        # Set up the checkpointer for persistent memory
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        memory = SqliteSaver(conn=conn)

        # Compile the graph, connecting it to the checkpointer
        return workflow.compile(checkpointer=memory) 