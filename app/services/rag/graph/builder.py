from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import sqlite3
from pathlib import Path
from langchain_core.tools import BaseTool
from typing import List
from langchain_core.messages import ToolMessage, AIMessage
import logging

from app.services.rag.generation_service import GenerationService
from app.services.rag.graph.state import AgentState
from langgraph.checkpoint.sqlite import SqliteSaver

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
        return "call_tool"
    if isinstance(last_message, ToolMessage):
        return "generate"
    return "end"


class GraphBuilder:
    """
    Builds the stateful LangGraph agent with separate planner and generator nodes.
    """

    def __init__(self, generation_service: GenerationService, tools: List[BaseTool]):
        self.generation_service = generation_service
        self.tools = tools

    def _planner_node(self, state: AgentState):
        """The 'brain' of the agent. Decides the next action."""
        planner_chain = self.generation_service.get_planner_chain(self.tools)
        response = planner_chain.invoke({"messages": state['messages']})
        return {"messages": [response]}

    def _generator_node(self, state: AgentState):
        """The 'voice' of the agent. Generates the final response."""
        # The context is now the content of the last message (the tool output)
        context = state["messages"][-1].content
        logger.info(f"Generator received context: {context[:200]}...")

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
                "end": END,
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