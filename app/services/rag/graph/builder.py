import sqlite3
from pathlib import Path
from typing import List

from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.services.rag.generation_service import GenerationService
from app.services.rag.graph.state import AgentState, should_continue


class GraphBuilder:
    """
    Builds the stateful LangGraph agent.

    This class is responsible for wiring together the nodes (agent, tools) and
    edges (conditional logic) of the graph. It takes the core services and
    tools as input and produces a compiled, persistent graph ready for use.
    """
    def __init__(self, generation_service: GenerationService, tools: List[BaseTool]):
        self.generation_service = generation_service
        self.tools = tools
        # The GraphBuilder's responsibility is now simplified.
        # It just gets the fully-formed agent chain from the GenerationService.
        self.agent_chain = self.generation_service.get_agent_runnable(self.tools)

    def _agent_node(self, state: AgentState):
        """The 'brain' of the agent. Invokes the agent chain with the current state."""
        response = self.agent_chain.invoke({"messages": state['messages']})
        return {"messages": [response]}

    def build(self, db_path: str):
        """
        Builds and compiles the graph with a checkpointer for persistence.

        The checkpointer (SQLite in this case) allows the agent to maintain
        memory of the conversation across multiple interactions.
        """
        workflow = StateGraph(AgentState)

        # Add the two main nodes: the agent and the tool executor
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("call_tool", ToolNode(self.tools))

        # Define the entry point and the conditional routing
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "call_tool": "call_tool",
                "end": END,
            }
        )
        # After a tool is called, the flow returns to the agent to process the result
        workflow.add_edge("call_tool", "agent")

        # Set up the checkpointer for persistent memory
        Path(db_path).parent.mkdir(parents=True, exist_ok=True) # a robust way to create the directory in all operating systems
        conn = sqlite3.connect(db_path, check_same_thread=False)
        memory = SqliteSaver(conn=conn)

        # Compile the graph, connecting it to the checkpointer
        return workflow.compile(checkpointer=memory) 