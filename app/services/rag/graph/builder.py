from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import sqlite3
from pathlib import Path
from langchain_core.tools import BaseTool
from typing import List
from langchain_core.messages import ToolMessage, AIMessage, SystemMessage
from app.core.logging import get_logger
from app.services.rag.generation_service import GenerationService
from app.services.rag.graph.state import AgentState
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.core.config import Settings
import asyncio
import json

logger = get_logger()


# This function can remain synchronous as it's pure logic
def should_continue(state: AgentState):
    """
    Determines the next step for the agent.
    - If the last message is an AIMessage with tool_calls, route to tool execution.
    - If the last message is a ToolMessage, the agent needs to re-evaluate the plan, so route back to the planner.
    - Otherwise, the plan is complete, so route to the generator.
    """
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info(f"Tool call detected. Routing to tool executor.")
        return "call_tool"
    
    # If the last message is a tool result, the agent should decide what to do next.
    if isinstance(last_message, ToolMessage):
        logger.info(f"Tool execution completed. Routing back to planner for re-evaluation.")
        return "planner"

    # If there are no tool calls and it's not a tool message, the agent is done planning.
    logger.info(f"No more tool calls. Routing to generator.")
    return "generate"


class GraphBuilder:
    """
    Builds the stateful LangGraph agent with separate planner and generator nodes.
    Now builds an ASYNCHRONOUS graph.
    """

    def __init__(self, generation_service: GenerationService, tools: List, settings: Settings = None):
        self.generation_service = generation_service
        self.tools = tools  # No need for sync wrappers anymore
        self.settings = settings
    
    async def _planner_node(self, state: AgentState):
        """The 'brain' of the agent. Decides the next action asynchronously."""
        # Increment interaction counter
        current_count = state.get("interaction_count", 0)
        updates = {"interaction_count": current_count + 1}
        logger.info(f"[State Management] Interaction count updated to {updates['interaction_count']}")

        # Check the current state for the database schema
        if state.get("database_schema"):
            schema_status = "Already retrieved and available in state."
        else:
            schema_status = "Not yet retrieved."

        planner_chain = self.generation_service.get_planner_chain(self.tools)
        response = await planner_chain.ainvoke({
            "messages": state['messages'],
            "database_schema_status": schema_status
        })

        # The final patch includes the new AIMessage and updated interaction count
        updates["messages"] = [response]
        
        return updates

    async def _custom_tool_node(self, state: AgentState):
        """
        A custom tool node that intercepts the output of `notion_retrieve_database`
        and saves it to the agent's state. It ensures all tool results are returned.
        """
        tool_node = ToolNode(self.tools)
        tool_updates = await tool_node.ainvoke(state)
        
        # Check if the notion_retrieve_database tool was called and save the schema
        last_ai_message = state["messages"][-1]
        if isinstance(last_ai_message, AIMessage) and last_ai_message.tool_calls:
            for tool_call in last_ai_message.tool_calls:
                if tool_call['name'] == 'notion_retrieve_database':
                    # Find the corresponding ToolMessage in the updates
                    for tool_message in tool_updates.get("messages", []):
                        if isinstance(tool_message, ToolMessage) and tool_message.tool_call_id == tool_call['id']:
                            try:
                                schema = json.loads(tool_message.content)
                                tool_updates['database_schema'] = schema
                                logger.info(f"Successfully retrieved and saved Notion database schema to state.")
                            except (json.JSONDecodeError, TypeError) as e:
                                logger.error(f"Failed to parse or save Notion schema: {e}")
                            break
        
        # Return the complete set of updates from the tool executions
        return tool_updates

 
    async def _generator_node(self, state: AgentState):
        """The 'voice' of the agent. Generates the final response asynchronously."""
        # The context is now the content of the last message (the tool output)
        context = state["messages"][-1].content
        #logger.info(f"Generator received context: {context[:200]}...")

        generator_chain = self.generation_service.get_generator_chain()
        response = await generator_chain.ainvoke({
            "messages": state['messages'],
            "context": context
        })
        return {"messages": [response]}

    async def build(self, checkpointer: AsyncSqliteSaver = None):
        """
        Builds and compiles the ASYNC graph with a checkpointer for persistence.
        """
        workflow = StateGraph(AgentState)

        # Add the async nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("call_tool", self._custom_tool_node) # Use the custom tool node
        workflow.add_node("generator", self._generator_node)

        # Define the entry point and the conditional routing
        workflow.set_entry_point("planner")

        workflow.add_conditional_edges(
            "planner",
            should_continue,
            {
                "call_tool": "call_tool",
                "generate": "generator",
                "planner": "planner", # Add self-loop for re-evaluation
            }
        )
        
        # After a tool is called, we must route back to the planner to decide the next step.
        workflow.add_edge("call_tool", "planner")
        workflow.add_edge("generator", END)

        # Compile the async graph, with the checkpointer if provided
        return workflow.compile(checkpointer=checkpointer)


# visualize graph
if __name__ == "__main__":
    from IPython.display import display, Image
    from app.services.rag.generation_service import GenerationService
    
    # Building without a checkpointer for visualization purposes
    graph_builder = GraphBuilder(generation_service=GenerationService(), tools=[])
    runnable = asyncio.run(graph_builder.build())
    display(Image(runnable.get_graph().draw_mermaid_png()))