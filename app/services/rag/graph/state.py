from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Represents the state of our agent.

    Attributes:
        messages: The history of messages in the conversation.
                  The `operator.add` anntation tells LangGraph to append new messages
                  to this list rather than overwriting it.
        context: The retrieved context from the vector store.
        interaction_count: Counter for tracking the number of interactions for memory management.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    context: str
    interaction_count: int

def should_continue(state: AgentState) -> str:
    """
    Determines the next step for the agent.

    If the last message is a tool call, it routes to the tool executor.
    Otherwise, it ends the conversation.
    """
    if "tool_calls" in state["messages"][-1].additional_kwargs:
        return "call_tool"
    return "end" 