from typing import TypedDict, Annotated, List, Optional
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
        database_schema: Stores the Notion database schema once retrieved to avoid redundant calls.
        rag_context: Stores the context retrieved from the knowledge base tool.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    context: str # This will be deprecated but kept for now to avoid breaking changes.
    rag_context: str
    interaction_count: Annotated[int, operator.add]
    database_schema: Optional[dict]

def should_continue(state: AgentState) -> str:
    """
    Determines the next step for the agent.

    If the last message is a tool call, it routes to the tool executor.
    Otherwise, it ends the conversation.
    """
    if "tool_calls" in state["messages"][-1].additional_kwargs:
        return "call_tool"
    return "end" 