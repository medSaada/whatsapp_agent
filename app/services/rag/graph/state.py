from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Represents the state of our graph.
    This structure is passed between nodes and updated at each step.
    It holds the history of messages in the conversation.
    """
    messages: Annotated[List[BaseMessage], operator.add]

def should_continue(state: AgentState) -> str:
    """
    Determines the next step in the graph.

    If the last message in the state contains tool calls, it directs the graph
    to execute the 'call_tool' node. Otherwise, it signifies that the agent's
    turn is complete, and the graph should 'end'.
    """
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        # The agent has decided to use a tool
        return "call_tool"
    else:
        # The agent has responded directly and the turn is over
        return "end" 