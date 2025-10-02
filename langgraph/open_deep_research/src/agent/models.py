"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

###################
# Structured Outputs
###################


class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )


class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


class ResearchQueries(BaseModel):
    """List of concrete queries to execute for research."""

    queries: list[str] = Field(
        description="A list of diverse, specific web queries that together comprehensively cover the research brief.",
    )


###################
# State Definitions
###################


def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


class AgentInputState(MessagesState):
    """InputState is only 'messages'."""


class AgentState(MessagesState):
    """Main agent state containing messages and research data."""

    maximum_clarification_attempts: int
    clarification_attempts: int
    research_brief: Optional[str]
    queries: Annotated[list[str], override_reducer]
    notes: Annotated[list[str], override_reducer]
    final_report: str
