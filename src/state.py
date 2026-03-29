from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[str, add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool