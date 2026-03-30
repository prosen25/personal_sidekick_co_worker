from datetime import datetime
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages

from .sidekick_tools import other_tools, playwright_tools
from .state import State

load_dotenv(override=True)

WORKER_MODEL = os.getenv("WORKER_MODEL", "gpt-4o-preview")

class Worker:

    def __init__(self):
        self.worker_llm_with_tools = None

    async def setup(self, tools: List[Any]):
        if not tools:
            raise ValueError("At least one tool is required for worker setup")

        self.worker_llm_with_tools = ChatOpenAI(model=WORKER_MODEL).bind_tools(tools=tools)

    def worker(self, state: State) -> Dict[str, Any]:
        if self.worker_llm_with_tools is None:
            raise RuntimeError("Worker is not initialized. Call setup(tools=...) before invoking worker().")

        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary-like object")

        if "success_criteria" not in state or "messages" not in state:
            raise ValueError("state must include 'success_criteria' and 'messages'")

        system_message_text = f"""You are a helpful assistant that can use tools to complete tasks.
You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
You have many tools to help you, including tools to browse the internet, navigating and retrieving web pages.
You have a tool to run python code, but note that you would need to include a print() statement if you wanted to receive output.
The current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This is the success criteria:
{state['success_criteria']}

You should reply either with a question for the user about this assignment, or with your final response.
If you have a question for the user, you need to reply by clearly stating your question. An example might be:

Question: please clarify whether you want a summary or a detailed answer

If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
"""

        if state.get("feedback_on_work"):
            system_message_text += f"""
Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
Here is the feedback on why this was rejected:
{state['feedback_on_work']}
With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""

        existing_messages = list(state["messages"])
        system_message_found = False

        for message in existing_messages:
            if isinstance(message, SystemMessage):
                message.content = system_message_text
                system_message_found = True

        if not system_message_found:
            existing_messages = [SystemMessage(content=system_message_text)] + existing_messages

        response = self.worker_llm_with_tools.invoke(input=existing_messages)

        if response is None:
            raise RuntimeError("Worker LLM returned no response")

        return {"messages": [response]}

    def worker_route(self, state: State) -> str:
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary-like object")

        messages = state.get("messages", [])

        if not messages:
            return "evaluator"

        last_message = messages[-1]

        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "evaluator"