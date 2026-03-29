from datetime import datetime
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages

from sidekick_tools import other_tools, playwright_tools
from state import State

load_dotenv(override=True)

WORKER_MODEL = os.getenv("WORKER_MODEL")

class Worker:

    def __init__(self):
        self.worker_llm_with_tools = None

    async def setup(self, tools: List[Any]):
        self.worker_llm_with_tools = ChatOpenAI(model=WORKER_MODEL).bind_tools(tools=tools)

    def worker(self, state: State) -> Dict[str, Any]:
        system_message = f"""You are a helpful assistant that can use tools to complete tasks.
You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
You have many tools to help you, including tools to browse the internet, navigating and retrieving web pages.
You have a tool to run python code, but note that you would need to include a print() statement if you wanted to receive output.
The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This is the success criteria:
{state["success_criteria"]}

You should reply either with a question for the user about this assignment, or with your final response.
If you have a question for the user, you need to reply by clearly stating your question. An example might be:

Question: please clarify whether you want a summary or a detailed answer

If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
"""
        
        if state["feedback_on_work"]:
            system_message += f"""
Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
Here is the feedback on why this was rejected:
{state["feedback_on_work"]}
With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""
            
        system_message_found = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                system_message_found = True

        if not system_message_found:
            messages = [SystemMessage(content=system_message)] + messages

        response = self.worker_llm_with_tools.invoke(input=messages)

        return {"messages": [response]}
    
    def worker_route(self, state: State) -> str:
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"