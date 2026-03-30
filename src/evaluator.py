import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .state import State


load_dotenv(override=True)

EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL")

class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(description="True is more information is needed from the user, or clarification or the assistant is stuck")

class Evaluator:
    def __init__(self):
        self.evaluator_llm_with_output = None

    def setup(self):
        self.evaluator_llm_with_output = ChatOpenAI(model=EVALUATOR_MODEL).with_structured_output(schema=EvaluatorOutput)

    def format_conversation(self, messages: List[Any]) -> str:
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            if isinstance(message, AIMessage):
                text = message.content or "[Tool use]"
                conversation += f"Assistant: {text}\n"

        return conversation

    def evaluator(self, state: State) -> Dict[str, Any]:
        system_message = """You are an evaluator that determines if a task has been completed successfully by an Assistant.
Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
and whether more input is needed from the user."""

        user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

The entire conversation with the assistant, with the user's original request and all replies, is:
{self.format_conversation(state["messages"])}

The success criteria for this assignment is:
{state["success_criteria"]}

And the final response from the Assistant that you are evaluating is:
{state["messages"][-1].content}Respond with your feedback, and decide if the success criteria is met by this response.
Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.

The Assistant has access to a tool to write files. If the Assistant says they have written a file, then you can assume they have done so.
Overall you should give the Assistant the benefit of the doubt if they say they've done something. But you should reject if you feel that more work should go into this."""
        
        if state["feedback_on_work"]:
            user_message += f"\n\nAlso, note that in a prior attempt from the Assistant, you provided this feedback: {state["feedback_on_work"]}\n"
            user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."

        messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]

        evaluator_output = self.evaluator_llm_with_output.invoke(input=messages)

        new_state = {
            "messages": [{"role": "assistant", "content": f"Evaluator Feedback on this answer: {evaluator_output.feedback}"}],
            "feedback_on_work": evaluator_output.feedback,
            "success_criteria_met": evaluator_output.success_criteria_met,
            "user_input_needed": evaluator_output.user_input_needed
        }

        return new_state
    
    def route_based_on_evaluation(self, state: State) -> str:
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        else:
            return "worker"