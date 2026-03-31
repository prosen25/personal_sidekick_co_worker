import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call
from typing import Any, Dict

# Add the parent directory to sys.path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator import Evaluator, EvaluatorOutput
from src.state import State
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage


class TestEvaluatorOutput:
    """Tests for EvaluatorOutput Pydantic model"""

    def test_evaluator_output_creation(self):
        """Test creating EvaluatorOutput instance"""
        output = EvaluatorOutput(
            feedback="Task completed successfully",
            success_criteria_met=True,
            user_input_needed=False
        )
        
        assert output.feedback == "Task completed successfully"
        assert output.success_criteria_met is True
        assert output.user_input_needed is False

    def test_evaluator_output_all_fields(self):
        """Test all fields are present in EvaluatorOutput"""
        output = EvaluatorOutput(
            feedback="Incomplete response",
            success_criteria_met=False,
            user_input_needed=True
        )
        
        assert hasattr(output, "feedback")
        assert hasattr(output, "success_criteria_met")
        assert hasattr(output, "user_input_needed")

    def test_evaluator_output_field_types(self):
        """Test field types are correct"""
        output = EvaluatorOutput(
            feedback="Test",
            success_criteria_met=False,
            user_input_needed=False
        )
        
        assert isinstance(output.feedback, str)
        assert isinstance(output.success_criteria_met, bool)
        assert isinstance(output.user_input_needed, bool)

    def test_evaluator_output_with_long_feedback(self):
        """Test EvaluatorOutput with long feedback string"""
        long_feedback = "x" * 1000
        output = EvaluatorOutput(
            feedback=long_feedback,
            success_criteria_met=True,
            user_input_needed=False
        )
        
        assert output.feedback == long_feedback


class TestEvaluatorInit:
    """Tests for Evaluator initialization"""

    def test_evaluator_init(self):
        """Test Evaluator initialization"""
        evaluator = Evaluator()
        assert evaluator.evaluator_llm_with_output is None

    def test_evaluator_init_creates_instance(self):
        """Test that Evaluator creates a valid instance"""
        evaluator = Evaluator()
        assert isinstance(evaluator, Evaluator)
        assert hasattr(evaluator, "evaluator_llm_with_output")
        assert hasattr(evaluator, "setup")
        assert hasattr(evaluator, "format_conversation")
        assert hasattr(evaluator, "evaluator")
        assert hasattr(evaluator, "route_based_on_evaluation")


class TestEvaluatorSetup:
    """Tests for Evaluator.setup method"""

    @pytest.mark.asyncio
    @patch("src.evaluator.ChatOpenAI")
    async def test_setup(self, mock_chat_openai):
        """Test setup method"""
        mock_llm = MagicMock()
        mock_llm_with_output = MagicMock()
        mock_llm.with_structured_output.return_value = mock_llm_with_output
        mock_chat_openai.return_value = mock_llm
        
        evaluator = Evaluator()
        await evaluator.setup()
        
        mock_chat_openai.assert_called_once()
        mock_llm.with_structured_output.assert_called_once_with(schema=EvaluatorOutput)
        assert evaluator.evaluator_llm_with_output == mock_llm_with_output

    @pytest.mark.asyncio
    @patch("src.evaluator.ChatOpenAI")
    async def test_setup_creates_chat_openai(self, mock_chat_openai):
        """Test that setup creates ChatOpenAI instance"""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        evaluator = Evaluator()
        await evaluator.setup()
        
        mock_chat_openai.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.evaluator.ChatOpenAI")
    async def test_setup_multiple_times(self, mock_chat_openai):
        """Test setup can be called multiple times"""
        mock_llm = MagicMock()
        mock_llm_with_output = MagicMock()
        mock_llm.with_structured_output.return_value = mock_llm_with_output
        mock_chat_openai.return_value = mock_llm
        
        evaluator = Evaluator()
        await evaluator.setup()
        await evaluator.setup()
        
        # Should be able to call setup multiple times
        assert mock_chat_openai.call_count == 2


class TestFormatConversation:
    """Tests for Evaluator.format_conversation method"""

    def test_format_conversation_empty(self):
        """Test format_conversation with empty messages"""
        evaluator = Evaluator()
        result = evaluator.format_conversation([])
        
        assert "Conversation history:" in result
        assert isinstance(result, str)

    def test_format_conversation_single_human_message(self):
        """Test format_conversation with single human message"""
        evaluator = Evaluator()
        message = HumanMessage(content="What is Python?")
        
        result = evaluator.format_conversation([message])
        
        assert "User: What is Python?" in result
        assert "Conversation history:" in result

    def test_format_conversation_single_ai_message(self):
        """Test format_conversation with single AI message"""
        evaluator = Evaluator()
        message = AIMessage(content="Python is a programming language")
        
        result = evaluator.format_conversation([message])
        
        assert "Assistant: Python is a programming language" in result
        assert "Conversation history:" in result

    def test_format_conversation_mixed_messages(self):
        """Test format_conversation with mixed messages"""
        evaluator = Evaluator()
        messages = [
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language"),
            HumanMessage(content="Why use Python?"),
            AIMessage(content="Python is easy to learn")
        ]
        
        result = evaluator.format_conversation(messages)
        
        assert "User: What is Python?" in result
        assert "Assistant: Python is a programming language" in result
        assert "User: Why use Python?" in result
        assert "Assistant: Python is easy to learn" in result

    def test_format_conversation_ai_message_with_empty_content(self):
        """Test format_conversation with AI message having empty content"""
        evaluator = Evaluator()
        message = AIMessage(content="")
        
        result = evaluator.format_conversation([message])
        
        # Empty content should still be included
        assert "Assistant: " in result

    def test_format_conversation_multiple_messages_order(self):
        """Test that format_conversation preserves message order"""
        evaluator = Evaluator()
        messages = [
            HumanMessage(content="First"),
            AIMessage(content="Second"),
            HumanMessage(content="Third")
        ]
        
        result = evaluator.format_conversation(messages)
        
        first_idx = result.find("User: First")
        second_idx = result.find("Assistant: Second")
        third_idx = result.find("User: Third")
        
        assert first_idx < second_idx < third_idx


class TestEvaluatorMethod:
    """Tests for Evaluator.evaluator method"""

    def _create_state(
        self,
        messages=None,
        success_criteria="Complete the task",
        feedback_on_work=None,
        success_criteria_met=False,
        user_input_needed=False
    ):
        """Helper to create a State object"""
        if messages is None:
            messages = [AIMessage(content="Task result")]
        
        return {
            "messages": messages,
            "success_criteria": success_criteria,
            "feedback_on_work": feedback_on_work,
            "success_criteria_met": success_criteria_met,
            "user_input_needed": user_input_needed,
        }

    @patch("src.evaluator.ChatOpenAI")
    def test_evaluator_basic(self, mock_chat_openai):
        """Test basic evaluator invocation"""
        mock_llm = MagicMock()
        evaluator_output = EvaluatorOutput(
            feedback="Good response",
            success_criteria_met=True,
            user_input_needed=False
        )
        mock_llm.invoke.return_value = evaluator_output
        
        evaluator = Evaluator()
        evaluator.evaluator_llm_with_output = mock_llm
        
        state = self._create_state(messages=[AIMessage(content="Result")])
        result = evaluator.evaluator(state)
        
        assert "messages" in result
        assert "feedback_on_work" in result
        assert "success_criteria_met" in result
        assert "user_input_needed" in result

    @patch("src.evaluator.ChatOpenAI")
    def test_evaluator_returns_feedback(self, mock_chat_openai):
        """Test that evaluator returns feedback"""
        mock_llm = MagicMock()
        feedback_text = "Response is incomplete"
        evaluator_output = EvaluatorOutput(
            feedback=feedback_text,
            success_criteria_met=False,
            user_input_needed=True
        )
        mock_llm.invoke.return_value = evaluator_output
        
        evaluator = Evaluator()
        evaluator.evaluator_llm_with_output = mock_llm
        
        state = self._create_state()
        result = evaluator.evaluator(state)
        
        assert result["feedback_on_work"] == feedback_text

    @patch("src.evaluator.ChatOpenAI")
    def test_evaluator_with_success_criteria(self, mock_chat_openai):
        """Test evaluator with different success criteria"""
        mock_llm = MagicMock()
        evaluator_output = EvaluatorOutput(
            feedback="Task met criteria",
            success_criteria_met=True,
            user_input_needed=False
        )
        mock_llm.invoke.return_value = evaluator_output
        
        evaluator = Evaluator()
        evaluator.evaluator_llm_with_output = mock_llm
        
        success_criteria = "Write a Python function that adds two numbers"
        state = self._create_state(success_criteria=success_criteria)
        result = evaluator.evaluator(state)
        
        assert result["success_criteria_met"] is True

    @patch("src.evaluator.ChatOpenAI")
    def test_evaluator_with_feedback_on_work(self, mock_chat_openai):
        """Test evaluator when there's prior feedback"""
        mock_llm = MagicMock()
        evaluator_output = EvaluatorOutput(
            feedback="Better response",
            success_criteria_met=True,
            user_input_needed=False
        )
        mock_llm.invoke.return_value = evaluator_output
        
        evaluator = Evaluator()
        evaluator.evaluator_llm_with_output = mock_llm
        
        prior_feedback = "Previous attempt missed requirements"
        state = self._create_state(feedback_on_work=prior_feedback)
        result = evaluator.evaluator(state)
        
        # Verify invoke was called
        mock_llm.invoke.assert_called_once()

    @patch("src.evaluator.ChatOpenAI")
    def test_evaluator_creates_messages(self, mock_chat_openai):
        """Test that evaluator creates proper messages for LLM"""
        mock_llm = MagicMock()
        evaluator_output = EvaluatorOutput(
            feedback="Good",
            success_criteria_met=True,
            user_input_needed=False
        )
        
        # Capture the messages passed to invoke
        invoked_messages = []
        def invoke_side_effect(**kwargs):
            invoked_messages.extend(kwargs.get("input", []))
            return evaluator_output
        
        mock_llm.invoke.side_effect = invoke_side_effect
        
        evaluator = Evaluator()
        evaluator.evaluator_llm_with_output = mock_llm
        
        state = self._create_state()
        result = evaluator.evaluator(state)
        
        # Verify messages were passed
        mock_llm.invoke.assert_called_once()

    @patch("src.evaluator.ChatOpenAI")
    def test_evaluator_with_multiple_messages(self, mock_chat_openai):
        """Test evaluator with multiple conversation messages"""
        mock_llm = MagicMock()
        evaluator_output = EvaluatorOutput(
            feedback="Response good",
            success_criteria_met=True,
            user_input_needed=False
        )
        mock_llm.invoke.return_value = evaluator_output
        
        evaluator = Evaluator()
        evaluator.evaluator_llm_with_output = mock_llm
        
        messages = [
            HumanMessage(content="Task: Find info"),
            AIMessage(content="Found information"),
            HumanMessage(content="More details?"),
            AIMessage(content="Here are more details")
        ]
        state = self._create_state(messages=messages)
        result = evaluator.evaluator(state)
        
        assert result["success_criteria_met"] is True

    @patch("src.evaluator.ChatOpenAI")
    def test_evaluator_result_structure(self, mock_chat_openai):
        """Test that evaluator returns correct result structure"""
        mock_llm = MagicMock()
        evaluator_output = EvaluatorOutput(
            feedback="Test feedback",
            success_criteria_met=True,
            user_input_needed=False
        )
        mock_llm.invoke.return_value = evaluator_output
        
        evaluator = Evaluator()
        evaluator.evaluator_llm_with_output = mock_llm
        
        state = self._create_state()
        result = evaluator.evaluator(state)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "messages" in result
        assert "feedback_on_work" in result
        assert "success_criteria_met" in result
        assert "user_input_needed" in result
        assert isinstance(result["messages"], list)


class TestRouteBasedOnEvaluation:
    """Tests for Evaluator.route_based_on_evaluation method"""

    def _create_state(
        self,
        success_criteria_met=False,
        user_input_needed=False,
        success_criteria="Task"
    ):
        """Helper to create state"""
        return {
            "messages": [],
            "success_criteria": success_criteria,
            "feedback_on_work": None,
            "success_criteria_met": success_criteria_met,
            "user_input_needed": user_input_needed,
        }

    def test_route_when_success_criteria_met(self):
        """Test routing returns 'END' when success criteria met"""
        evaluator = Evaluator()
        state = self._create_state(success_criteria_met=True)
        
        result = evaluator.route_based_on_evaluation(state)
        
        assert result == "END"

    def test_route_when_user_input_needed(self):
        """Test routing returns 'END' when user input needed"""
        evaluator = Evaluator()
        state = self._create_state(user_input_needed=True)
        
        result = evaluator.route_based_on_evaluation(state)
        
        assert result == "END"

    def test_route_when_neither_criteria_met_nor_input_needed(self):
        """Test routing returns 'worker' when criteria not met and no input needed"""
        evaluator = Evaluator()
        state = self._create_state(success_criteria_met=False, user_input_needed=False)
        
        result = evaluator.route_based_on_evaluation(state)
        
        assert result == "worker"

    def test_route_when_both_success_and_input_needed(self):
        """Test routing with both success criteria met and input needed"""
        evaluator = Evaluator()
        state = self._create_state(success_criteria_met=True, user_input_needed=True)
        
        result = evaluator.route_based_on_evaluation(state)
        
        # Should return END because success_criteria is truthy
        assert result == "END"

    def test_route_with_empty_success_criteria(self):
        """Test routing with empty success criteria string"""
        evaluator = Evaluator()
        state = self._create_state(
            success_criteria="",
            success_criteria_met=False,
            user_input_needed=False
        )
        
        result = evaluator.route_based_on_evaluation(state)
        
        # Should return worker because success_criteria_met is False
        assert result == "worker"

    def test_route_logic_checks_success_criteria_met(self):
        """Test routing checks success_criteria_met field"""
        evaluator = Evaluator()
        
        # When success_criteria_met is True, should return END
        state = self._create_state(
            success_criteria="Some criteria",
            success_criteria_met=True,
            user_input_needed=False
        )
        result = evaluator.route_based_on_evaluation(state)
        assert result == "END"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
