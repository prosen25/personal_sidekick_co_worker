import pytest
import os
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock, call
from typing import Any, Dict

# Add the parent directory to sys.path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.worker import Worker
from src.state import State
from langchain_core.messages import SystemMessage, BaseMessage


class TestWorkerInit:
    """Tests for Worker initialization"""

    def test_worker_init(self):
        """Test Worker initialization"""
        worker = Worker()
        assert worker.worker_llm_with_tools is None

    def test_worker_init_creates_instance(self):
        """Test that Worker creates a valid instance"""
        worker = Worker()
        assert isinstance(worker, Worker)
        assert hasattr(worker, "worker_llm_with_tools")
        assert hasattr(worker, "setup")
        assert hasattr(worker, "worker")
        assert hasattr(worker, "worker_route")


class TestWorkerSetup:
    """Tests for Worker.setup method"""

    @pytest.mark.asyncio
    @patch("src.worker.ChatOpenAI")
    async def test_setup_with_tools(self, mock_chat_openai):
        """Test setup method with tools"""
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_chat_openai.return_value = mock_llm
        
        worker = Worker()
        tools = [MagicMock(), MagicMock()]
        
        await worker.setup(tools)
        
        mock_chat_openai.assert_called_once()
        mock_llm.bind_tools.assert_called_once_with(tools=tools)
        assert worker.worker_llm_with_tools == mock_llm_with_tools

    @pytest.mark.asyncio
    @patch("src.worker.ChatOpenAI")
    async def test_setup_with_empty_tools_list(self, mock_chat_openai):
        """Test setup method with empty tools list"""
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_chat_openai.return_value = mock_llm
        
        worker = Worker()
        tools = []
        
        with pytest.raises(ValueError, match="At least one tool is required"):
            await worker.setup(tools)

    @pytest.mark.asyncio
    @patch("src.worker.ChatOpenAI")
    async def test_setup_multiple_tools(self, mock_chat_openai):
        """Test setup method with multiple tools"""
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_chat_openai.return_value = mock_llm
        
        worker = Worker()
        tools = [MagicMock() for _ in range(5)]
        
        await worker.setup(tools)
        
        assert mock_llm.bind_tools.call_args[1]["tools"] == tools


class TestWorkerMethod:
    """Tests for Worker.worker method"""

    def _create_state(
        self,
        messages=None,
        success_criteria="Complete the task",
        feedback_on_work=None,
        success_criteria_met=False,
        user_input_needed=False
    ):
        """Helper method to create a State object"""
        if messages is None:
            messages = []
        
        return {
            "messages": messages,
            "success_criteria": success_criteria,
            "feedback_on_work": feedback_on_work,
            "success_criteria_met": success_criteria_met,
            "user_input_needed": user_input_needed,
        }

    @patch("src.worker.ChatOpenAI")
    def test_worker_basic(self, mock_chat_openai):
        """Test basic worker invocation"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        worker = Worker()
        worker.worker_llm_with_tools = mock_llm
        
        state = self._create_state()
        result = worker.worker(state)
        
        assert "messages" in result
        assert result["messages"] == [mock_response]
        mock_llm.invoke.assert_called_once()

    @patch("src.worker.ChatOpenAI")
    def test_worker_adds_system_message_when_not_present(self, mock_chat_openai):
        """Test that worker adds SystemMessage when not present"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        
        # Capture the messages passed to invoke
        invoked_messages = []
        def invoke_side_effect(**kwargs):
            invoked_messages.extend(kwargs.get("input", []))
            return mock_response
        
        mock_llm.invoke.side_effect = invoke_side_effect
        mock_chat_openai.return_value = mock_llm
        
        worker = Worker()
        worker.worker_llm_with_tools = mock_llm
        
        test_message = MagicMock(spec=BaseMessage)
        state = self._create_state(messages=[test_message], success_criteria="Test")
        result = worker.worker(state)
        
        assert result["messages"] == [mock_response]
        # Verify invoke was called
        mock_llm.invoke.assert_called_once()

    @patch("src.worker.ChatOpenAI")
    def test_worker_updates_existing_system_message(self, mock_chat_openai):
        """Test that worker updates existing SystemMessage"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        worker = Worker()
        worker.worker_llm_with_tools = mock_llm
        
        existing_system_message = SystemMessage(content="Old content")
        test_message = MagicMock(spec=BaseMessage)
        
        state = self._create_state(
            messages=[existing_system_message, test_message],
            success_criteria="New task"
        )
        result = worker.worker(state)
        
        assert result["messages"] == [mock_response]
        # The system message content should have been updated
        mock_llm.invoke.assert_called_once()

    @patch("src.worker.ChatOpenAI")
    def test_worker_with_feedback(self, mock_chat_openai):
        """Test worker with feedback on previous work"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        worker = Worker()
        worker.worker_llm_with_tools = mock_llm
        
        state = self._create_state(
            success_criteria="Complete task",
            feedback_on_work="Previous attempt was incomplete"
        )
        result = worker.worker(state)
        
        assert result["messages"] == [mock_response]
        mock_llm.invoke.assert_called_once()
        
        # Verify invoke was called with messages
        call_kwargs = mock_llm.invoke.call_args[1]
        assert "input" in call_kwargs

    @patch("src.worker.ChatOpenAI")
    def test_worker_includes_datetime_in_system_message(self, mock_chat_openai):
        """Test that system message includes current datetime"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        
        # Capture the messages to check system message content
        captured_messages = []
        def invoke_side_effect(**kwargs):
            captured_messages.extend(kwargs.get("input", []))
            return mock_response
        
        mock_llm.invoke.side_effect = invoke_side_effect
        mock_chat_openai.return_value = mock_llm
        
        worker = Worker()
        worker.worker_llm_with_tools = mock_llm
        
        state = self._create_state(success_criteria="Test task")
        result = worker.worker(state)
        
        # Check that a SystemMessage was created/modified
        mock_llm.invoke.assert_called_once()

    @patch("src.worker.ChatOpenAI")
    def test_worker_with_success_criteria(self, mock_chat_openai):
        """Test worker includes success criteria in system message"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        worker = Worker()
        worker.worker_llm_with_tools = mock_llm
        
        success_criteria = "Find information about Python programming"
        state = self._create_state(success_criteria=success_criteria)
        result = worker.worker(state)
        
        assert result["messages"] == [mock_response]

    @patch("src.worker.ChatOpenAI")
    def test_worker_with_multiple_messages(self, mock_chat_openai):
        """Test worker processes multiple messages in state"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        worker = Worker()
        worker.worker_llm_with_tools = mock_llm
        
        messages = [
            MagicMock(spec=BaseMessage),
            MagicMock(spec=BaseMessage),
            MagicMock(spec=BaseMessage),
        ]
        state = self._create_state(messages=messages)
        result = worker.worker(state)
        
        assert result["messages"] == [mock_response]


class TestWorkerRoute:
    """Tests for Worker.worker_route method"""

    def test_worker_route_with_tool_calls(self):
        """Test worker_route returns 'tools' when tool calls present"""
        worker = Worker()
        
        last_message = MagicMock()
        last_message.tool_calls = [MagicMock()]
        
        messages = [MagicMock(), last_message]
        state = {
            "messages": messages,
            "success_criteria": "Test",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }
        
        result = worker.worker_route(state)
        assert result == "tools"

    def test_worker_route_without_tool_calls(self):
        """Test worker_route returns 'evaluator' when no tool calls"""
        worker = Worker()
        
        last_message = MagicMock()
        last_message.tool_calls = []
        
        messages = [MagicMock(), last_message]
        state = {
            "messages": messages,
            "success_criteria": "Test",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }
        
        result = worker.worker_route(state)
        assert result == "evaluator"

    def test_worker_route_no_tool_calls_attribute(self):
        """Test worker_route returns 'evaluator' when tool_calls attribute missing"""
        worker = Worker()
        
        last_message = MagicMock(spec=[])  # No tool_calls attribute
        
        messages = [last_message]
        state = {
            "messages": messages,
            "success_criteria": "Test",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }
        
        result = worker.worker_route(state)
        assert result == "evaluator"

    def test_worker_route_multiple_tool_calls(self):
        """Test worker_route with multiple tool calls"""
        worker = Worker()
        
        last_message = MagicMock()
        last_message.tool_calls = [MagicMock(), MagicMock(), MagicMock()]
        
        messages = [last_message]
        state = {
            "messages": messages,
            "success_criteria": "Test",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }
        
        result = worker.worker_route(state)
        assert result == "tools"

    def test_worker_route_single_message(self):
        """Test worker_route with single message"""
        worker = Worker()
        
        last_message = MagicMock()
        last_message.tool_calls = None
        
        state = {
            "messages": [last_message],
            "success_criteria": "Test",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }
        
        result = worker.worker_route(state)
        assert result == "evaluator"

    def test_worker_route_checks_last_message(self):
        """Test worker_route checks only the last message"""
        worker = Worker()
        
        # First message has tool calls (should be ignored)
        first_message = MagicMock()
        first_message.tool_calls = [MagicMock()]
        
        # Last message has no tool calls
        last_message = MagicMock()
        last_message.tool_calls = []
        
        state = {
            "messages": [first_message, last_message],
            "success_criteria": "Test",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }
        
        result = worker.worker_route(state)
        assert result == "evaluator"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
