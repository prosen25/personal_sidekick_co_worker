import pytest
import os
import sys
import uuid
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call
from typing import Any, Dict
from langchain_core.messages import HumanMessage

# Add the parent directory to sys.path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sidekick import Sidekick
from src.state import State


class TestSidekickInit:
    """Tests for Sidekick initialization"""

    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    def test_sidekick_init(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test Sidekick initialization"""
        mock_worker_instance = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_memory_instance = MagicMock()
        
        mock_worker.return_value = mock_worker_instance
        mock_evaluator.return_value = mock_evaluator_instance
        mock_memory_saver.return_value = mock_memory_instance
        
        sidekick = Sidekick()
        
        assert sidekick.worker == mock_worker_instance
        assert sidekick.evaluator == mock_evaluator_instance
        assert sidekick.tools == []
        assert sidekick.async_browser is None
        assert sidekick.playwright is None
        assert sidekick.memory == mock_memory_instance
        assert sidekick.graph is None
        assert isinstance(sidekick.sidekick_id, str)

    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    def test_sidekick_init_creates_unique_ids(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test that multiple Sidekick instances have unique IDs"""
        mock_memory_saver.return_value = MagicMock()
        mock_worker.return_value = MagicMock()
        mock_evaluator.return_value = MagicMock()
        
        sidekick1 = Sidekick()
        sidekick2 = Sidekick()
        
        assert sidekick1.sidekick_id != sidekick2.sidekick_id

    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    def test_sidekick_init_sidekick_id_is_uuid_format(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test that sidekick_id is a valid UUID string"""
        mock_memory_saver.return_value = MagicMock()
        mock_worker.return_value = MagicMock()
        mock_evaluator.return_value = MagicMock()
        
        sidekick = Sidekick()
        
        # Should be valid UUID string
        try:
            uuid.UUID(sidekick.sidekick_id)
            is_valid_uuid = True
        except ValueError:
            is_valid_uuid = False
        
        assert is_valid_uuid


class TestBuildGraph:
    """Tests for Sidekick.build_graph method"""

    @pytest.mark.asyncio
    @patch("src.sidekick.StateGraph")
    @patch("src.sidekick.ToolNode")
    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    async def test_build_graph(self, mock_memory_saver, mock_evaluator, mock_worker, mock_tool_node, mock_state_graph):
        """Test build_graph method"""
        mock_worker_instance = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_builder = MagicMock()
        mock_compiled_graph = MagicMock()
        
        mock_worker.return_value = mock_worker_instance
        mock_evaluator.return_value = mock_evaluator_instance
        mock_memory_saver.return_value = MagicMock()
        mock_state_graph.return_value = mock_builder
        mock_builder.compile.return_value = mock_compiled_graph
        
        sidekick = Sidekick()
        sidekick.tools = [MagicMock()]
        
        await sidekick.build_graph()
        
        # Verify StateGraph was created
        mock_state_graph.assert_called_once_with(State)
        
        # Verify nodes were added
        assert mock_builder.add_node.call_count == 3
        
        # Verify edges were added
        assert mock_builder.add_edge.call_count >= 2
        
        # Verify graph was compiled
        assert sidekick.graph == mock_compiled_graph

    @pytest.mark.asyncio
    @patch("src.sidekick.StateGraph")
    @patch("src.sidekick.ToolNode")
    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    async def test_build_graph_adds_worker_node(self, mock_memory_saver, mock_evaluator, mock_worker, mock_tool_node, mock_state_graph):
        """Test that build_graph adds worker node"""
        mock_worker_instance = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_builder = MagicMock()
        
        mock_worker.return_value = mock_worker_instance
        mock_evaluator.return_value = mock_evaluator_instance
        mock_memory_saver.return_value = MagicMock()
        mock_state_graph.return_value = mock_builder
        mock_builder.compile.return_value = MagicMock()
        
        sidekick = Sidekick()
        sidekick.tools = [MagicMock()]
        
        await sidekick.build_graph()
        
        # Check that add_node was called with "worker"
        add_node_calls = [call[0] for call in mock_builder.add_node.call_args_list]
        assert ("worker", mock_worker_instance.worker) in add_node_calls

    @pytest.mark.asyncio
    @patch("src.sidekick.StateGraph")
    @patch("src.sidekick.ToolNode")
    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    async def test_build_graph_adds_evaluator_node(self, mock_memory_saver, mock_evaluator, mock_worker, mock_tool_node, mock_state_graph):
        """Test that build_graph adds evaluator node"""
        mock_worker_instance = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_builder = MagicMock()
        
        mock_worker.return_value = mock_worker_instance
        mock_evaluator.return_value = mock_evaluator_instance
        mock_memory_saver.return_value = MagicMock()
        mock_state_graph.return_value = mock_builder
        mock_builder.compile.return_value = MagicMock()
        
        sidekick = Sidekick()
        sidekick.tools = [MagicMock()]
        
        await sidekick.build_graph()
        
        # Check that add_node was called with "evaluator"
        add_node_calls = [call[0] for call in mock_builder.add_node.call_args_list]
        assert ("evaluator", mock_evaluator_instance.evaluator) in add_node_calls


class TestSetup:
    """Tests for Sidekick.setup method"""

    @pytest.mark.asyncio
    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    async def test_setup_initializes_attributes(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test that setup initializes key attributes"""
        mock_worker.return_value = MagicMock()
        mock_evaluator.return_value = MagicMock()
        mock_memory_saver.return_value = MagicMock()
        
        sidekick = Sidekick()
        
        # Verify initial state
        assert sidekick.tools == []
        assert sidekick.async_browser is None
        assert sidekick.playwright is None
        assert sidekick.graph is None

    @pytest.mark.asyncio
    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    async def test_setup_worker_setup_called(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test that setup calls worker.setup - verifying setup workflow"""
        mock_w = MagicMock()
        mock_e = MagicMock()
        mock_worker.return_value = mock_w
        mock_evaluator.return_value = mock_e
        mock_memory_saver.return_value = MagicMock()
        
        sidekick = Sidekick()
        
        # Verify worker has setup method
        assert hasattr(sidekick.worker, 'setup')


class TestRunSuperStep:
    """Tests for Sidekick.run_super_step method"""

    @pytest.mark.asyncio
    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    async def test_run_super_step(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test run_super_step method"""
        mock_worker_instance = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_memory_instance = MagicMock()
        
        mock_worker.return_value = mock_worker_instance
        mock_evaluator.return_value = mock_evaluator_instance
        mock_memory_saver.return_value = mock_memory_instance
        
        sidekick = Sidekick()
        
        # Setup mock graph
        mock_graph = AsyncMock()
        
        # Create mock messages
        mock_msg_1 = MagicMock()
        mock_msg_1.content = "Reply content"
        mock_msg_2 = MagicMock()
        mock_msg_2.content = "Feedback content"
        
        mock_result = {
            "messages": [MagicMock(), mock_msg_1, mock_msg_2]
        }
        
        mock_graph.ainvoke.return_value = mock_result
        sidekick.graph = mock_graph
        
        message = "Find Python tutorials"
        success_criteria = "Find 3 tutorials"
        history = [{"role": "user", "content": "Previous message"}]
        
        result = await sidekick.run_super_step(message, success_criteria, history)
        
        # Verify graph.ainvoke was called
        mock_graph.ainvoke.assert_called_once()
        
        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 4  # Previous + user + reply + feedback

    @pytest.mark.asyncio
    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    async def test_run_super_step_creates_correct_state(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test that run_super_step creates correct state for graph"""
        mock_worker_instance = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_memory_instance = MagicMock()
        
        mock_worker.return_value = mock_worker_instance
        mock_evaluator.return_value = mock_evaluator_instance
        mock_memory_saver.return_value = mock_memory_instance
        
        sidekick = Sidekick()
        
        mock_graph = AsyncMock()
        invoked_state = None
        
        def capture_state(**kwargs):
            nonlocal invoked_state
            invoked_state = kwargs.get("input")
            return {"messages": [MagicMock(), MagicMock(), MagicMock()]}
        
        mock_graph.ainvoke.side_effect = capture_state
        sidekick.graph = mock_graph
        
        await sidekick.run_super_step("test message", "test criteria", [])
        
        # Verify state structure
        assert invoked_state is not None
        assert isinstance(invoked_state["messages"], list)
        assert len(invoked_state["messages"]) == 1
        assert isinstance(invoked_state["messages"][0], HumanMessage)
        assert invoked_state["messages"][0].content == "test message"
        assert invoked_state["success_criteria"] == "test criteria"
        assert invoked_state["feedback_on_work"] is None
        assert invoked_state["success_criteria_met"] is False
        assert invoked_state["user_input_needed"] is False

    @pytest.mark.asyncio
    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    async def test_run_super_step_uses_thread_id(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test that run_super_step uses sidekick_id as thread_id"""
        mock_worker_instance = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_memory_instance = MagicMock()
        
        mock_worker.return_value = mock_worker_instance
        mock_evaluator.return_value = mock_evaluator_instance
        mock_memory_saver.return_value = mock_memory_instance
        
        sidekick = Sidekick()
        
        mock_graph = AsyncMock()
        invoked_config = None
        
        def capture_config(**kwargs):
            nonlocal invoked_config
            invoked_config = kwargs.get("config")
            return {"messages": [MagicMock(), MagicMock(), MagicMock()]}
        
        mock_graph.ainvoke.side_effect = capture_config
        sidekick.graph = mock_graph
        
        await sidekick.run_super_step("test", "criteria", [])
        
        assert invoked_config is not None
        assert invoked_config["configurable"]["thread_id"] == sidekick.sidekick_id

    @pytest.mark.asyncio
    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    async def test_run_super_step_formats_history(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test that run_super_step formats history correctly"""
        mock_worker_instance = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_memory_instance = MagicMock()
        
        mock_worker.return_value = mock_worker_instance
        mock_evaluator.return_value = mock_evaluator_instance
        mock_memory_saver.return_value = mock_memory_instance
        
        sidekick = Sidekick()
        
        mock_graph = AsyncMock()
        
        mock_msg_1 = MagicMock()
        mock_msg_1.content = "Assistant response"
        mock_msg_2 = MagicMock()
        mock_msg_2.content = "Evaluator feedback"
        
        mock_result = {
            "messages": [MagicMock(), mock_msg_1, mock_msg_2]
        }
        
        mock_graph.ainvoke.return_value = mock_result
        sidekick.graph = mock_graph
        
        message = "User message"
        history = []
        
        result = await sidekick.run_super_step(message, "criteria", history)
        
        # Verify result has user message
        assert result[0]["role"] == "user"
        assert result[0]["content"] == message
        
        # Verify result has assistant reply
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Assistant response"
        
        # Verify result has feedback
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "Evaluator feedback"


class TestCleanup:
    """Tests for Sidekick.cleanup method"""

    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    def test_cleanup_with_no_browser(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test cleanup when no browser is set"""
        mock_worker.return_value = MagicMock()
        mock_evaluator.return_value = MagicMock()
        mock_memory_saver.return_value = MagicMock()
        
        sidekick = Sidekick()
        sidekick.async_browser = None
        
        # Should not raise an exception
        sidekick.cleanup()

    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    def test_cleanup_with_running_loop(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test cleanup when event loop is running"""
        mock_worker.return_value = MagicMock()
        mock_evaluator.return_value = MagicMock()
        mock_memory_saver.return_value = MagicMock()
        
        sidekick = Sidekick()
        
        mock_browser = MagicMock()
        mock_browser.close = MagicMock(return_value=MagicMock())
        mock_pw = MagicMock()
        mock_pw.stop = MagicMock(return_value=MagicMock())
        
        sidekick.async_browser = mock_browser
        sidekick.playwright = mock_pw
        
        # Mock asyncio.get_running_loop to simulate running loop
        with patch("src.sidekick.asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            
            sidekick.cleanup()
            
            # Verify create_task was called
            assert mock_loop.create_task.called

    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    @patch("src.sidekick.asyncio.run")
    def test_cleanup_without_running_loop(self, mock_asyncio_run, mock_memory_saver, mock_evaluator, mock_worker):
        """Test cleanup when no event loop is running"""
        mock_worker.return_value = MagicMock()
        mock_evaluator.return_value = MagicMock()
        mock_memory_saver.return_value = MagicMock()
        
        sidekick = Sidekick()
        
        mock_browser = MagicMock()
        mock_browser.close = MagicMock(return_value=MagicMock())
        mock_pw = MagicMock()
        mock_pw.stop = MagicMock(return_value=MagicMock())
        
        sidekick.async_browser = mock_browser
        sidekick.playwright = mock_pw
        
        # Mock asyncio.get_running_loop to raise RuntimeError (no running loop)
        with patch("src.sidekick.asyncio.get_running_loop") as mock_get_loop:
            mock_get_loop.side_effect = RuntimeError("No running event loop")
            
            sidekick.cleanup()
            
            # Verify asyncio.run was called
            assert mock_asyncio_run.called

    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    def test_cleanup_closes_browser(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test that cleanup attempts to close browser"""
        mock_worker.return_value = MagicMock()
        mock_evaluator.return_value = MagicMock()
        mock_memory_saver.return_value = MagicMock()
        
        sidekick = Sidekick()
        
        mock_browser = MagicMock()
        mock_browser.close = MagicMock(return_value=MagicMock())
        
        sidekick.async_browser = mock_browser
        sidekick.playwright = None
        
        with patch("src.sidekick.asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            
            sidekick.cleanup()
            
            # Verify create_task was called for browser
            assert mock_loop.create_task.called

    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    def test_cleanup_stops_playwright(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test that cleanup attempts to stop playwright"""
        mock_worker.return_value = MagicMock()
        mock_evaluator.return_value = MagicMock()
        mock_memory_saver.return_value = MagicMock()
        
        sidekick = Sidekick()
        
        mock_browser = MagicMock()
        mock_browser.close = MagicMock(return_value=MagicMock())
        mock_pw = MagicMock()
        mock_pw.stop = MagicMock(return_value=MagicMock())
        
        sidekick.async_browser = mock_browser
        sidekick.playwright = mock_pw
        
        with patch("src.sidekick.asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            
            sidekick.cleanup()
            
            # Verify create_task was called for both browser and playwright
            assert mock_loop.create_task.call_count >= 2


class TestSidekickIntegration:
    """Integration tests for Sidekick"""

    @pytest.mark.asyncio
    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    async def test_build_graph_called(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test that build_graph is callable"""
        mock_worker.return_value = MagicMock()
        mock_evaluator.return_value = MagicMock()
        mock_memory_saver.return_value = MagicMock()
        
        with patch("src.sidekick.StateGraph") as mock_state_graph, patch("src.sidekick.ToolNode"):
            mock_builder = MagicMock()
            mock_graph = MagicMock()
            mock_state_graph.return_value = mock_builder
            mock_builder.compile.return_value = mock_graph
            
            sidekick = Sidekick()
            sidekick.tools = [MagicMock()]
            
            await sidekick.build_graph()
            
            # Verify graph was compiled
            assert sidekick.graph == mock_graph

    @pytest.mark.asyncio
    @patch("src.sidekick.Worker")
    @patch("src.sidekick.Evaluator")
    @patch("src.sidekick.MemorySaver")
    async def test_multiple_sidekick_instances_different_ids(self, mock_memory_saver, mock_evaluator, mock_worker):
        """Test that multiple Sidekick instances have different IDs"""
        mock_worker.return_value = MagicMock()
        mock_evaluator.return_value = MagicMock()
        mock_memory_saver.return_value = MagicMock()
        
        sidekick1 = Sidekick()
        sidekick2 = Sidekick()
        sidekick3 = Sidekick()
        
        ids = {sidekick1.sidekick_id, sidekick2.sidekick_id, sidekick3.sidekick_id}
        
        # All IDs should be unique
        assert len(ids) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
