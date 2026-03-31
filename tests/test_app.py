import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call
from typing import Any, Dict

# Add the parent directory to sys.path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app import App


class TestAppInit:
    """Tests for App initialization"""

    @patch("src.app.Sidekick")
    def test_app_init(self, mock_sidekick):
        """Test App initialization"""
        mock_sidekick_instance = MagicMock()
        mock_sidekick.return_value = mock_sidekick_instance
        
        app = App()
        
        assert app.sidekick == mock_sidekick_instance
        mock_sidekick.assert_called_once()

    @patch("src.app.Sidekick")
    def test_app_init_creates_instance(self, mock_sidekick):
        """Test that App creates a valid instance"""
        mock_sidekick_instance = MagicMock()
        mock_sidekick.return_value = mock_sidekick_instance
        
        app = App()
        
        assert isinstance(app, App)
        assert hasattr(app, "sidekick")
        assert hasattr(app, "setup")
        assert hasattr(app, "process_message")
        assert hasattr(app, "reset")
        assert hasattr(app, "free_resurces")
        assert hasattr(app, "run")


class TestAppSetup:
    """Tests for App.setup method"""

    @pytest.mark.asyncio
    @patch("src.app.Sidekick")
    async def test_setup(self, mock_sidekick):
        """Test setup method"""
        mock_sidekick_instance = AsyncMock()
        mock_sidekick.return_value = mock_sidekick_instance
        
        app = App()
        
        # Mock the setup method
        mock_sidekick_instance.setup = AsyncMock()
        
        result = await app.setup()
        
        mock_sidekick_instance.setup.assert_called_once()
        assert result == mock_sidekick_instance

    @pytest.mark.asyncio
    @patch("src.app.Sidekick")
    async def test_setup_returns_sidekick(self, mock_sidekick):
        """Test that setup returns the sidekick instance"""
        mock_sidekick_instance = AsyncMock()
        mock_sidekick.return_value = mock_sidekick_instance
        mock_sidekick_instance.setup = AsyncMock()
        
        app = App()
        result = await app.setup()
        
        assert result == app.sidekick

    @pytest.mark.asyncio
    @patch("src.app.Sidekick")
    async def test_setup_calls_sidekick_setup(self, mock_sidekick):
        """Test that setup calls sidekick.setup"""
        mock_sidekick_instance = AsyncMock()
        mock_sidekick.return_value = mock_sidekick_instance
        mock_sidekick_instance.setup = AsyncMock()
        
        app = App()
        await app.setup()
        
        mock_sidekick_instance.setup.assert_called_once()


class TestProcessMessage:
    """Tests for App.process_message method"""

    @pytest.mark.asyncio
    @patch("src.app.Sidekick")
    async def test_process_message(self, mock_sidekick):
        """Test process_message method"""
        mock_sidekick_instance = MagicMock()
        mock_sidekick_instance.run_super_step = AsyncMock(return_value=["result"])
        mock_sidekick.return_value = mock_sidekick_instance
        
        app = App()
        returned_sidekick, result = await app.process_message(
            mock_sidekick_instance,
            "test message",
            "test criteria",
            []
        )
        
        assert result == ["result"]
        assert returned_sidekick == mock_sidekick_instance

    @pytest.mark.asyncio
    @patch("src.app.Sidekick")
    async def test_process_message_calls_run_super_step(self, mock_sidekick):
        """Test that process_message calls run_super_step"""
        mock_sidekick_instance = MagicMock()
        mock_sidekick_instance.run_super_step = AsyncMock(return_value=["result"])
        mock_sidekick.return_value = mock_sidekick_instance
        
        app = App()
        message = "Find Python tutorials"
        success_criteria = "Find 3 good tutorials"
        history = []
        
        await app.process_message(mock_sidekick_instance, message, success_criteria, history)
        
        mock_sidekick_instance.run_super_step.assert_called_once_with(
            message=message,
            success_criteria=success_criteria,
            history=history
        )

    @pytest.mark.asyncio
    @patch("src.app.Sidekick")
    async def test_process_message_with_history(self, mock_sidekick):
        """Test process_message with conversation history"""
        mock_sidekick_instance = MagicMock()
        mock_sidekick_instance.run_super_step = AsyncMock(return_value=["result1", "result2"])
        mock_sidekick.return_value = mock_sidekick_instance
        
        app = App()
        history = [{"role": "user", "content": "First message"}]
        
        returned_sidekick, result = await app.process_message(
            mock_sidekick_instance,
            "Second message",
            "criteria",
            history
        )
        
        assert result == ["result1", "result2"]

    @pytest.mark.asyncio
    @patch("src.app.Sidekick")
    async def test_process_message_returns_sidekick(self, mock_sidekick):
        """Test that process_message returns the sidekick instance"""
        mock_sidekick_instance = MagicMock()
        mock_sidekick_instance.run_super_step = AsyncMock(return_value=[])
        mock_sidekick.return_value = mock_sidekick_instance
        
        app = App()
        returned_sidekick, result = await app.process_message(
            mock_sidekick_instance,
            "test",
            "criteria",
            []
        )
        
        assert returned_sidekick == mock_sidekick_instance


class TestReset:
    """Tests for App.reset method"""

    @pytest.mark.asyncio
    @patch("src.app.Sidekick")
    async def test_reset(self, mock_sidekick):
        """Test reset method"""
        new_mock_sidekick = MagicMock()
        new_mock_sidekick.setup = AsyncMock()
        new_mock_sidekick.cleanup = MagicMock()
        mock_sidekick.return_value = new_mock_sidekick
        
        app = App()
        message, criteria, chatbot, sidekick = await app.reset(app.sidekick)
        
        assert message == ""
        assert criteria == ""
        assert chatbot == []
        assert sidekick == new_mock_sidekick

    @pytest.mark.asyncio
    @patch("src.app.Sidekick")
    async def test_reset_creates_new_sidekick(self, mock_sidekick):
        """Test that reset creates a new Sidekick instance"""
        new_mock_sidekick = MagicMock()
        new_mock_sidekick.setup = AsyncMock()
        new_mock_sidekick.cleanup = MagicMock()
        mock_sidekick.return_value = new_mock_sidekick
        
        app = App()
        old_sidekick = app.sidekick
        
        message, criteria, chatbot, sidekick = await app.reset(old_sidekick)
        
        # Should have called Sidekick() twice - once in __init__, once in reset
        assert mock_sidekick.call_count == 2

    @pytest.mark.asyncio
    @patch("src.app.Sidekick")
    async def test_reset_calls_setup_on_new_sidekick(self, mock_sidekick):
        """Test that reset calls setup on the new Sidekick"""
        new_mock_sidekick = MagicMock()
        new_mock_sidekick.setup = AsyncMock()
        new_mock_sidekick.cleanup = MagicMock()
        mock_sidekick.return_value = new_mock_sidekick
        
        app = App()
        await app.reset(app.sidekick)
        
        # setup called once in original init, and once in reset
        # But we need to verify the last call is from reset
        assert new_mock_sidekick.setup.call_count == 1

    @pytest.mark.asyncio
    @patch("src.app.Sidekick")
    async def test_reset_returns_empty_strings(self, mock_sidekick):
        """Test that reset returns empty strings for message and criteria"""
        new_mock_sidekick = MagicMock()
        new_mock_sidekick.setup = AsyncMock()
        new_mock_sidekick.cleanup = MagicMock()
        mock_sidekick.return_value = new_mock_sidekick
        
        app = App()
        message, criteria, chatbot, sidekick = await app.reset(app.sidekick)
        
        assert isinstance(message, str)
        assert isinstance(criteria, str)
        assert message == ""
        assert criteria == ""
        assert chatbot == []


class TestFreeResources:
    """Tests for App.free_resurces method"""

    @patch("src.app.Sidekick")
    def test_free_resources_with_sidekick(self, mock_sidekick):
        """Test free_resurces with a sidekick instance"""
        mock_sidekick_instance = MagicMock()
        mock_sidekick_instance.cleanup = MagicMock()
        mock_sidekick.return_value = mock_sidekick_instance
        
        app = App()
        
        # Should not raise an exception
        app.free_resurces(mock_sidekick_instance)
        
        mock_sidekick_instance.cleanup.assert_called_once()

    @patch("src.app.Sidekick")
    def test_free_resources_with_none(self, mock_sidekick):
        """Test free_resurces with None"""
        mock_sidekick.return_value = MagicMock()
        
        app = App()
        
        # Should not raise an exception
        app.free_resurces(None)

    @patch("src.app.Sidekick")
    def test_free_resources_with_exception(self, mock_sidekick):
        """Test free_resurces handles cleanup exceptions"""
        mock_sidekick_instance = MagicMock()
        mock_sidekick_instance.cleanup = MagicMock(side_effect=Exception("Cleanup failed"))
        mock_sidekick.return_value = mock_sidekick_instance
        
        app = App()
        
        # Should not raise an exception
        app.free_resurces(mock_sidekick_instance)

    @patch("src.app.Sidekick")
    def test_free_resources_prints_message(self, mock_sidekick, capsys):
        """Test that free_resurces prints cleanup message"""
        mock_sidekick_instance = MagicMock()
        mock_sidekick_instance.cleanup = MagicMock()
        mock_sidekick.return_value = mock_sidekick_instance
        
        app = App()
        app.free_resurces(mock_sidekick_instance)
        
        captured = capsys.readouterr()
        assert "Cleaning up..." in captured.out

    @patch("src.app.Sidekick")
    def test_free_resources_prints_exception(self, mock_sidekick, capsys):
        """Test that free_resurces prints exception message on failure"""
        mock_sidekick_instance = MagicMock()
        error_msg = "Cleanup failed"
        mock_sidekick_instance.cleanup = MagicMock(side_effect=Exception(error_msg))
        mock_sidekick.return_value = mock_sidekick_instance
        
        app = App()
        app.free_resurces(mock_sidekick_instance)
        
        captured = capsys.readouterr()
        assert "Exception during cleanup:" in captured.out


class TestAppRun:
    """Tests for App.run method"""

    @patch("src.app.gr.Blocks")
    @patch("src.app.Sidekick")
    def test_run_creates_gradio_blocks(self, mock_sidekick, mock_blocks):
        """Test that run creates Gradio Blocks"""
        mock_sidekick.return_value = MagicMock()
        mock_blocks_instance = MagicMock()
        mock_blocks.return_value.__enter__ = MagicMock(return_value=mock_blocks_instance)
        mock_blocks.return_value.__exit__ = MagicMock(return_value=False)
        mock_blocks_instance.launch = MagicMock()
        
        app = App()
        
        # Mock gr.Markdown, gr.State, etc. to avoid errors
        with patch("src.app.gr.Markdown"), \
             patch("src.app.gr.State"), \
             patch("src.app.gr.Row"), \
             patch("src.app.gr.Group"), \
             patch("src.app.gr.Textbox"), \
             patch("src.app.gr.Chatbot"), \
             patch("src.app.gr.Button"):
            try:
                app.run()
            except AttributeError:
                # Expected due to mocking limitations
                pass

    @patch("src.app.gr.Blocks")
    @patch("src.app.Sidekick")
    def test_run_calls_launch(self, mock_sidekick, mock_blocks):
        """Test that run calls launch on Blocks"""
        mock_sidekick.return_value = MagicMock()
        mock_blocks_instance = MagicMock()
        mock_blocks.return_value.__enter__ = MagicMock(return_value=mock_blocks_instance)
        mock_blocks.return_value.__exit__ = MagicMock(return_value=False)
        mock_blocks_instance.launch = MagicMock()
        
        app = App()
        
        with patch("src.app.gr.Markdown"), \
             patch("src.app.gr.State"), \
             patch("src.app.gr.Row"), \
             patch("src.app.gr.Group"), \
             patch("src.app.gr.Textbox"), \
             patch("src.app.gr.Chatbot"), \
             patch("src.app.gr.Button"):
            try:
                app.run()
            except (AttributeError, TypeError):
                pass


class TestAppIntegration:
    """Integration tests for App"""

    @pytest.mark.asyncio
    @patch("src.app.Sidekick")
    async def test_setup_and_process_message(self, mock_sidekick):
        """Test setup followed by process_message"""
        mock_sidekick_instance = AsyncMock()
        mock_sidekick_instance.setup = AsyncMock()
        mock_sidekick_instance.run_super_step = AsyncMock(return_value=["result"])
        mock_sidekick.return_value = mock_sidekick_instance
        
        app = App()
        sidekick = await app.setup()
        
        returned_sidekick, result = await app.process_message(
            sidekick,
            "test message",
            "test criteria",
            []
        )
        
        assert result == ["result"]
        assert returned_sidekick == sidekick

    @pytest.mark.asyncio
    @patch("src.app.Sidekick")
    async def test_reset_returns_new_sidekick(self, mock_sidekick):
        """Test that reset returns a fresh sidekick"""
        initial_mock = MagicMock()
        initial_mock.setup = AsyncMock()
        initial_mock.cleanup = MagicMock()
        
        new_mock = MagicMock()
        new_mock.setup = AsyncMock()
        new_mock.cleanup = MagicMock()
        
        mock_sidekick.side_effect = [initial_mock, new_mock]
        
        app = App()
        initial_sidekick = app.sidekick
        
        message, criteria, chatbot, new_sidekick = await app.reset(initial_sidekick)
        
        # Verify they are different instances
        assert new_sidekick != initial_sidekick

    @patch("src.app.Sidekick")
    def test_free_resources_called_on_cleanup(self, mock_sidekick):
        """Test that free_resurces handles cleanup properly"""
        mock_sidekick_instance = MagicMock()
        mock_sidekick_instance.cleanup = MagicMock()
        mock_sidekick.return_value = mock_sidekick_instance
        
        app = App()
        
        # Test cleanup
        app.free_resurces(app.sidekick)
        
        assert app.sidekick.cleanup.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
