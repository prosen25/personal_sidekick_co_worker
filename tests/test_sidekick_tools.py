import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call
from langchain_core.tools import Tool

# Add the parent directory to sys.path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sidekick_tools import (
    push,
    get_file_tools,
    playwright_tools,
    other_tools,
)


class TestPushNotification:
    """Tests for the push notification function"""

    @patch.dict(os.environ, {
        "PUSHOVER_TOKEN": "test_token",
        "PUSHOVER_USER": "test_user"
    })
    @patch("src.sidekick_tools.requests.post")
    def test_push_success(self, mock_post):
        """Test successful push notification"""
        result = push("Test message")
        
        assert result == "success"
        mock_post.assert_called_once_with(
            url="https://api.pushover.net/1/messages.json",
            data={
                "token": "test_token",
                "user": "test_user",
                "message": "Test message"
            },
            timeout=10
        )

    @patch.dict(os.environ, {
        "PUSHOVER_TOKEN": "test_token",
        "PUSHOVER_USER": "test_user"
    })
    @patch("src.sidekick_tools.requests.post")
    def test_push_with_empty_message(self, mock_post):
        """Test push notification with empty message"""
        result = push("")
        
        assert result == "success"
        mock_post.assert_called_once()

    @patch.dict(os.environ, {
        "PUSHOVER_TOKEN": "test_token",
        "PUSHOVER_USER": "test_user"
    })
    @patch("src.sidekick_tools.requests.post")
    def test_push_with_long_message(self, mock_post):
        """Test push notification with long message"""
        long_message = "x" * 1000
        result = push(long_message)
        
        assert result == "success"
        mock_post.assert_called_once()

    @patch.dict(os.environ, {
        "PUSHOVER_TOKEN": "",
        "PUSHOVER_USER": ""
    })
    @patch("src.sidekick_tools.requests.post")
    def test_push_with_missing_env_vars(self, mock_post):
        """Test push notification with missing environment variables"""
        result = push("Test message")
        
        assert "not configured" in result
        mock_post.assert_not_called()

    @patch.dict(os.environ, {
        "PUSHOVER_TOKEN": "test_token",
        "PUSHOVER_USER": "test_user"
    })
    @patch("src.sidekick_tools.requests.post")
    def test_push_with_special_characters(self, mock_post):
        """Test push notification with special characters"""
        special_msg = "Test 🎉 with émojis and spëcial çharâcters!"
        result = push(special_msg)
        
        assert result == "success"
        call_args = mock_post.call_args
        assert call_args[1]["data"]["message"] == special_msg


class TestFileTools:
    """Tests for file handling tools"""

    @patch("src.sidekick_tools.FileManagementToolkit")
    def test_get_file_tools(self, mock_toolkit_class):
        """Test getting file tools"""
        mock_toolkit = MagicMock()
        mock_tools = [MagicMock(), MagicMock()]
        mock_toolkit.get_tools.return_value = mock_tools
        mock_toolkit_class.return_value = mock_toolkit
        
        result = get_file_tools()
        
        mock_toolkit_class.assert_called_once_with(root_dir="sandbox")
        assert result == mock_tools

    @patch("src.sidekick_tools.FileManagementToolkit")
    def test_get_file_tools_returns_list(self, mock_toolkit_class):
        """Test that get_file_tools returns a list"""
        mock_toolkit = MagicMock()
        mock_toolkit.get_tools.return_value = []
        mock_toolkit_class.return_value = mock_toolkit
        
        result = get_file_tools()
        
        assert isinstance(result, list)


class TestPlaywrightTools:
    """Tests for playwright tools"""

    @pytest.mark.asyncio
    @patch("src.sidekick_tools.async_playwright")
    async def test_playwright_tools(self, mock_async_pw):
        """Test getting playwright tools"""
        # Setup mocks
        mock_playwright = MagicMock()
        mock_browser = MagicMock()
        mock_toolkit = MagicMock()
        mock_tools = [MagicMock(), MagicMock()]
        
        mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright)
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_toolkit.get_tools.return_value = mock_tools
        
        with patch("src.sidekick_tools.PlayWrightBrowserToolkit", return_value=mock_toolkit):
            tools, browser, playwright = await playwright_tools()
        
        assert tools == mock_tools
        assert browser == mock_browser
        assert playwright == mock_playwright

    @pytest.mark.asyncio
    @patch("src.sidekick_tools.async_playwright")
    async def test_playwright_tools_returns_three_items(self, mock_async_pw):
        """Test that playwright_tools returns three items"""
        mock_playwright = MagicMock()
        mock_browser = MagicMock()
        mock_toolkit = MagicMock()
        mock_toolkit.get_tools.return_value = []
        
        mock_async_pw.return_value.start = AsyncMock(return_value=mock_playwright)
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        
        with patch("src.sidekick_tools.PlayWrightBrowserToolkit", return_value=mock_toolkit):
            result = await playwright_tools()
        
        assert len(result) == 3


class TestOtherTools:
    """Tests for other tools consolidation"""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"SERPER_API_KEY": "test_key"})
    @patch("src.sidekick_tools.PythonREPLTool")
    @patch("src.sidekick_tools.WikipediaQueryRun")
    @patch("src.sidekick_tools.WikipediaAPIWrapper")
    @patch("src.sidekick_tools.GoogleSerperAPIWrapper")
    @patch("src.sidekick_tools.get_file_tools")
    @patch("src.sidekick_tools.push")
    async def test_other_tools(
        self,
        mock_push,
        mock_get_file_tools,
        mock_serper,
        mock_wiki_wrapper,
        mock_wiki_query_run,
        mock_python_repl
    ):
        """Test consolidating all other tools"""
        # Setup mocks
        mock_get_file_tools.return_value = [MagicMock()]
        mock_serper_instance = MagicMock()
        mock_serper.return_value = mock_serper_instance
        
        result = await other_tools()
        
        # Verify result structure
        assert isinstance(result, list)
        assert len(result) >= 4
        
        # Verify tool names
        tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in result]
        assert any('send_push_notification' in str(name) or name == 'send_push_notification' for name in tool_names)
        assert any('google_search' in str(name) or name == 'google_search_tool' for name in tool_names)

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"SERPER_API_KEY": "test_key"})
    @patch("src.sidekick_tools.PythonREPLTool")
    @patch("src.sidekick_tools.WikipediaQueryRun")
    @patch("src.sidekick_tools.WikipediaAPIWrapper")
    @patch("src.sidekick_tools.GoogleSerperAPIWrapper")
    @patch("src.sidekick_tools.get_file_tools")
    @patch("src.sidekick_tools.push")
    async def test_other_tools_includes_file_tools(
        self,
        mock_push,
        mock_get_file_tools,
        mock_serper,
        mock_wiki_wrapper,
        mock_wiki_query_run,
        mock_python_repl
    ):
        """Test that other_tools includes file handling tools"""
        file_tools = [MagicMock(name="file_tool_1"), MagicMock(name="file_tool_2")]
        mock_get_file_tools.return_value = file_tools
        
        result = await other_tools()
        
        # File tools should be included
        assert len(result) >= len(file_tools)

    @pytest.mark.asyncio
    @patch("src.sidekick_tools.PythonREPLTool")
    @patch("src.sidekick_tools.WikipediaQueryRun")
    @patch("src.sidekick_tools.WikipediaAPIWrapper")
    @patch("src.sidekick_tools.GoogleSerperAPIWrapper")
    @patch("src.sidekick_tools.get_file_tools")
    @patch("src.sidekick_tools.push")
    async def test_other_tools_creates_send_push_notification_tool(
        self,
        mock_push,
        mock_get_file_tools,
        mock_serper,
        mock_wiki_wrapper,
        mock_wiki_query_run,
        mock_python_repl
    ):
        """Test that other_tools creates send_push_notification tool"""
        mock_get_file_tools.return_value = []
        
        result = await other_tools()
        
        # Find send_push_notification tool
        push_tools = [tool for tool in result if hasattr(tool, 'name') and tool.name == 'send_push_notification']
        assert len(push_tools) == 1
        
        # Verify it's a Tool instance
        push_tool = push_tools[0]
        assert isinstance(push_tool, Tool)

    @pytest.mark.asyncio
    @patch("src.sidekick_tools.PythonREPLTool")
    @patch("src.sidekick_tools.WikipediaQueryRun")
    @patch("src.sidekick_tools.WikipediaAPIWrapper")
    @patch("src.sidekick_tools.GoogleSerperAPIWrapper")
    @patch("src.sidekick_tools.get_file_tools")
    @patch("src.sidekick_tools.push")
    async def test_other_tools_creates_google_search_tool(
        self,
        mock_push,
        mock_get_file_tools,
        mock_serper,
        mock_wiki_wrapper,
        mock_wiki_query_run,
        mock_python_repl
    ):
        """Test that other_tools creates google_search_tool"""
        mock_get_file_tools.return_value = []
        mock_serper_instance = MagicMock()
        mock_serper.return_value = mock_serper_instance
        
        result = await other_tools()
        
        # Find google search tool
        search_tools = [tool for tool in result if hasattr(tool, 'name') and tool.name == 'google_search_tool']
        assert len(search_tools) == 1
        
        # Verify it's a Tool instance
        search_tool = search_tools[0]
        assert isinstance(search_tool, Tool)

    @pytest.mark.asyncio
    @patch("src.sidekick_tools.PythonREPLTool")
    @patch("src.sidekick_tools.WikipediaQueryRun")
    @patch("src.sidekick_tools.WikipediaAPIWrapper")
    @patch("src.sidekick_tools.GoogleSerperAPIWrapper")
    @patch("src.sidekick_tools.get_file_tools")
    @patch("src.sidekick_tools.push")
    async def test_other_tools_includes_wiki_tool(
        self,
        mock_push,
        mock_get_file_tools,
        mock_serper,
        mock_wiki_wrapper,
        mock_wiki_query_run,
        mock_python_repl
    ):
        """Test that other_tools includes Wikipedia tool"""
        mock_get_file_tools.return_value = []
        
        result = await other_tools()
        
        # Verify WikipediaAPIWrapper was instantiated
        mock_wiki_wrapper.assert_called_once()
        
        # Verify WikipediaQueryRun was called with the wrapper
        mock_wiki_query_run.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.sidekick_tools.PythonREPLTool")
    @patch("src.sidekick_tools.WikipediaQueryRun")
    @patch("src.sidekick_tools.WikipediaAPIWrapper")
    @patch("src.sidekick_tools.GoogleSerperAPIWrapper")
    @patch("src.sidekick_tools.get_file_tools")
    @patch("src.sidekick_tools.push")
    async def test_other_tools_includes_python_repl_tool(
        self,
        mock_push,
        mock_get_file_tools,
        mock_serper,
        mock_wiki_wrapper,
        mock_wiki_query_run,
        mock_python_repl
    ):
        """Test that other_tools includes Python REPL tool"""
        mock_get_file_tools.return_value = []
        
        result = await other_tools()
        
        # Verify PythonREPLTool was instantiated
        mock_python_repl.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
