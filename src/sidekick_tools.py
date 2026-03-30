import os

import requests
from dotenv import load_dotenv
from langchain_community.agent_toolkits import (
    FileManagementToolkit,
    PlayWrightBrowserToolkit,
)
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import Tool
from langchain_experimental.tools import PythonREPLTool
from playwright.async_api import async_playwright


class SidekickTools:
    """Factory and utility methods used by the sidekick app."""

    def __init__(self, file_root_dir: str = "sandbox", headless: bool = False):
        load_dotenv(override=True)
        self.file_root_dir = file_root_dir
        self.headless = headless

    async def playwright_tools(self):
        """Get Playwright tools from LangChain community."""
        playwright = await async_playwright().start()
        async_browser = await playwright.chromium.launch(headless=self.headless)
        toolkit = PlayWrightBrowserToolkit(async_browser=async_browser)
        return toolkit.get_tools(), async_browser, playwright

    def push(self, text: str) -> str:
        """Send a push notification to the user."""
        pushover_token = os.getenv("PUSHOVER_TOKEN")
        pushover_user = os.getenv("PUSHOVER_USER")
        pushover_url = "https://api.pushover.net/1/messages.json"

        requests.post(
            url=pushover_url,
            data={
                "token": pushover_token,
                "user": pushover_user,
                "message": text,
            },
        )
        return "success"

    def get_file_tools(self):
        """Get file handling tools of LangChain."""
        toolkit = FileManagementToolkit(root_dir=self.file_root_dir)
        return toolkit.get_tools()

    async def other_tools(self):
        """Get all tools except Playwright used by the sidekick app."""
        send_push_notification_tool = Tool(
            name="send_push_notification",
            func=self.push,
            description="Use this tool when you want to send a push notification to a user",
        )
        file_handling_tools = self.get_file_tools()

        serper = GoogleSerperAPIWrapper()
        google_search_tool = Tool(
            name="google_search_tool",
            func=serper.run,
            description="Use this tool when you want result of an online web search",
        )

        wikipedia = WikipediaAPIWrapper()
        wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)
        python_repl_tool = PythonREPLTool()

        return file_handling_tools + [
            send_push_notification_tool,
            google_search_tool,
            wiki_tool,
            python_repl_tool,
        ]


# Backward-compatible function API
_default_tools = SidekickTools()


async def playwright_tools():
    return await _default_tools.playwright_tools()


def push(text: str) -> str:
    return _default_tools.push(text)


def get_file_tools():
    return _default_tools.get_file_tools()


async def other_tools():
    return await _default_tools.other_tools()
