from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
# from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit, FileManagementToolkit
import requests
import os
from langchain_core.tools import Tool

load_dotenv(override=True)

# Get the playwright tools from langchain community
async def playwright_tools():
    """ Get playwright tools of langchain """
    playwright = await async_playwright().start()
    async_browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit(async_browser=async_browser)
    return toolkit.get_tools(), async_browser, playwright

# Used to send push notification to a user
def push(text: str) -> str:
    """ Send a push notification to the user """
    pushover_token = os.getenv("PUSHOVER_TOKEN")
    pushover_user = os.getenv("PUSHOVER_USER")
    pushover_url = "https://api.pushover.net/1/messages.json"

    requests.post(url=pushover_url, data={"token": pushover_token, "user": pushover_user, "message": text})
    
    return "success"

# Get the langchain file handling tools
def get_file_tools():
    """ Get file handling tools of langchain """
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()

# Consolidate all tools except playwritght
async def other_tools():
    """ Get all other tools except playwright which is going to used by sidekick app """
    send_push_notification_tool = Tool(
        name="send_push_notification",
        func=push,
        description="Use this tool when you want to send a push notification to a user"
    )
    file_handling_tools = get_file_tools()

    serper = GoogleSerperAPIWrapper()
    google_search_tool = Tool(
        name="google_search_tool",
        func=serper.run,
        description="Use this tool when you want result of an online web search"
    )

    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    python_repl_tool = PythonREPLTool()

    return file_handling_tools + [send_push_notification_tool, google_search_tool, wiki_tool, python_repl_tool]