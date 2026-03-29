import asyncio
import uuid

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from evaluator import Evaluator
from sidekick_tools import other_tools, playwright_tools
from state import State
from worker import Worker


load_dotenv(override=True)

class Sidekick:
    def __init__(self):
        self.worker = Worker()
        self.evaluator = Evaluator()
        self.tools = None
        self.async_browser = None
        self.playwright = None
        self.memory = MemorySaver()
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())

    async def build_graph(self):
        # Set up graph builder
        graph_builder = StateGraph(State)

        # Create nodes
        graph_builder.add_node("worker", self.worker.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator.evaluator)

        # Create edges
        graph_builder.add_edge(START, "worker")
        graph_builder.add_conditional_edges("worker", self.worker.worker_route, {"tools": "tools", "evaluator": "evaluator"})
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges("evaluator", self.evaluator.route_based_on_evaluation, {"worker": "worker", "END": END})

        # Compile the graph builder with memory
        self.graph = graph_builder.compile(checkpointer=self.memory)

    async def setup(self):
        self.tools, self.async_browser, self.playwright = await playwright_tools()
        self.tools += other_tools()
        self.worker.setup(tools=self.tools)
        self.evaluator.setup()
        await self.build_graph()

    async def run_super_step(self, message, success_criteria, history):
        config = {"configurable": {"thread_id": self.sidekick_id}}

        state = {
            "messages": message,
            "success_criteria": success_criteria,
            "success_criteria_met": False,
            "user_input_needed": False
        }

        result = await self.graph.ainvoke(input=state, config=config)
        user = {"role": "user", "content": message}
        reply = {"role": "assistant", "content": result["messages"][-2].content}
        feedback = {"role": "assistant", "content": result["messages"][-1].content}

        return history + [user, reply, feedback]
    
    def cleanup(self):
        if self.async_browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.async_browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except:
                # If no loop is running, do a direct run
                asyncio.run(self.async_browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())