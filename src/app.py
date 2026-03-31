from .sidekick import Sidekick

import gradio as gr
from typing import Dict, List, Optional


class App:
    def __init__(self):
        self.sidekick = Sidekick()

    async def setup(self) -> Sidekick:
        await self.sidekick.setup()
        return self.sidekick
    
    async def process_message(
        self,
        sidekick: Optional[Sidekick],
        message: str,
        success_criteria: str,
        history: Optional[List[Dict[str, str]]],
    ):
        history = history or []

        if sidekick is None:
            sidekick = Sidekick()
            await sidekick.setup()

        if not isinstance(message, str) or not message.strip():
            return sidekick, history

        if not isinstance(success_criteria, str) or not success_criteria.strip():
            feedback = {
                "role": "assistant",
                "content": "Please provide success criteria so I can evaluate task completion.",
            }
            return sidekick, history + [feedback]

        try:
            result = await sidekick.run_super_step(
                message=message,
                success_criteria=success_criteria,
                history=history,
            )
            return sidekick, result
        except Exception as e:
            error_feedback = {"role": "assistant", "content": f"Error while processing request: {e}"}
            return sidekick, history + [error_feedback]
    
    async def reset(self, sidekick: Optional[Sidekick]):
        if sidekick:
            sidekick.cleanup()

        new_sidekick = Sidekick()
        await new_sidekick.setup()
        return "", "", [], new_sidekick
    
    def free_resources(self, sidekick: Optional[Sidekick]):
        print("Cleaning up...")
        try:
            if sidekick:
                sidekick.cleanup()
        except Exception as e:
            print(f"Exception during cleanup: {e}")

    # Backward-compatible typo alias.
    def free_resurces(self, sidekick: Optional[Sidekick]):
        self.free_resources(sidekick)

    def run(self):
        with gr.Blocks(title="Sidekick") as ui:
            gr.Markdown("## Sidekick Personal Co-Worker")
            sidekick = gr.State(delete_callback=self.free_resources)

            with gr.Row():
                chatbot = gr.Chatbot(label="Sidekick", height=300)
            with gr.Group():
                with gr.Row():
                    message = gr.Textbox(show_label=False, placeholder="Your request to the Sidekick")
                with gr.Row():
                    success_criteria = gr.Textbox(show_label=False, placeholder="What are your success criteria?")
            with gr.Row():
                reset_button = gr.Button(value="Reset", variant="stop")
                go_button = gr.Button(value="Go!", variant="primary")

            ui.load(fn=self.setup, inputs=[], outputs=[sidekick])

            message.submit(
                fn=self.process_message,
                inputs=[sidekick, message, success_criteria, chatbot],
                outputs=[sidekick, chatbot]
            )
            success_criteria.submit(
                fn=self.process_message,
                inputs=[sidekick, message, success_criteria, chatbot],
                outputs=[sidekick, chatbot]
            )
            go_button.click(
                fn=self.process_message,
                inputs=[sidekick, message, success_criteria, chatbot],
                outputs=[sidekick, chatbot]
            )
            reset_button.click(
                fn=self.reset,
                inputs=[sidekick],
                outputs=[message, success_criteria, chatbot, sidekick]
            )
        
        ui.launch(inbrowser=True)

if __name__ == "__main__":
    App().run()
