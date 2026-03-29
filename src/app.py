from sidekick import Sidekick

import gradio as gr


class App:
    def __init__(self):
        self.sidekick = Sidekick()

    async def setup(self) -> Sidekick:
        await self.sidekick.setup()
        return self.sidekick
    
    async def process_message(self, sidekick, message, success_criteria, history):
        result = sidekick.run_super_step(message=message, success_criteria=success_criteria, history=history)
        return result, sidekick
    
    async def reset(self):
        new_sidekick = Sidekick()
        await new_sidekick.setup()
        return "", "", None, new_sidekick
    
    def free_resurces(self, sidekick):
        print("Cleaning up...")
        try:
            if sidekick:
                sidekick.cleanup()
        except Exception as e:
            print(f"Exception during cleanup: {e}")

    def run(self):
        with gr.Blocks(title="Sidekick") as ui:
            gr.Markdown("## Sidekick Personal Co-Worker")
            sidekick = gr.State(delete_callback=self.free_resurces)

            with gr.Row():
                chatbot = gr.Chatbot(label="Sidekick", height=300, type="messages")
            with gr.Group():
                with gr.Row():
                    message = gr.Textbox(show_label=False, placeholder="Your request to the Sidekick")
                with gr.Row():
                    success_criteria = gr.Textbox(show_label=False, placeholder="What are your success critiera?")
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
                inputs=[],
                outputs=[message, success_criteria, chatbot, sidekick]
            )
        
        ui.launch(inbrowser=True)

if __name__ == "__main__":
    App().run()