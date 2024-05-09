import gradio as gr
from datetime import datetime
import os
from .model import Model
from .cli import encode_file_to_base64, parse_backend_response
from colorama import Fore, Style
from .cli_demo import parse_backend_queue

from loguru import logger
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
os.makedirs(f"./logs", exist_ok=True)
logger.remove()
logger.add(f"./logs/BioMANIA_log_{timestamp}.log", rotation="500 MB", retention="7 days", level="INFO")
logger.info("Loguru initialized successfully.")
#global model
#model.__init__()

def process_input(user_input, library):
    global model
    global conversation_started
    #session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    if "<file>" in user_input:
        path_start = user_input.find("<file>") + 6
        path_end = user_input.find("</file>")
        filepath = user_input[path_start:path_end]
        user_input = user_input[:path_start-6] + user_input[path_end+7:]
        file_content = encode_file_to_base64(filepath)
    model.run_pipeline(user_input, library, top_k=1, files=[], conversation_started=conversation_started, session_id="")
    conversation_started = False
    messages = parse_backend_queue(model.queue)
    #output_texts = ["<div style='font-family: Arial, sans-serif;'>"]
    output_texts = []
    for msg in messages:
        output = parse_backend_response([msg], yield_load=False, add_color=False)
        for out in output:
            output_texts.append(out)
            #formatted_output = ansi_to_html(out)
            #output_texts.append(formatted_output)
    #output_texts.append("</div>")
    return "\n".join(output_texts)

def ansi_to_html(text):
    mappings = {
        "\x1b[33m": '<span style="color: orange;">',
        "\x1b[35m": '<span style="color: purple;">',
        "\x1b[0m": '</span>'
    }
    for ansi, html in mappings.items():
        text = text.replace(ansi, html)
    return text

def main():
    global model
    global conversation_started
    conversation_started = True
    model = Model(logger=logger, device='cpu', model_llm_type='llama3')
    libs = ["scanpy", "squidpy", "ehrapy", "snapatac2"]
    description = "Welcome to BioMANIA Web Demo! Choose a library and enter your inquiry."
    iface = gr.Interface(
        fn=process_input,
        inputs=[
            gr.Textbox(lines=5, placeholder="Type your inquiry here..."), 
            gr.Dropdown(choices=libs, label="Library")
        ],
        #outputs=gr.Textbox(),
        outputs=gr.Textbox(),
        title="BioMANIA Model Interaction",
        description=description
    )
    iface.launch()

if __name__ == "__main__":
    main()

