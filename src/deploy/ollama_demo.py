from ..deploy.model import Model
import os, torch
from datetime import datetime
from colorama import Fore, Style
from ..deploy.cli import encode_file_to_base64, parse_backend_response
from ..deploy.cli_demo import parse_backend_queue

def initialize_model():
    from loguru import logger
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(f"./logs", exist_ok=True)
    logger.remove()
    logger.add(f"./logs/BioMANIA_log_{timestamp}.log", rotation="500 MB", retention="7 days", level="INFO")
    logger.info("Loguru initialized successfully.")
    print("Logging setup complete.")
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(logger, device)
    return model

def process_input(model, user_input, library, session_id, conversation_started):
    if "<file>" in user_input:
        path_start = user_input.find("<file>") + 6
        path_end = user_input.find("</file>")
        filepath = user_input[path_start:path_end]
        user_input = user_input[:path_start-6] + user_input[path_end+7:]
        file_content = encode_file_to_base64(filepath)
        print(Fore.YELLOW + "File encoded to base64 for processing: " + file_content[:30] + "..." + Style.RESET_ALL)
    model.run_pipeline(user_input, library, top_k=1, files=[], conversation_started=conversation_started, session_id=session_id)
    messages = parse_backend_queue(model.queue)
    responses = []
    for msg in messages:
        output = parse_backend_response([msg], yield_load=False)
        responses.extend(output)
    return responses
