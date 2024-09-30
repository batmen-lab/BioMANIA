"""
Author: Zhengyuan Dong
Email: zydong122@gmail.com
Description: This script contains functions to interact with the BioMANIA CLI.
"""
from ..deploy.model import Model

import os, torch
from datetime import datetime
from colorama import Fore, Style
from ..deploy.cli import encode_file_to_base64, parse_backend_response
import multiprocessing  # Added for multiprocessing

def parse_backend_queue(queue):
    messages = []
    while not queue.empty():
        event = queue.get()
        messages.append(event)
    return messages

def cli_demo():
    from loguru import logger
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(f"./logs", exist_ok=True)
    logger.remove()
    logger.add(f"./logs/BioMANIA_log_{timestamp}.log", rotation="500 MB", retention="7 days", level="INFO")
    logger.info("Loguru initialized successfully.")
    print("Logging setup complete.")
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(logger,device)
    print(model.user_states)
    print(Fore.GREEN + "Welcome to BioMANIA CLI Demo!" + Style.RESET_ALL)
    print(Fore.BLUE + "[Would you like to see some examples to learn how to interact with the bot?](https://github.com/batmen-lab/BioMANIA/tree/main/examples)" + Style.RESET_ALL)
    libs = ["scanpy", "squidpy", "ehrapy", "snapatac2"]
    library = None
    while True and (not library):
        print("="*80)
        user_input = input(Fore.MAGENTA + "Before we start our chat, please choose one lib from Our Currently Available libraries: [" + ", ".join(libs) + "]: " + Style.RESET_ALL)
        if user_input.lower() not in libs:
            print(Fore.RED + "BioMANIA 🤔: Invalid library. Please choose from the available libraries." + Style.RESET_ALL)
        else:
            library = user_input.lower()
        print("="*80)
    session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    conversation_started = True
    while True:
        user_input = input("Enter your inquiry:\n (if you have uploaded files, present as '<file>/path/to/file</file>') or enter 'exit' to quit: ")
        print("="*80)
        print("User: ", Fore.BLUE + user_input + Style.RESET_ALL)
        if user_input.lower() == "exit":
            print(Fore.GREEN + "BioMANIA 👋: Goodbye!" + Style.RESET_ALL)
            break
        if "<file>" in user_input:
            path_start = user_input.find("<file>") + 6
            path_end = user_input.find("</file>")
            filepath = user_input[path_start:path_end]
            user_input = user_input[:path_start-6] + user_input[path_end+7:]
            file_content = encode_file_to_base64(filepath)
            print(Fore.YELLOW + "File encoded to base64 for processing: " + file_content[:30] + "..." + Style.RESET_ALL)
        model.run_pipeline(user_input, library, top_k=1, files=[],conversation_started=conversation_started,session_id=session_id,dialog_mode='T')
        conversation_started = False
        messages = parse_backend_queue(model.queue)
        #print('messages:', messages)
        for msg in messages:
            output = parse_backend_response([msg], yield_load=False)
            for out in output:
                print(out)
        """queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=model.run_pipeline, args=(user_input, library, 1, [], conversation_started, session_id, 'T'))
        process.start()

        while process.is_alive():
            if not queue.empty():
                messages = parse_backend_queue(queue)
                for msg in messages:
                    output = parse_backend_response([msg], yield_load=False)
                    for out in output:
                        print(out)
            process.join()"""

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    cli_demo()