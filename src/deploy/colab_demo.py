from ..deploy.model import Model

import os
from datetime import datetime
from colorama import Fore, Style
from ..deploy.cli import encode_file_to_base64, parse_backend_response

def parse_backend_queue(queue):
    messages = []
    while not queue.empty():
        event = queue.get()
        messages.append(event)
    return messages

def colab_demo():
    from loguru import logger
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(f"./logs", exist_ok=True)
    logger.remove()
    logger.add(f"./logs/BioMANIA_log_{timestamp}.log", rotation="500 MB", retention="7 days", level="INFO")
    logger.info("Loguru initialized successfully.")
    print("Logging setup complete.")
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    device='cuda'
    model = Model(logger,device)
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
        model.run_pipeline(user_input, library, top_k=1, files=[],conversation_started=conversation_started,session_id=session_id)
        conversation_started = False
        messages = parse_backend_queue(model.queue)
        for msg in messages:
            output = parse_backend_response([msg], yield_load=False)
            for out in output:
                print(out)

if __name__ == "__main__":
    colab_demo()