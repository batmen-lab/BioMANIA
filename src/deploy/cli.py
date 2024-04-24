import re
import base64
import json
import requests
from datetime import datetime
from colorama import Fore, Style  ##### Added for colored output

def encode_file_to_base64(filepath):
    """Encode file content to base64."""
    with open(filepath, "rb") as file:
        return base64.b64encode(file.read()).decode('utf-8')

def create_request(text, lib, filepath=None):
    """Create JSON request data for the backend."""
    session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    request_data = {
        "text": text,
        "top_k": 1,
        "method": "No specified",
        "Lib": lib,
        "files": [],
        "new_lib_github_url": "",
        "new_lib_doc_url": "",
        "api_html": "",
        "lib_alias": "",
        "conversation_started": False,
        "session_id": session_id,
        "optionalParams": ""
    }
    if filepath:
        file_content = encode_file_to_base64(filepath)
        request_data['files'].append(file_content)
    return request_data

def send_request_to_backend(data, url = "http://localhost:5000/stream"):
    """Send the request to the backend server with streaming response."""
    response = requests.post(url, json=data, stream=True)
    return response.iter_lines()

def parse_backend_response(response):
    """Parse the response from the backend and extract meaningful data."""
    messages = []
    for line in response:
        if line:
            event = json.loads(line.decode('utf-8'))
            method_name = event.get('method_name')
            block_id = event.get('block_id')
            if method_name in ["on_tool_start", "on_tool_end", "on_agent_end"]:
                pass
            elif method_name == "on_agent_action":
                task_title = event.get('task_title', '')
                task = event.get('task', '')
                block_id = event.get('block_id', '')
                imageData = event.get('imageData', '')
                tableData = event.get('tableData', '')  ##### Check and include tableData
                if 'log-' in block_id:
                    message = f"**{task_title}**:\n {task}\n"
                    if '[Fail]' in task_title:
                        messages.append(Fore.RED + message + Style.RESET_ALL)
                    elif '[Success]' in task_title:
                        messages.append(Fore.GREEN + message + Style.RESET_ALL)
                    elif 'Double Check' in task_title:
                        messages.append(Fore.MAGENTA + message + Style.RESET_ALL)
                    elif 'Predicted API' in task_title:
                        messages.append(Fore.YELLOW + message + Style.RESET_ALL)
                    else:
                        messages.append(message)
                    if imageData:
                        image_path = f"tmp/{event['session_id']}.png"
                        with open(image_path, "wb") as img_file:
                            img_file.write(base64.b64decode(imageData))
                        messages.append(Fore.GREEN + f"Image saved to {image_path}" + Style.RESET_ALL)
                    if tableData:
                        messages.append(Fore.GREEN + f"Data: {tableData}" + Style.RESET_ALL)
                if 'code-' in block_id:
                    messages.append(Fore.YELLOW + f"**{task_title}**\n: {task}\n" + Style.RESET_ALL)
            else:
                raise ValueError(f"Unknown method name: {method_name}")
    return messages

def generate_biomania_pattern():
    a = "#####   " + "##### " +" ##### "+" ##   ## "+"   ##   " + " ##    ## "+ "##### "+"   ##   \n"\
        "##   ## " + "  ##  " +"##   ##"+" ### ### "+"  ####  " + " ###   ## "+ "  ##  "+"  ####  \n"\
        "#####   " + "  ##  " +"##   ##"+" ## # ## "+" ##  ## " + " ####  ## "+ "  ##  "+" ##  ## \n" \
        "##   ## " + "  ##  " +"##   ##"+" ##   ## "+"########" + " ## ## ## "+ "  ##  "+"########\n"\
        "#####   " + "##### " +" ##### "+" ##   ## "+"##    ##" + " ##  #### "+ "##### "+"##    ##\n"
    return a

def main():
    biomania_pattern = generate_biomania_pattern()
    print(Fore.GREEN + biomania_pattern + Style.RESET_ALL)
    print("="*80)
    print(Fore.GREEN + "Welcome to BioMANIA!" + Style.RESET_ALL)
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
    while True:
        user_input = input("Enter your inquiry:\n (if you have uploaded files, present as '<file>/path/to/file</file>') or enter 'exit' to quit: ")
        print("="*80)
        print("User: ", Fore.BLUE + user_input + Style.RESET_ALL)
        if user_input.lower() == "exit":
            print(Fore.GREEN + "BioMANIA 👋: Goodbye!" + Style.RESET_ALL)
            break
        match = re.search(r"<file>(.*?)</file>", user_input)
        if match:
            file_path = match.group(1)
            text = re.sub(r"<file>.*?</file>", "", user_input).strip()
            request_data = create_request(text, library, file_path)
        else:
            request_data = create_request(user_input, library)
        response = send_request_to_backend(request_data)
        output = parse_backend_response(response)
        print(Fore.GREEN + "BioMANIA: 🤔" + Style.RESET_ALL)
        for msg in output:
            print(msg)
        print("="*80)

if __name__ == "__main__":
    main()
