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

def create_request(text, lib, session_id, filepath=None):
    """Create JSON request data for the backend."""
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

def parse_backend_response(response, yield_load=True, add_color=True):
    """Parse the response from the backend and extract meaningful data."""
    from colorama import Fore, Style  # Ensure these are imported only if required
    def colorize(message, color):
        """Apply color format to the message if use_colors is True."""
        return f"{color}{message}{Style.RESET_ALL}" if add_color else message
    messages = []
    for line in response:
        if line:
            if yield_load:
                event = json.loads(line.decode('utf-8'))
            else:
                event = line
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
                    if '[Fail]' in task_title or 'GPT predict Error' in task_title:
                        messages.append(colorize(message, Fore.RED))
                    elif '[Success]' in task_title:
                        messages.append(colorize(message, Fore.GREEN))
                    elif 'Double Check' in task_title or 'Enter Parameters' in task_title:
                        messages.append(colorize(message, Fore.MAGENTA))
                    elif 'Predicted API' in task_title:
                        messages.append(colorize(message, Fore.YELLOW))
                    else:
                        messages.append(message)
                    if imageData:
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        image_path = f"tmp/{timestamp}.webp"
                        with open(image_path, "wb") as img_file:
                            img_file.write(base64.b64decode(imageData))
                        messages.append(colorize(f"Image saved to {image_path}", Fore.GREEN))
                    if tableData:
                        messages.append(colorize(f"Data: {tableData}", Fore.GREEN))
                if 'code-' in block_id:
                    messages.append(colorize(f"**{task_title}**:\n {task}\n", Fore.YELLOW))
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
    session_id = datetime.now().strftime("%Y%m%d%H%M%S")
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
            request_data = create_request(text, library, session_id, file_path)
        else:
            request_data = create_request(user_input, library, session_id)
        response = send_request_to_backend(request_data)
        output = parse_backend_response(response)
        print(Fore.GREEN + "BioMANIA: 🤔" + Style.RESET_ALL)
        for msg in output:
            print(msg)
        print("="*80)

if __name__ == "__main__":
    main()
