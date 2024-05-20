from ..configs.model_config import *
from ..gpt import gpt_interface
import requests, json

def generate_completion_stream(model_name, prompt):
    # Requires Ollama downloaded!
    api_url = "http://localhost:11434/api/generate"  # Replace with the actual API URL if different
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
    }
    # Making a request with streaming enabled
    response = requests.post(api_url, headers=headers, data=json.dumps(payload), stream=True)
    final_response = ""
    if response.status_code == 200:
        for line in response.iter_lines():
            # Ensure the line has content
            if line:
                decoded_line = json.loads(line.decode('utf-8'))
                # Check if 'response' key exists and append its content
                if 'response' in decoded_line:
                    final_response += decoded_line['response']
                    #print(decoded_line['response'], end='', flush=True)  # Print each character without newline
        #print("\nComplete response:", final_response)
        return final_response
    else:
        print(f"Error: {response.status_code}")
        return response.status_code

def LLM_response(chat_prompt,llm_model="gpt-3.5-turbo-0125",history=[],kwargs={}): # "gpt-4-0125-preview"
    """
    get response from LLM
    """
    if llm_model.startswith('gpt-3.5') or llm_model.startswith('gpt-4') or llm_model.startswith('gpt3.5') or llm_model.startswith('gpt4'):
        gpt_interface.setup_openai('', mode='openai')
        response = gpt_interface.query_openai(chat_prompt, mode="openai", model=llm_model, max_tokens=MAX_NEW_TOKENS)
        history.append([chat_prompt, response])
    elif llm_model in ['llama3','llama2','mistral','dolphin-phi','phi','neural-chat','starling-lm','codellama','llama2-uncensored','llama2:13b','llama2:70b','orca-mini','vicuna','llava','gemma:2b','gemma:7b']:
        # use ollama instead, required ollama installed and models downloaded, https://github.com/ollama/ollama/tree/main?tab=readme-ov-file
        response = generate_completion_stream(llm_model, chat_prompt)
        history.append([chat_prompt, response])
    else:
        raise NotImplementedError
    return response, history

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    #llm_model = "dolphin-phi"
    llm_model = "gpt-3.5-turbo-0125"
    prompt = "hello"
    response, history = LLM_response(prompt, llm_model)
    print(f'User: {prompt}')
    print(f'LLM: {response}')