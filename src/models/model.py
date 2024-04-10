from configs.model_config import *
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

def LLM_model():
    if LLM_MODEL in llm_model_dict and llm_model_dict[LLM_MODEL]['platform']=='OPENAI':
        from gpt import gpt_interface
        gpt_interface.setup_openai('', mode='openai')
        llm = None
        tokenizer = None
    elif LLM_MODEL in ['llama2','mistral','dolphin-phi','phi','neural-chat','starling-lm','codellama','llama2-uncensored','llama2:13b','llama2:70b','orca-mini','vicuna','llava','gemma:2b','gemma:7b']:
        # use ollama instead, required ollama installed and models downloaded, https://github.com/ollama/ollama/tree/main?tab=readme-ov-file
        llm = None
        tokenizer = None
    else:
        raise NotImplementedError
    print('Using model: ', LLM_MODEL)
    return llm, tokenizer

def LLM_response(llm,tokenizer,chat_prompt,version="gpt-3.5-turbo-0125",history=[],kwargs={}): # "gpt-4-0125-preview"
    """
    get response from LLM
    """
    if LLM_MODEL in llm_model_dict and llm_model_dict[LLM_MODEL]['platform'] in ['OPENAI']:
        from gpt import gpt_interface
        gpt_interface.setup_openai('', mode='openai')
        response = gpt_interface.query_openai(chat_prompt, mode="openai", model=version, max_tokens=MAX_NEW_TOKENS)
        history.append([chat_prompt, response])
    elif LLM_MODEL in llm_model_dict and llm_model_dict[LLM_MODEL]['platform'] in ['HUGGINGFACE']:
        response, history = llm.chat(tokenizer, chat_prompt, history=history)
    elif LLM_MODEL in ['llama2','mistral','dolphin-phi','phi','neural-chat','starling-lm','codellama','llama2-uncensored','llama2:13b','llama2:70b','orca-mini','vicuna','llava','gemma:2b','gemma:7b']:
        # use ollama instead, required ollama installed and models downloaded, https://github.com/ollama/ollama/tree/main?tab=readme-ov-file
        response = generate_completion_stream(LLM_MODEL, chat_prompt)
        history.append([chat_prompt, response])
    else:
        raise NotImplementedError
    return response, history

def embedding_model():
    from langchain.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return embeddings

if __name__=='__main__':
    LLM_MODEL = "dolphin-phi"
    llm, tokenizer =LLM_model()
    prompt = "hello"
    response, history = LLM_response(llm,tokenizer,prompt)
    print(f'User: {prompt}')
    print(f'LLM: {response}')