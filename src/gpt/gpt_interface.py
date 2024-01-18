import json
import openai
import requests
import logging
import tenacity as T
import os
from dotenv import load_dotenv

def setup_openai(fname, mode='azure'):
    assert mode in {'openai', 'azure'}
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-test')
    if mode == 'openai':
        openai.api_type = "open_ai"
        openai.api_base = "https://api.openai.com/v1"
        openai.api_key = OPENAI_API_KEY
        secrets = None
    else:
        #openai.api_version = "2023-03-15-preview"
        with open(fname) as f:
            secrets = json.load(f)
        openai.api_type = "azure"
        openai.api_base = secrets['MS_ENDPOINT']
        openai.api_key = secrets['MS_KEY']
    return secrets

@T.retry(stop=T.stop_after_attempt(5), wait=T.wait_fixed(60), after=lambda s: logging.error(repr(s)))
def query_openai(prompt, mode='azure', model='gpt-35-turbo', **kwargs):
    if mode == 'openai':
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            **kwargs
        )
    else:
        response = openai.ChatCompletion.create(
            deployment_id=model,
            messages=[{'role': 'user', 'content': prompt}],
            **kwargs,
        )
    return response['choices'][0]['message']['content']