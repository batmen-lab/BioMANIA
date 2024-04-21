import openai
import logging
import tenacity as T
import os
from dotenv import load_dotenv
from ..gpt.utils import load_json

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
        secrets = load_json(fname)
        openai.api_type = "azure"
        openai.api_base = secrets['MS_ENDPOINT']
        openai.api_key = secrets['MS_KEY']
    return secrets

@T.retry(stop=T.stop_after_attempt(5), wait=T.wait_fixed(60), after=lambda s: logging.error(repr(s)))
def query_openai(prompt, mode='azure', model='gpt-35-turbo', **kwargs):
    # 240127: update openai version
    if mode == 'openai':
        response = openai.chat.completions.create(model=model,
                                             messages=[{'role': 'user', 'content': prompt}],
                                             **kwargs
                                             )
    else:
        response = openai.chat.completions.create(
            deployment_id=model,
            messages=[{'role': 'user', 'content': prompt}],
            **kwargs,
        )
    return response.choices[0].message.content

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))
