# [BIOAGENT]
# Updated httpx package version to 0.27.2
# Updated openai package version to 1.65.4
# Updated pydantic package version to 2.10.6
from openai import OpenAI
import logging
import tenacity as T
import json
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

client = OpenAI()

@T.retry(stop=T.stop_after_attempt(5), wait=T.wait_fixed(60), after=lambda s: logging.error(repr(s)))
def query_structured_output_openai(prompt: str, data_model: BaseModel, model: str = 'gpt-4o-2024-11-20') -> dict:
    res = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {'role': 'user', 'content': prompt},
        ],
        response_format=data_model,
    )
    return res.choices[0].message.parsed.model_dump()

# Test case
if __name__ == '__main__':
    class Response(BaseModel):
        answer: str
    prompt = "What is the capital of France?"
    response = query_structured_output_openai(prompt, data_model=Response, model='gpt-4o-mini-2024-07-18')
    print(response)
