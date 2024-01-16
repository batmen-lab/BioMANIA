"""
Author: Zhengyuan Dong
Date Created: January 16, 2024
Last Modified: January 16, 2024
Description: prompts for composite docstring/name generation
"""

def build_prompt_for_composite_docstring(API_description, func_inputs, func_outputs, description_text):
    return f"""
Write a concise docstring for an invisible function in Python, focusing solely on its core functionality derived from the sequential composition of sub APIs.
- API Description: {API_description}
- Parameters: {func_inputs}
- Returns: {func_outputs}
- Additional Description: {description_text}
Craft a 1-2 sentence docstring that extracts and polishes the core information. The response should be in reStructuredText format, excluding specific API names and unprofessional terms. Remember to use parameter details only to refine the core functionality explanation, not for plain input/output information.
"""

def build_prompt_for_composite_name(sub_API_names, llm_docstring):
    return f"""Your task is to suggest an appropriate name for the given invisible function:
- Here are the sub API used together with function's docstring, please consider the API name to generate function name. sub API names: {sub_API_names}, 
function docstring: ```{llm_docstring}```
- Your name should consist of 4 to 5 keywords that combined with `_`, name should be recognizable and contain as much information as you can in keywords, and should display API information in a sequential order.
Your Response format: {{'func_name': (your designed function name)}}
Please do not include other information except for response format.
"""
