import ast, re, os, json, sys, astor, asyncio, ast, argparse
from types import FunctionType, MethodType
from itertools import chain
from unittest.mock import patch
from datetime import datetime
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
from docstring_parser import parser

from configs.model_config import ANALYSIS_PATH, get_all_variable_from_cheatsheet #tut, html_dict, code
from dataloader.extract_function_from_sourcecode import get_returnparam_docstring

parser = argparse.ArgumentParser()
parser.add_argument('--LIB', type=str, default='MIOSTONE', help='PyPI tool')
args = parser.parse_args()
info_json = get_all_variable_from_cheatsheet(args.LIB)
LIB_ALIAS = info_json['LIB_ALIAS']

def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def generate_docstring_signature_with_LLM(llm, code_body):
    # LLM for generating docstring and function-signature
    prompt = f"""You are an expert in Python programming. Your task is to write well-documented function signatures and corresponding well-documented docstrings for given code segments. Interpret the function description, assigned inputs and the return variables in the docstring. The code body is: `{code_body}`
Extract the core information in 1-2 sentences and refine it.
The docstring description should only contain API description, parameter information, and return information.
Ensure that the parameter types in the docstring match the types available in the namespace, i.e., use importable types and avoid creative interpretations.
Ensure that the return information part return a tuple if it containing multiple variables. You need not to print information for each variable.
Please provide a detailed docstring in `NumPy` format. 
Please add type annotations to all the function parameters (both for required parameters and optional parameters) and return value.
Please add the necessary import statements related only to type annotations before the equation, for example, if you use np.array, you must import import numpy as np.
If the input functions belongs to a Class, ignore the class and only treat it as a function.
Response only Function Declaration, with its docstring. Substitute the main function Body as `pass`.
"""
    response = llm.predict(prompt)
    print(f'==>GPT docstring response: {response}')
    if '```python' in response:
        return response.split('```python')[-1].split('```')[0]
    else:
        return response

def generate_description_and_parameters_with_LLM(llm, code_body):
    # LLM for generating docstring and function-signature
    prompt = f'''You are an expert in Python programming. Your task is to extract information from the given code segments and recognize the parameters and summarize the docstring information. Interpret the function description, assigned inputs and the return variables in the docstring. The code body is: `{code_body}`
Extract the core information in 1-2 sentences and refine it.
The docstring description should only contain API description, parameter information, and return information.
Ensure that the parameter types in the docstring match the types available in the namespace, i.e., use importable types and avoid creative interpretations.
Ensure that the return information part return a tuple if it containing multiple variables. You need not to print information for each variable.
Please provide a detailed docstring in `NumPy` format. 
Restrict to the json Response format: 
{{
"Parameters": {{
    "x": {{
        "type": "float",
        "description": "Strictly positive rank-1 array of weights, ...",
        }},
    }}
"Returns": {{
        "type": None,
        "description": "tck : tuple\nA tuple (t,c,k) containing the vector of knots..."
    }},
"Docstring": "Find the B-spline representation of a 1-D curve.\n\n...",
}}
'''
    count=0
    max_trial=3
    while count<max_trial:
        response = llm.predict(prompt)
        print(f'==>GPT API json response: {response}')
        try:
            API_json = json.loads(response)
            assert 'Parameters' in API_json
            assert 'Returns' in API_json
            assert 'Docstring' in API_json
            API_json['description'] = API_json['Docstring'].split('\n')[0]
            return API_json
        except:
            pass
        count+=1
    return {"Parameters": {},
            "Returns": {
                    "type": None,
                    "description": ""
                },
            "Docstring": "",
            "description": ""
            }

import pydoc
import inspect
from dataloader.get_API_init_from_sourcecode import extract_return_type_and_description, format_type, get_returnparam_docstring, generate_api_callings
from docstring_parser import parse
import ast
import inspect
import types

def extract_function_from_string(function_string, module_name='dynamic_module'):
    # Parse the function string into an AST
    parsed = ast.parse(function_string)
    # Define a new module to hold the function
    module = types.ModuleType(module_name)
    # Convert the parsed code into a code object
    code = compile(parsed, filename='<string>', mode='exec')
    # Execute the code in the new module
    exec(code, module.__dict__)
    # Find the function in the module
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            return obj
    return None

def get_params_info(function_code, params_list, llm, code_body):
    try:
        # Extract the function from the string
        add_function = extract_function_from_string(function_code)
        # Use inspect to analyze the function
        signature = inspect.signature(add_function)
        parameters = list(signature.parameters.values())
        # Extract docstring
        docstring = add_function.__doc__
        # Extract parameter information
        params, returns, example = get_returnparam_docstring(docstring)
        print('returns:', returns)
        return_type, return_desc = extract_return_type_and_description(returns)
        param_info = {}
        for param in parameters:
            if param in params_list: # filter the parameters we need
                param_info[param.name] = {
                    "type": str(param.annotation.__name__) if param.annotation != inspect.Parameter.empty else str(params[param.name]['type']),
                    "default": str(param.default) if param.default!= inspect.Parameter.empty else None,
                    "optional": param.default != inspect.Parameter.empty,
                    "description": str(params[param.name]['description']) if param.name in params else ""
                }
        # Format the information in the desired output format
        output_info = {
            "Parameters": param_info,
            "Returns": {
                            "type": format_type(signature.return_annotation) if signature and signature.return_annotation else return_type,
                            "description": returns if signature and signature.return_annotation else return_desc,
                        },
            "Docstring": docstring,
            "description": docstring.split('\n')[0] if not docstring.startswith('\n')  else docstring.split('\n')[1] 
        }
        return output_info
    except:
        print("Failed to extract function information. Using LLM polishing json directly")
        API_json = generate_description_and_parameters_with_LLM(llm, code_body)
        return API_json

def main():
    with open(f'data/standard_process/{args.LIB}/API_init_prepare.json', 'r') as f:
        data = json.load(f)
    from models.model import LLM_model
    from tqdm import tqdm
    # LLM model
    llm, tokenizer = LLM_model()
    for API_key in tqdm(data):
        # only run when running on code that designed by yourself
        if (not data[API_key]['Docstring']) or (not any([data[API_key]['Parameters'][paramet]['type'] in [None, "None"] for paramet in data[API_key]['Parameters']])):
            pass
        else:
            continue
        # get code
        code = data[API_key]['code']
        # get docstring and signature
        modified_code = generate_docstring_signature_with_LLM(llm, code)
        tmp_API_key = get_params_info(modified_code, list(data[API_key]['Parameters'].keys()), llm, code)
        # fillin description and Docstring
        if not data[API_key]['description']:
            print('start copying description')
            data[API_key]["Docstring"] = tmp_API_key["Docstring"]
            data[API_key]["description"] = tmp_API_key["description"]
        # if there exist unassigned parameters, fillin parameters
        if any([data[API_key]['Parameters'][paramet]['type'] is None for paramet in data[API_key]['Parameters']]):
            print('start copying parameters')
            for param_name, param_info in tmp_API_key["Parameters"].items():
                # check which parameters need to fillin
                if param_name in data[API_key]["Parameters"]: # avoid illusion
                    if data[API_key]["Parameters"][param_name]["type"] in [None, "None"]:
                        data[API_key]["Parameters"][param_name]["type"] = param_info["type"]
                        # the default value and optional key are accurate in original code files
                    if data[API_key]["Parameters"][param_name]["description"] in [None, "None", ""]:
                        data[API_key]["Parameters"][param_name]["description"] = param_info["description"]
        # fillin return information
        if ((not data[API_key]['Returns']['type']) and (not data[API_key]['Returns']['description'])) and ((not tmp_API_key['Returns']['type']) and (not tmp_API_key['Returns']['description'])):
            print('start copying returns')
            data[API_key]["Returns"]["description"] = tmp_API_key['Returns']["description"]
            data[API_key]["Returns"]["type"] = tmp_API_key['Returns']["type"]
    # generate api calling, update "optional_value"
    data = generate_api_callings(data)
    # substitute to API_init.json directly
    with open(f'data/standard_process/{args.LIB}/API_init.json', 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()