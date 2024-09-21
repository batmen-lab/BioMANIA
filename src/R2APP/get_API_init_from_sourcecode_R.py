"""
Author: Zhengyuan Dong
Email: zydong122@gmail.com
Description: The script contains functions to extract API information from R source code.
"""
import pydoc, argparse, json, re, os, collections, inspect, importlib, typing, functools
from docstring_parser import parse
from langchain.document_loaders import BSHTMLLoader
from configs.model_config import ANALYSIS_PATH, get_all_variable_from_cheatsheet, get_all_basic_func_from_cheatsheet
from dataloader.extract_function_from_sourcecode import get_returnparam_docstring

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

parser = argparse.ArgumentParser()
parser.add_argument('--LIB', type=str, default='ggplot2', help='PyPI tool')
args = parser.parse_args()
#info_json = get_all_variable_from_cheatsheet(args.LIB)
#LIB_ALIAS, API_HTML, TUTORIAL_GITHUB, API_HTML_PATH = [info_json[key] for key in ['LIB_ALIAS', 'API_HTML', 'TUTORIAL_GITHUB','API_HTML_PATH']]
LIB = args.LIB
#CHEATSHEET = get_all_basic_func_from_cheatsheet()

# STEP1: get API from R library
# STEP2: get docstring/parameters from source code, based on API
# STEP3: produce API calling

def parse_r_docstring_to_json(docstring, keywords, func_name):
    """
    Parses an R function's docstring into a JSON object containing distinct sections with improved block detection logic.

    :param docstring: The docstring to be parsed.
    :param keywords: A list of keywords that are used to identify different sections in the docstring.
    :param func_name: The name of the function for accurate block identification.
    :return: A JSON object with each section keyed by the keyword.
    """
    # Remove introductory content
    intro_marker = "Wrapper around an R function.\n\nThe docstring below is built from the R documentation."
    processed_docstring = docstring.replace(intro_marker, "").strip()
    # Splitting the docstring into blocks based on keywords
    sections = {}
    current_block = []
    current_keyword = None
    lines = processed_docstring.split('\n')
    for line in lines:
        # Check if line starts with any of the keywords
        if any(line.lower().startswith(keyword) for keyword in keywords):
            if current_block:
                # filter empty lines and pure ---- lines from current_block
                current_block = [line for line in current_block if line.strip() and ('---' not in line.strip())]
                # Add the current block to the sections if it exists
                sections[current_keyword] = '\n'.join(current_block)
                current_block = []
            # Extracting the keyword and ensuring it doesn't have trailing characters like '('
            current_keyword = line.lower().split(' ')[0].split('(')[0]
        elif current_keyword:
            current_block.append(line)
    # Add the last block if it exists
    if current_block:
        # filter empty lines and pure ---- lines from current_block
        current_block = [line for line in current_block if line.strip() and ('---' not in line.strip())]
        sections[current_keyword] = '\n'.join(current_block)
    # Ensure the function name block starts correctly
    if not sections.get(func_name, '').startswith(func_name):
        sections[func_name] = func_name + '(\n' + sections.get(func_name, '')
    return sections

def extract_parameters_from_function(function_input):
    """
    Extracts parameter names and their default values from a given function input string.

    :param function_input: A string representing the function input.
    :return: A dictionary where keys are parameter names and values are their default values.
    """
    parameters = {}
    # Splitting the input string into individual parameters
    params = function_input.split(',')
    for param in params:
        # Splitting each parameter at the '=' character
        parts = param.split('=')
        param_name = parts[0].strip()
        if '(' in param_name:
            param_name = param_name.split('(')[-1]
        if ')' in param_name:
            param_name = param_name.split(')')[0]
        if '\n' in param_name:
            param_name = param_name.replace('\n','')
        param_name = param_name.strip()
        if param_name:
            default_value = parts[1].strip() if len(parts) > 1 else "None"
            parameters[param_name] = default_value
    return parameters

def extract_parameters_from_args_section(args_section, param_json):
    """
    Extracts parameter descriptions from the 'Args' section of a docstring, given a list of parameter names.

    :param args_section: The 'Args' section of the docstring.
    :param param_list: A list of parameter names to look for.
    :return: A dictionary with parameter names as keys and their descriptions as values.
    """
    parameters = {param: {"type": "unknown",
                        "default": param_json[param] if param_json[param]!='rinterface.NULL' else None,
                        "optional": True if param_json[param] not in [None, 'None'] else False,
                        "description": "",
                        "optional_value": False} for param in param_json}  # Initialize all parameters with empty descriptions
    current_param = None
    found_ellipsis = True
    for line in args_section.split('\n'):
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        # Check if the line starts with any of the parameter names
        if any(line.startswith(param + ":") or line.startswith(param + " :") or line.startswith(param.replace('_','.') + ":") or line.startswith(param.replace('_','.') + " :") for param in param_json):
            # Extract the parameter name and its description
            current_param, desc = line.split(':', 1)
            current_param = current_param.strip().replace('.','_')
            parameters[current_param]['description'] = desc.strip()
        elif current_param and current_param in parameters:
            # Continue appending description for the current parameter
            parameters[current_param]['description'] += " " + line
    # Mark all parameters after '___' as optional
    if found_ellipsis:
        ellipsis_reached = False
        for param in parameters:
            if param in ['___','...']:
                ellipsis_reached = True
            if ellipsis_reached:
                parameters[param]['optional'] = True
    if '___' in parameters:
        del parameters['___']
    if '...' in parameters:
        del parameters['...']
    if ')' in parameters:
        del parameters[')']
    return parameters

def parse_r_docstring_to_json_structure(docstring, func_name):
    """
    Parses an R function's docstring into a structured JSON format with enhanced parameter parsing.

    :param docstring: The docstring to be parsed.
    :param func_name: The name of the function.
    :return: A JSON object with structured information from the docstring.
    """
    # Keywords for splitting the docstring
    keywords = ["description", "usage", "value", "values", "arguments", "args", "params", "parameters", "details", "see also", "examples", "returns", func_name]
    # Splitting the docstring into sections
    sections = parse_r_docstring_to_json(docstring, keywords, func_name)
    param_json = extract_parameters_from_function(sections.get(func_name, ''))
    if 'args:' in sections:
        arg_name = 'args:'
    elif 'arguments:' in sections:
        arg_name = 'arguments:'
    elif 'params:' in sections:
        arg_name = 'params:'
    elif 'parameters:' in sections:
        arg_name = 'parameters:'
    else:
        arg_name='placeholder'
    if 'value:' in sections:
        description_section = sections.get('value:', '')
    elif 'values:' in sections:
        description_section = sections.get('values:', '')
    else:
        description_section = sections.get('placeholder', '')
    examples_section = sections.get('examples:','')
    params = extract_parameters_from_args_section(sections.get(arg_name, ''), param_json)
    # Structuring the JSON output
    json_structure = {
        "Parameters": params,
        "Returns": {"type": "unknown", "description": description_section},
        "Docstring": docstring,
        "description": sections.get('description', ''),
        "api_type": "function",
        "api_calling": [f"{func_name}({', '.join([f'{param}=@' if details['optional'] else f'{param}=$' for param, details in params.items()])})"],
        "relevant APIs": [],
        "example": examples_section,
        "type": "singleAPI"
    }
    return json_structure

from rpy2.rinterface_lib.embedded import RRuntimeError
def get_r_function_doc_rd2txt(package_name, function_name):
    # (deprecate) can not work under python environment
    try:
        tools = importr('tools')
        rd_path = robjects.r['system.file']("help", "{0}.Rd".format(function_name), package=package_name)[0]
        if not rd_path:
            return "Documentation file not found for function: {}".format(function_name)
        doc_text = robjects.r['capture.output'](tools.Rd2txt(rd_path))
        return '\n'.join(doc_text)
    except RRuntimeError as e:
        return "Runtime error: {}".format(e)
    except Exception as e:
        return "Error: {}".format(e)

def main_get_API_init():
    # import R
    ggplot2 = importr('ggplot2')
    # get functions name
    package_functions = [func for func in dir(ggplot2) if not func.startswith("_")]
    api_info = {}
    for func in package_functions:
        try:
            r_func = getattr(ggplot2, func, None)
            if r_func:
                doc_string = r_func.__doc__
                # doc_string = robjects.r('capture.output(??("{function_name}", package="{package_name}"))'.format(function_name=func, package_name=args.LIB))
                if doc_string:
                    # Parsing the provided docstring with the new function
                    parsed_json_structure = parse_r_docstring_to_json_structure(doc_string, func)
                    print(parsed_json_structure)
                    api_info[func] = parsed_json_structure
        except:
            pass
    os.makedirs(f'./data/standard_process/{args.LIB}', exist_ok=True)
    with open(f'./data/standard_process/{args.LIB}/API_init.json', 'w') as file:
        file.write(json.dumps(api_info, indent=4))
        
if __name__=='__main__':
    main_get_API_init()
    # currently we do not need the API_base.json
    #main_get_API_basic(ANALYSIS_PATH,CHEATSHEET)
