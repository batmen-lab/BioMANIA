"""
Author: Zhengyuan Dong
Date Created: Jul 6, 2023
Last Modified: Sep 5, 2024
Description: combine unit-test into API init json data
"""

import os, ast, json, fnmatch
from typing import Dict, Any, List, Tuple, Union
from ..configs.model_config import GITHUB_PATH
from ..gpt.utils import load_json, save_json

import os, ast, json, fnmatch
from typing import Dict, Any, List, Tuple, Union

def find_test_files(directory: str):
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, 'test_*.py'):
            yield os.path.join(root, filename)

def parse_imports(filename: str) -> Dict[str, str]:
    with open(filename, 'r') as file:
        code = file.read()
    module = ast.parse(code)
    imports = {}
    for stmt in module.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                if alias.asname:
                    imports[alias.asname] = alias.name
                else:
                    imports[alias.name] = alias.name
    return imports

def resolve_function_name(name: str, imports: Dict[str, str]) -> str:
    parts = name.split('.')
    if parts[0] in imports:
        parts[0] = imports[parts[0]]
    return '.'.join(parts)

def parse_functions(filename: str, imports: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    #print(imports)
    with open(filename, 'r') as file:
        code = file.read()
    module = ast.parse(code)
    functions = {}
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            function_body = ast.unparse(node)
            function_code = '\n'.join([ast.unparse(stmt) for stmt in module.body if not isinstance(stmt, ast.FunctionDef)]) + '\n' + function_body
            for inner_node in ast.walk(node):
                if isinstance(inner_node, ast.Call):
                    function_name = resolve_function_name(ast.unparse(inner_node.func), imports)
                    if function_name not in functions:
                        functions[function_name] = []
                    functions[function_name].append({'code': function_code, 'file': filename, 'function_body': function_body})
    return functions

def generate_json_file(directory: str, output_filename: str, lib_name: str) -> None:
    output = {}
    for test_file in find_test_files(directory):
        imports = parse_imports(test_file)
        functions = parse_functions(test_file, imports)
        output.update(functions)
    output = filter_json(output, lib_name)
    output = deduplicate_json_data(output)
    save_json(output_filename, output)

def filter_json(json_data, lib_name):
    return {k: v for k, v in json_data.items() if k.startswith(lib_name + '.') or k.split('.')[0] in [lib_name]}

def append_example_to_json(json_data: Dict[str, Union[Dict[str, List[str]], str]], function_full_name: str, new_example: List[str]) -> Dict[str, Union[Dict[str, List[str]], str]]:
    if function_full_name in json_data:
        if 'example' in json_data[function_full_name]:
            json_data[function_full_name]['example'].extend(new_example)
        else:
            json_data[function_full_name]['example'] = new_example
    else:
        print(f"Function {function_full_name} not found in the json data.")
    return json_data

def test_functions_from_json(json_filename: str):
    functions = load_json(json_filename)
    success_count = 0
    failure_count = 0
    for function_name, function_info_list in functions.items():
        for function_info in function_info_list:
            try:
                exec(function_info['code'])
                success_count += 1
                function_info['error'] = "None"
            except Exception as e:
                function_info['error'] = str(e)
                failure_count += 1
    print(f"Executed {success_count} functions successfully.")
    print(f"Failed to execute {failure_count} functions.")
    save_json(json_filename, functions)

def deduplicate_json_data(json_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    deduplicated_json_data = {}
    for function_name, function_info_list in json_data.items():
        seen_function_bodies = set()
        deduplicated_info_list = []
        for function_info in function_info_list:
            function_body = function_info['function_body']
            if function_body not in seen_function_bodies:
                seen_function_bodies.add(function_body)
                deduplicated_info_list.append(function_info)
        deduplicated_json_data[function_name] = deduplicated_info_list
    return deduplicated_json_data

def merge_dicts(dict1: Dict[str, List[Dict[str, Any]]], dict2: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    no_unittest_list = []
    for key in dict1:
        if key in dict2:
            unit_examples = [func_info['function_body'] for func_info in dict1[key]]
            dict2[key].update({'unit_example': unit_examples})
        else:
            no_unittest_list.append(key)
    return dict2, no_unittest_list

def find_no_unittest(dict2: Dict[str, List[Dict[str, str]]]) -> List[str]:
    no_unittest_list = []
    for key, value in dict2.items():
        if not any('unit_example' in v for v in value):
            no_unittest_list.append(key)
    return no_unittest_list

def extract_example_from_docstring(docstring: str) -> str:
    example_marker_singular = "Example:"
    example_marker_plural = "Examples:"
    example_start = None
    if example_marker_plural in docstring:
        example_start = docstring.find(example_marker_plural) + len(example_marker_plural)
    elif example_marker_singular in docstring:
        example_start = docstring.find(example_marker_singular) + len(example_marker_singular)
    
    if example_start is not None:
        example_block = docstring[example_start:].strip()
        return example_block
    return ""

def merge_unittest_examples_into_API_init(generate_flag: bool, lib_name: str, analysis_path: str, github_path: str, API_full={}) -> None:
    if lib_name=='squidpy':
        lib_github_path = os.path.join(github_path, lib_name)
    elif lib_name=='snapatac2':
        lib_github_path = os.path.join(github_path, "snapatac2/SnapATAC2/snapatac2-python")
    else: # for 'scanpy' and 'ehrapy'
        lib_github_path = os.path.join(github_path, lib_name, lib_name)
    output_filepath = os.path.join(analysis_path,lib_name,"unittest.json")
    if not API_full:
        print(output_filepath)
        API_full = load_json(os.path.join(analysis_path,lib_name,"API_init.json"))
    """
    # re-extract docstring example for ehrapy
    if lib_name=='ehrapy':
        for api_name, api_info in API_full.items():
            if not api_info["example"]:
                docstring = api_info["Docstring"]
                if docstring:
                    extracted_example = extract_example_from_docstring(docstring)
                    if extracted_example:
                        api_info["example"] = extracted_example
        save_json(os.path.join(analysis_path, lib_name, 'API_init_tmp.json'), API_full)"""
    if generate_flag: # do not need to re-generate unittest.json
        generate_json_file(lib_github_path, output_filepath, lib_name)
    test_functions_from_json(output_filepath)
    Unittest_json = load_json(output_filepath)
    API_with_test, no_unittest_list = merge_dicts(Unittest_json, API_full)
    print('Unique in unittest api: ', len(no_unittest_list))
    no_unittest_list = find_no_unittest(API_with_test)
    print('Unique in API_init api: ',len(no_unittest_list))
    return API_with_test

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--LIB', type=str, required=True, help='PyPI tool')
    args = parser.parse_args()
    API_with_test = merge_unittest_examples_into_API_init(False, args.LIB, "data/standard_process", GITHUB_PATH)
    '''if 'scanpy.tl.ingest' in API_with_test:
        print(API_with_test['scanpy.tl.ingest'])
    for key in API_with_test: # add unit_example into examples
        if 'example' in API_with_test[key] and 'unit_example' in API_with_test[key]:
            if not API_with_test[key]['example'] and API_with_test[key]['unit_example']:
                API_with_test[key]['example'] = '\n\n'.join(API_with_test[key]['unit_example'])
                print('processed some API with unit_example')'''
