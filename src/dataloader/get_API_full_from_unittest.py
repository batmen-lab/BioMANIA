import os, ast, json, fnmatch
from typing import Dict, Any, List, Tuple, Union
from ..configs.model_config import LIB, ANALYSIS_PATH, GITHUB_PATH
from ..gpt.utils import load_json, save_json

def find_test_files(directory: str):
    for root, dirnames, filenames in os.walk(directory):
        if os.path.basename(root) == "tests":
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
                    # If the function name is not in the dict, add an empty list
                    if function_name not in functions:
                        functions[function_name] = []
                    # Append the function information to the list
                    functions[function_name].append({'code': function_code, 'file': filename, 'function_body': function_body})
    return functions

def generate_json_file(directory: str, output_filename: str, lib_name: str) -> None:
    """
    Generates a JSON file containing parsed test functions data.

    Parameters:
    - directory (str): Directory path containing test files.
    - output_filename (str): File path for the output JSON file.
    - lib_name (str): Name of the library.

    This function parses test files in the specified directory, extracts functions and imports,
    filters and deduplicates the data, and saves it as a JSON file.
    """
    output = {}
    for test_file in find_test_files(directory):
        imports = parse_imports(test_file)
        functions = parse_functions(test_file, imports)
        output.update(functions)
    output = filter_json(output, lib_name)
    output = deduplicate_json_data(output)
    save_json(output_filename, output)

def filter_json(json_data, lib_name):
    return {k: v for k, v in json_data.items() if k.startswith(lib_name + '.')}

def append_example_to_json(json_data: Dict[str, Union[Dict[str, List[str]], str]], function_full_name: str, new_example: List[str]) -> Dict[str, Union[Dict[str, List[str]], str]]:
    """
    Appends a new example to a JSON data entry or creates a new one if it doesn't exist.

    Parameters:
    - json_data (Dict[str, Union[Dict[str, List[str]], str]]): A dictionary containing keys associated with either a dictionary
      with a key 'example' holding a list of strings or a string value.
    - function_full_name (str): The name of the function to which the example is appended.
    - new_example (List[str]): List of strings representing the new example.

    Returns:
    - Dict[str, Union[Dict[str, List[str]], str]]: Updated JSON data with the appended or newly created example.
    """
    if function_full_name in json_data:
        if 'example' in json_data[function_full_name]:
            json_data[function_full_name]['example'].extend(new_example)
        else:
            json_data[function_full_name]['example'] = new_example
    else:
        print(f"Function {function_full_name} not found in the json data.")
    return json_data

def test_functions_from_json(json_filename: str):
    # guarantee it is executable
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
    """
    Removes duplicate function bodies within the JSON data.

    Parameters:
    - json_data (Dict[str, List[Dict[str, Any]]]): A dictionary containing keys associated with lists of dictionaries.

    Returns:
    - Dict[str, List[Dict[str, Any]]]: Deduplicated JSON data without duplicate function bodies.
    """
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
    """
    Merges information from dict1 into dict2, adding unit examples based on matching keys.

    Parameters:
    - dict1 (Dict[str, List[Dict[str, Any]]]): A dictionary containing keys associated with lists of dictionaries.
    - dict2 (Dict[str, Any]): A dictionary to merge data into.

    Returns:
    - Tuple[Dict[str, Any], List[str]]: A tuple containing the merged dictionary (dict2) and a list of keys from dict1
      that do not exist in dict2.
    """
    no_unittest_list = []
    for key in dict1:
        if key in dict2:
            unit_examples = [func_info['function_body'] for func_info in dict1[key]]
            dict2[key].update({'unit_example': unit_examples})
        else:
            no_unittest_list.append(key)
    return dict2, no_unittest_list

def filter_no_error(data):
    return {key: [val for val in vals if val.get('error') == 'None'] for key, vals in data.items()}

def find_no_unittest(dict2: Dict[str, List[Dict[str, str]]]) -> List[str]:
    """
    Finds keys in the dictionary without 'unit_example' within their values.

    Parameters:
    - dict2 (Dict[str, List[Dict[str, str]]]): A dictionary containing keys associated with lists of dictionaries.

    Returns:
    - List[str]: A list of keys that don't have 'unit_example' within their associated values.
    """
    no_unittest_list = []
    for key, value in dict2.items():
        if not any('unit_example' in v for v in value):
            no_unittest_list.append(key)
    return no_unittest_list

def merge_unittest_examples_into_API_init(lib_name: str, analysis_path: str, github_path: str) -> None:
    """
    Merges unittest examples into the existing API initialization, filtering unique APIs.

    Parameters:
    - lib_name (str): Name of the library.
    - analysis_path (str): Path to the analysis directory.
    - github_path (str): Path to the GitHub directory.

    This function merges unittest examples into the existing API initialization data. 
    It loads unittest examples, filters the ones with no errors, merges them with the current API initialization,
    and saves the updated API data back to the file.
    """
    lib_github_path = os.path.join(github_path, lib_name)
    output_filepath = os.path.join(analysis_path,lib_name,"unittest.json")
    API_full = load_json(os.path.join(analysis_path,lib_name,"API_init.json"))
    print('Loading api examples from unittest!')
    generate_json_file(lib_github_path, output_filepath, lib_name)
    test_functions_from_json(output_filepath)
    Unittest_json = load_json(output_filepath)
    Unittest_json =  filter_no_error(Unittest_json)
    API_with_test, no_unittest_list = merge_dicts(Unittest_json, API_full)
    print('Unique in unittest api: ', len(no_unittest_list))
    no_unittest_list = find_no_unittest(API_with_test)
    print('Unique in API_init api: ',len(no_unittest_list))
    save_json(os.path.join(analysis_path,lib_name, 'API_init.json'), API_with_test)

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    merge_unittest_examples_into_API_init(LIB, ANALYSIS_PATH, GITHUB_PATH)
