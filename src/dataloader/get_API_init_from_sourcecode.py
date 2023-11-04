import inspect
import collections
import pydoc
import json
import re
import os
from sklearn.neighbors import DistanceMetric
from docstring_parser import parse
from langchain.document_loaders import BSHTMLLoader
from configs.model_config import LIB, LIB_ALIAS, CHEATSHEET, ANALYSIS_PATH, API_HTML_PATH
from dataloader.extract_function_from_sourcecode import get_returnparam_docstring
import importlib
import typing
import functools

# STEP1: get API from web
# STEP2: get docstring/parameters from source code, based on API
# STEP3: produce API calling

def process_html(html_path: str) -> str:
    """Loads HTML file content after removing excessive spaces."""
    loader = BSHTMLLoader(html_path)
    webcontent = loader.load()
    content = ' '.join([i.page_content for i in webcontent])
    content = re.sub(r'\s+', ' ', content) # remove large blanks
    return content

def get_dynamic_types():
    basic_types = [int, float, str, bool, list, tuple, dict, set, type(None)]
    useful_types_from_typing = [typing.Any, typing.Callable, typing.Union, typing.Optional, 
        typing.List, typing.Dict, typing.Tuple, typing.Set, typing.Type, typing.Collection]
    useful_types_from_collections = [collections.deque, collections.Counter,
        collections.OrderedDict, collections.defaultdict]
    useful_types_from_collections_abc = [collections.abc.Iterable, collections.abc.Iterator, collections.abc.Mapping,
        collections.abc.Sequence, collections.abc.MutableSequence, collections.abc.Set,
        collections.abc.MutableSet, collections.abc.Collection, collections.abc.MappingView, collections.abc.Callable]
    all_types = basic_types + useful_types_from_typing + useful_types_from_collections + useful_types_from_collections_abc
    return all_types

def type_to_string(t):
    type_str = str(t)
    if "<class '" in type_str:
        return type_str.split("'")[1]
    elif hasattr(t, "_name"):
        return t._name
    else:
        return type_str.split(".")[-1]

type_strings = get_dynamic_types()
typing_list = [type_to_string(t) for t in type_strings]

def expand_types(param_type):
    if is_outer_level_separator(param_type, "|"):
        types = param_type.split('|')
    elif is_outer_level_separator(param_type, " or "):
        types = param_type.split(' or ')
    else:
        types = [param_type]
    return [t.strip() for t in types]

def is_outer_level_separator(s, sep="|"):
    """
    Check if the separator (like '|' or 'or') is at the top level (not inside brackets).
    """
    bracket_count = 0
    for i, char in enumerate(s):
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
        if s[i:i+len(sep)] == sep and bracket_count == 0:
            return True
    return False

def resolve_forwardref(forward_ref_str):
    """
    Resolve a string representation of a ForwardRef type into the actual type.
    """
    namespace = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "None": None,
        "Optional": typing.Optional,
        "List": typing.List,
        "Dict": typing.Dict,
        "Union": typing.Union,
        "DistanceMetric": DistanceMetric,
    }
    try:
        return eval(forward_ref_str, namespace)
    except:
        # Special type that failed to resolve forward ref
        return forward_ref_str

def format_type_ori(annotation):
    if not annotation:
        return None
    if annotation == inspect.Parameter.empty:
        return None
    if isinstance(annotation, typing.ForwardRef):
        annotation = resolve_forwardref(annotation.__forward_arg__)
    if isinstance(annotation, str):
        if annotation in ["None", "NoneType"]:
            return None
        expanded_types = expand_types(annotation)
        if "None" in expanded_types or "NoneType" in expanded_types:
            expanded_types = [t for t in expanded_types if t not in ["None", "NoneType"]]
            if len(expanded_types) == 1:
                return f"Optional[{expanded_types[0]}]"
            return f"Optional[Union[{', '.join(expanded_types)}]]"
        if len(expanded_types) > 1:
            return f"Union[{', '.join(expanded_types)}]"
        return expanded_types[0]
    if hasattr(annotation, "__origin__"):
        formatted_args = [format_type(t) for t in getattr(annotation, '__args__', ())]
        origin_name = getattr(annotation.__origin__, "__name__", str(annotation.__origin__).replace("typing.", ""))
        origin_name = origin_name.title() if origin_name in {"list", "dict"} else origin_name
        if origin_name == "Union":
            if "None" in formatted_args or "NoneType" in formatted_args:
                formatted_args = [t for t in formatted_args if t not in ["None", "NoneType"]]
                if len(formatted_args) == 1:
                    return f"Optional[{formatted_args[0]}]"
                return f"Optional[Union[{', '.join(formatted_args)}]]"
            return f"Union[{', '.join(formatted_args)}]"
        return f"{origin_name}[{', '.join(formatted_args)}]"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")

def format_type(annotation):
    ans = format_type_ori(annotation)
    if ans:
        ans = ans.replace("NDArrayA", "ndarray[Any, dtype[Any]]")
    return ans

def is_valid_member(obj):
    return (
        callable(obj) or 
        isinstance(obj, (dict, list, tuple, set)) or  # , property
        inspect.isclass(obj) or 
        inspect.isfunction(obj) or 
        inspect.ismethod(obj) or 
        inspect.ismodule(obj) or
        (hasattr(obj, '__call__') and 'method' in str(obj))
    )

def is_unwanted_api(member):
    if inspect.isclass(member):
        if issubclass(member, BaseException):
            return True
        module_name = member.__module__
        for base_class in inspect.getmro(member):
            if base_class.__module__ != module_name:
                return True
    return False

def is_from_external_module(lib_name, member):
    """Check if a member is from an external module with robustness checks."""
    if inspect.ismodule(member):
        module_name = getattr(member, '__name__', None)
        if module_name is None:
            return False
        else:
            return lib_name not in module_name
    else:
        """ module_name = getattr(member, '__module__', None)
        if module_name is None:
            return False
        return LIB not in module_name"""
        return False

def are_most_strings_modules(api_strings):
    valid_modules = 0
    total_strings = len(api_strings)
    for api in api_strings:
        try:
            importlib.import_module(api)
            valid_modules += 1
        except:
            continue
    return valid_modules / total_strings > 0.5

def recursive_member_extraction(module, prefix, visited=None, depth=None):
    if visited is None:
        visited = set()
    members = []
    module_id = id(module)
    if module_id in visited:  # Check if module has already been visited
        return members
    visited.add(module_id)  # Mark module as visited
    for name, member in inspect.getmembers(module):
        if name.startswith('_'):
            continue
        if is_from_external_module(LIB, member):
            continue
        if inspect.isclass(member):
            if issubclass(member, Exception): #inspect.isclass(member) and 
                continue
        if inspect.isabstract(member):  # 排除抽象属性
            continue
        """if member.__module__ == 'builtins':
            continue"""
        """if is_unwanted_api(member):
            continue"""
        full_name = f"{prefix}.{name}"
        if inspect.ismodule(member):
            members.extend(recursive_member_extraction(member, full_name, visited))
        if inspect.isclass(member) and (depth is None or depth > 0):
            members.extend(recursive_member_extraction(member, full_name, visited, depth= None if depth is None else depth-1))
        else:
            members.append((full_name, member))
    return members

def get_api_type(member):
    try:
        member_str = str(member)
    except:
        member_str = ""
    if inspect.isfunction(member):
        return 'function'
    elif inspect.ismodule(member):
        return 'module'
    elif inspect.isclass(member):
        return 'class'
    elif inspect.ismethod(member) or 'method' in member_str:
        return 'method'
    elif isinstance(member, property):
        return 'property'
    elif isinstance(member, functools.partial):
        return 'functools.partial'
    elif type(member) in [int, float, str, tuple, bool, list, dict]:
        return 'constant'
    else:
        return 'unknown'# built-in

def import_member(api_string, expand=True):
    """
    Given an API string, it tries to import the module or attribute iteratively.
    1. Iteratively try importing a part of the api_string and accessing the rest.
    2. The function returns a list of (name, module/attribute) pairs.
    """
    api_parts = api_string.split('.')
    all_members = []
    for i in range(len(api_parts), 0, -1):
        module_name_attempt = '.'.join(api_parts[:i])
        member_name_sequence = api_parts[i:]
        module = None
        try:
            module = importlib.import_module(module_name_attempt)
        except ImportError:
            continue
        if module:
            current_module = module
            try:
                for sub_member_name in member_name_sequence:
                    current_module = getattr(current_module, sub_member_name)
                full_api_name = f"{module_name_attempt}{'.' if member_name_sequence else ''}{'.'.join(member_name_sequence)}"
                # If it's a full name import and the current member is a module
                if i == len(api_parts) and get_api_type(current_module) == 'module' and expand:
                    all_members.extend(recursive_member_extraction(current_module, full_api_name))
                    return all_members  # Return without including the parent module
                # If it's a function or any other non-module type
                elif inspect.isclass(current_module):
                    all_members.append((full_api_name, current_module))
                    all_members.extend(recursive_member_extraction(current_module, full_api_name, depth=1))
                else:
                    all_members.append((full_api_name, current_module))
                    return all_members
            except AttributeError:
                continue
    return all_members

def extract_return_type_and_description(return_block):
    lines = return_block.strip().split("\n")
    lines = [i for i in lines if i]
    if len(lines)>=1:
        return_type = lines[0].strip()
    else:
        return_type = ""
    if len(lines)>=2:
        description = "\n".join(lines[1:]).strip()
    else:
        description = ""
    return return_type, description

def get_docparam_from_source(web_APIs):
    success_count = 0
    failure_count = 0
    results = {}
    expand = are_most_strings_modules(web_APIs)
    print('Most api modules: ', expand, '!')
    for api_string in web_APIs:
        members = import_member(api_string, expand)
        if not members:
            print(f"Error import {api_string}: No members found!")
            failure_count += 1
            continue
        for member_name, member in members:
            api_type = get_api_type(member)
            """if api_type in ['unknown']:
                continue"""
            """if is_unwanted_api(member):
                continue"""
            try:
                if not callable(member):
                    parsed_doc = parse(member.__doc__ or "")
                    short_desc = parsed_doc.short_description or ""
                    long_desc = parsed_doc.long_description or ""
                    description = short_desc + long_desc
                    if isinstance(member, property) and member.fget:
                        if hasattr(member.fget, '__annotations__'):
                            prop_type = format_type(member.fget.__annotations__.get('return', ''))
                        else:
                            prop_type = None
                        default_value = None
                        result = {
                            'Parameters':{},
                            'Returns': {'type':prop_type,
                                        'description':None,
                                        },
                            'Value': default_value,
                            'Docstring':member.__doc__,
                            'description': description,
                            'example': "",
                            'api_type': api_type,
                        }
                    else:
                        result = {
                            'Parameters':{},
                            'Returns': {'type':type_to_string(type(member)),
                                        'description':None,
                                        },
                            'Value': str(member),
                            'Docstring':member.__doc__,
                            'description': description,
                            'example': "",
                            'api_type': api_type,
                        }
                else:
                    print('debug1')
                    current_member_doc = pydoc.getdoc(member) or ""
                    """if not current_member_doc:
                        continue"""
                    params, returns, example = get_returnparam_docstring(current_member_doc)
                    print('debug2')
                    api_short_desc = parse(current_member_doc).short_description or ""
                    api_long_desc = parse(current_member_doc).long_description or ""
                    api_desc = api_short_desc+api_long_desc
                    result_params = {}
                    print('debug3')
                    try:
                        signature = inspect.signature(member) if callable(member) else None
                    except:
                        # for built-in functions
                        signature = None
                    print('debug4')
                    if signature:
                        for param, param_type in signature.parameters.items():
                            if (param in ['self', 'kwargs', 'options', 'params', 'parameters', 'kwds', 'args']) or any(keyword in param for keyword in ['kwargs', 'kwds', 'args']) or param.startswith('*'):
                                continue
                            else:
                                print('debug5')
                                param_type_str = format_type(param_type.annotation)
                                if not param_type_str:
                                    param_type_str = params.get(param, {}).get('type', None)
                                print('debug6')
                                result_params[param] = {
                                    'type': param_type_str,
                                    'default': str(param_type.default) if param_type.default != inspect.Parameter.empty else None,
                                    'optional': True if param_type.default != inspect.Parameter.empty else False,
                                    'description': params.get(param, {}).get('description', '')
                                }
                                print('debug7')
                    else:
                        for param, param_type in params.items():
                            if (param in ['self', 'kwargs', 'options', 'params', 'parameters', 'kwds', 'args']) or any(keyword in param for keyword in ['kwargs', 'kwds', 'args']) or param.startswith('*'):
                                continue
                            else:
                                result_params[param] = {
                                    'type': param_type.get('type', None),
                                    'default': param_type.get('default', None),
                                    'optional': True if param_type.get('default', None) is not None else False,
                                    'description': param_type.get('description', '')
                                }
                    print('debug8')
                    return_type, return_desc = extract_return_type_and_description(returns)
                    print('debug9')
                    result = {
                        'Parameters': result_params,
                        'Returns': {
                            'type': format_type(signature.return_annotation) if signature and signature.return_annotation else return_type,
                            'description': returns if signature and signature.return_annotation else return_desc,
                        },
                        'Docstring': current_member_doc,
                        'description': api_desc,
                        'example': example,
                        'api_type': api_type
                    }
                    print('debug10')
                results[member_name] = result
                success_count += 1
            except Exception as e:
                print(f"Failed to process API: {member_name}, Exception: {str(e)}")
                failure_count += 1
    print('Success count:', success_count, 'Failure count:', failure_count)
    return results

def filter_optional_parameters(api_data):
    filtered_data = api_data.copy()
    for key, api_info in filtered_data.items():
        parameters = api_info.get("Parameters", {})
        filtered_parameters = {param_name: param_info for param_name, param_info in parameters.items() 
                               if not param_info.get("optional", False)
                               }
        api_info["Parameters"] = filtered_parameters
    return filtered_data

def generate_api_callings(results, basic_types=['str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set', 'any', 'List', 'Dict']):
    updated_results = {}
    for api_name, api_info in results.items():
        if api_info["api_type"] in ['function', 'method', 'class', 'functools.partial']:
            # Update the optional_value key for each parameter
            for param_name, param_details in api_info["Parameters"].items():
                param_type = param_details.get('type')
                if param_type is not None:  # Add this line to check for None
                    param_details['optional_value'] = not any(basic_type in param_type for basic_type in basic_types)
                else:
                    param_details['optional_value'] = True  # or False, depending on your default logic
            api_calling_keys = generate_api_calling_simple(api_name, api_info["Parameters"])
            api_info["api_calling"] = api_calling_keys
        else:
            api_info["api_calling"] = []
        updated_results[api_name] = api_info
    return updated_results

def generate_api_calling_simple(api_name, parameters):
    api_calling = []
    if parameters:
        param_strs = []
        for idx, (param_name, param_details) in enumerate(parameters.items(), 1):
            if param_details.get('optional_value', False):
                param_strs.append(f"{param_name}=$")
            else:
                param_strs.append(f"{param_name}=@")
        api_call = f"{api_name}({', '.join(param_strs)})"
        api_calling.append(api_call)
    else:
        api_call = f"{api_name}()"
        api_calling.append(api_call)
    return api_calling

def merge_jsons(json_list):
    merged_json = {}
    for json_dict in json_list:
        merged_json.update(json_dict)
    return merged_json

def filter_specific_apis(data):
    #### TODO
    # can modify the filtering here
    """
    Remove APIs from the data based on specific criteria:
    """
    filtered_data = {}
    filter_counts = {
        "api_type_module_constant_property": 0,
        "api_type_unknown": 0,
        "empty_docstring": 0
    }
    filter_API = {"api_type_module_constant_property": [],
        "api_type_unknown": [],
        "empty_docstring": []}
    for api, details in data.items():
        api_type = details.get('api_type', None)
        docstring = details.get('Docstring', None)
        if api_type in ["module", "constant", "property"]:
            filter_counts["api_type_module_constant_property"] += 1
            filter_API["api_type_module_constant_property"].append(api)
            continue
        if api_type == "unknown":
            filter_counts["api_type_unknown"] += 1
            filter_API["api_type_unknown"].append(api)
            continue
        # Filter by empty docstring
        if not docstring:
            filter_counts["empty_docstring"] += 1
            filter_API["empty_docstring"].append(api)
            continue
        filtered_data[api] = details
    print(filter_counts, filter_API)
    assert sum(filter_counts.values())+len(filtered_data)==len(data)
    return filtered_data

def main_get_API_init(lib_name,lib_alias,analysis_path,api_html_path=None):
    # STEP1
    if api_html_path:
        if os.path.isdir(api_html_path):
            content = ''
            for file_name in os.listdir(api_html_path):
                if file_name.endswith('html'):
                    file_path = os.path.join(api_html_path, file_name)
                    content += process_html(file_path)
        elif api_html_path.endswith('html'):
            content = process_html(api_html_path)
        else:
            print('Please double check the given html! File format error!')
            #raise ValueError
        pattern = re.compile(r"(\b\w+(\.\w+)+\b)")
        ori_content_keys = list(set([match[0] for match in pattern.findall(content)]))
        ori_content_keys = [i for i in ori_content_keys if lib_alias in i]
    else:
        ori_content_keys = [lib_alias]
    if len(ori_content_keys)==0:
        ori_content_keys = [lib_alias]
    print(ori_content_keys, ', # is', len(ori_content_keys))
    
    # STEP2
    print('Start getting docparam from source')
    results = get_docparam_from_source(ori_content_keys)
    #results = filter_optional_parameters(results)
    results = generate_api_callings(results)
    print('Get API #numbers are: ', len(results))
    tmp_results = filter_specific_apis(results)
    print('After filtering non-calling #numbers are: ', len(tmp_results))
    output_file = os.path.join(analysis_path,lib_name,"API_init.json")
    for api_name, api_info in tmp_results.items():
        api_info['relevant APIs'] = []
        api_info['type'] = 'singleAPI'
    with open(output_file, 'w') as file:
        file.write(json.dumps(tmp_results, indent=4))

def main_get_API_basic(analysis_path,cheatsheet):
    # STEP1: get API from cheatsheet, save to basic_API.json
    output_file = os.path.join(analysis_path,"API_base.json")
    result_list = []
    print('Start getting docparam from source')
    for api in cheatsheet:
        print('-'*10)
        print(f'Processing {api}!')
        results = get_docparam_from_source(cheatsheet[api])
        #results = filter_optional_parameters(results)
        results = generate_api_callings(results)
        results = {r: results[r] for r in results if r in cheatsheet[api]}
        result_list.append(results)
    outputs = merge_jsons(result_list)
    with open(output_file, 'w') as file:
        file.write(json.dumps(outputs, indent=4))

if __name__=='__main__':
    main_get_API_init(LIB,LIB_ALIAS,ANALYSIS_PATH,API_HTML_PATH)
    main_get_API_basic(ANALYSIS_PATH,CHEATSHEET)