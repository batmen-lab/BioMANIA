import pydoc, json, re, os, collections, inspect, importlib, typing, functools
from docstring_parser import parse
from configs.model_config import ANALYSIS_PATH, get_all_variable_from_cheatsheet, get_all_basic_func_from_cheatsheet
from dataloader.extract_function_from_sourcecode import get_returnparam_docstring
from typing import Dict, List, Optional, Tuple, Union, Any
from gpt.utils import save_json

# STEP1: get API from web
# STEP2: get docstring/parameters from source code, based on API
# STEP3: produce API calling

def process_html(html_path: str) -> str:
    """
    Loads HTML file content after removing excessive spaces.

    Parameters
    ----------
    html_path : str
        The file path of the HTML document to process.

    Returns
    -------
    str
        Processed HTML content as a single string with spaces normalized.
    """
    from langchain.document_loaders import BSHTMLLoader
    loader = BSHTMLLoader(html_path)
    webcontent = loader.load()
    content = ' '.join([i.page_content for i in webcontent])
    content = re.sub(r'\s+', ' ', content) # remove large blanks
    return content

def get_dynamic_types() -> List[type]:
    """
    Retrieves a list of various basic and complex data types from Python's built-in, typing,
    collections, and collections.abc modules.

    Returns
    -------
    List[type]
        A list containing types such as int, float, str, list, dict, and more specialized types
        like typing.Union, collections.deque, collections.abc.Iterable, etc.
    """
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

def type_to_string(t: type) -> str:
    """
    Convert a type to its string representation.

    Parameters
    ----------
    t : Type[Any]
        The type to be converted to string.

    Returns
    -------
    str
        The string representation of the type. If the type is a Cython function or method,
        it returns "method". If it's a class, it returns the class name. Otherwise, it returns
        the type name or its string representation.
    """
    type_str = str(t)
    if 'cython_function_or_method' in type_str:
        return "method" # label cython func/method as "method"
    if "<class '" in type_str:
        return type_str.split("'")[1]
    elif hasattr(t, "_name"):
        return t._name
    else:
        return type_str.split(".")[-1]

type_strings = get_dynamic_types()
typing_list = [type_to_string(t) for t in type_strings]

def expand_types(param_type: str) -> List[str]:
    """
    Expands a string representing a type or multiple types separated by '|' or 'or' into a list 
    of individual type strings.

    Parameters
    ----------
    param_type : str
        A string representing a single type or multiple types separated by '|' or 'or'.

    Returns
    -------
    List[str]
        A list of strings, where each string is a type extracted from the input string. 
        The types are stripped of leading and trailing whitespace.

    """
    if is_outer_level_separator(param_type, "|"):
        types = param_type.split('|')
    elif is_outer_level_separator(param_type, " or "):
        types = param_type.split(' or ')
    else:
        types = [param_type]
    return [t.strip() for t in types]

def is_outer_level_separator(s: str, sep: str = "|") -> bool:
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

def resolve_forwardref(forward_ref_str: str) -> Union[type, str]:
    """
    Resolve a string representation of a ForwardRef type into the actual type.

    Parameters
    ----------
    forward_ref_str : str
        The string that represents a forward reference.

    Returns
    -------
    Union[type, str]
        The resolved type if successful, or the original string if resolution fails.
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
    }
    try:
        return eval(forward_ref_str, namespace)
    except:
        # Special type that failed to resolve forward ref
        return forward_ref_str

def format_type_ori(annotation: Any) -> Optional[str]:
    """
    Formats a type annotation into a string representation, resolving forward references and
    handling various special cases like None, Optional, and Union types.

    Parameters
    ----------
    annotation : Any
        The type annotation to format.

    Returns
    -------
    Optional[str]
        The formatted type as a string, or None if the type cannot be formatted.
    """
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

def format_type(annotation: Any) -> Optional[str]:
    """
    Formats a type annotation into a string representation with specific handling for NumPy's
    'NDArrayA' type, converting it into 'ndarray[Any, dtype[Any]]'.

    Parameters
    ----------
    annotation : Any
        The type annotation to format.

    Returns
    -------
    Optional[str]
        The formatted type as a string, or None if the type cannot be formatted.
    """
    ans = format_type_ori(annotation)
    if ans:
        ans = ans.replace("NDArrayA", "ndarray[Any, dtype[Any]]")
    return ans

def is_valid_member(obj: Any) -> bool:
    """
    Determines whether the given object is a valid member based on its type.
    Valid members include callable objects, specific collections (dict, list, tuple, set),
    classes, functions, methods, modules, and objects with a '__call__' method that are
    identified as methods.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a valid member, False otherwise.
    """
    return (
        callable(obj) or 
        isinstance(obj, (dict, list, tuple, set)) or  # , property
        get_api_type(obj)=='class' or 
        inspect.isfunction(obj) or 
        inspect.ismethod(obj) or 
        inspect.ismodule(obj) or
        (hasattr(obj, '__call__') and 'method' in str(obj))
    )

def is_unwanted_api(member: Any) -> bool:
    """
    Determines whether a member (typically a class) is considered an unwanted API.
    Unwanted APIs are identified as either subclasses of BaseException or classes whose 
    base classes are defined in a different module than the class itself.

    Parameters
    ----------
    member : Any
        The member to check.

    Returns
    -------
    bool
        True if the member is considered unwanted, False otherwise.
    """
    if get_api_type(member)=='class':
        if issubclass(member, BaseException):
            return True
        module_name = member.__module__
        for base_class in inspect.getmro(member):
            if base_class.__module__ != module_name:
                return True
    return False

def is_from_external_module(lib_name: str, member: Any) -> bool:
    """
    Check if a member is from an external module with robustness checks.

    Parameters
    ----------
    lib_name : str
        The name of the library to check against.
    member : Any
        The member to check.

    Returns
    -------
    bool
        True if the member is from an external module, False otherwise.
    """
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

def are_most_strings_modules(api_strings: List[str]) -> bool:
    """
    Determines whether the majority of strings in a given list represent valid Python modules.
    It tries to import each string as a module and counts the successful imports.

    Parameters
    ----------
    api_strings : list
        A list of strings, each potentially representing a module name.

    Returns
    -------
    bool
        True if more than 50% of the strings in the list are valid module names, False otherwise.

    """
    valid_modules = 0
    total_strings = len(api_strings)
    for api in api_strings:
        try:
            importlib.import_module(api)
            valid_modules += 1
        except:
            continue
    return valid_modules / total_strings > 0.5

def recursive_member_extraction(module: Any, prefix: str, lib_name: str, visited: Optional[set] = None, depth: Optional[int] = None) -> List[Tuple[str, Any]]:
    """
    Recursively extracts members from a module, including classes and submodules, 
    while avoiding duplicates and unwanted members.

    Parameters
    ----------
    module : Any
        The module from which to extract members.
    prefix : str
        The prefix to append to member names.
    lib_name : str
        The name of the library being processed.
    visited : Optional[set], optional
        A set of visited modules to avoid duplicates. Default is None.
    depth : Optional[int], optional
        The depth of recursion. Default is None.

    Returns
    -------
    List[Tuple[str, Any]]
        A list of tuples containing the member name and the member itself.
    """
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
        if is_from_external_module(lib_name, member):
            continue
        if get_api_type(member)=='class':
            if issubclass(member, Exception): #inspect.isclass(member) and 
                continue
        if inspect.isabstract(member):  # remove abstract module
            continue
        """if member.__module__ == 'builtins':
            continue"""
        """if is_unwanted_api(member):
            continue"""
        full_name = f"{prefix}.{name}"
        if inspect.ismodule(member):
            members.extend(recursive_member_extraction(member, full_name, lib_name, visited))
        if get_api_type(member)=='class' and (depth is None or depth > 0):
            members.append((full_name, member))
            members.extend(recursive_member_extraction(member, full_name, lib_name, visited, depth= None if depth is None else depth-1))
        else:
            members.append((full_name, member))
    return members

def get_api_type(member: Any) -> str:
    """
    Determine the API type of a member.

    Parameters
    ----------
    member : Any
        The member to classify.

    Returns
    -------
    str
        The type of the API as a string.
    """
    try:
        member_str = str(member)
    except:
        member_str = ""
    if inspect.isclass(member): #  or (hasattr(member, '__class__') and )
        return 'class'
    elif inspect.isfunction(member):
        return 'function'
    elif inspect.ismethod(member) or 'method' in member_str:
        return 'method'
    elif inspect.ismodule(member):
        return 'module'
    elif isinstance(member, property):
        return 'property'
    elif isinstance(member, functools.partial):
        return 'functools.partial'
    elif type(member) in [int, float, str, tuple, bool, list, dict, set]:
        return 'constant'
    elif type(member)==re.Pattern:
        return 'rePattern'
    elif 'abc.ABCMeta' in str(type(member)):
        return 'class'
    elif member.__class__ is type:
        return 'class'
    elif 'cython_function_or_method' in str(type(member)):
        return 'cython'
    elif 'builtin_function_or_method' in str(type(member)):
        return 'builtin'
    elif 'getset_descriptor' in str(type(member)):
        return 'getset_descriptor'
    else:
        return 'unknown'# TypeVar

def import_member(api_string: str, lib_name: str, expand: bool = True) -> List[Tuple[str, Any]]:
    """
    Given an API string, it tries to import the module or attribute iteratively.
    1. Iteratively try importing a part of the api_string and accessing the rest.
    2. The function returns a list of (name, module/attribute) pairs.

    Parameters
    ----------
    api_string : str
        The API string to process.
    lib_name : str
        The library name to consider during import.
    expand : bool, optional
        Whether to expand the import to include all sub-members. Default is True.

    Returns
    -------
    List[Tuple[str, Any]]
        A list of tuples containing the full API name and the imported module or attribute.
    """
    api_parts = api_string.split('.')
    all_members = []
    for i in range(len(api_parts), 0, -1):
        module_name_attempt = '.'.join(api_parts[:i])
        member_name_sequence = api_parts[i:]
        module = None
        try:
            # for python
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
                    all_members.append((full_api_name, current_module))
                    all_members.extend(recursive_member_extraction(current_module, full_api_name, lib_name))
                    return all_members  # Return without including the parent module
                # If it's a function or any other non-module type
                # TODO: modified here 231129, notify the changes
                #elif inspect.isclass(current_module):
                elif get_api_type(current_module)=='class':
                    all_members.append((full_api_name, current_module))
                    all_members.extend(recursive_member_extraction(current_module, full_api_name, lib_name, depth=1))
                else:
                    all_members.append((full_api_name, current_module))
                    return all_members
            except Exception as e:
                print('import member Error:', e)
                continue
    return all_members

def extract_return_type_and_description(return_block: str) -> Tuple[str, str]:
    """
    Extracts the return type and description from a formatted string block.

    Parameters
    ----------
    return_block : str
        The string block containing the return type and its description.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the return type and its description.
    """
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

def get_docparam_from_source(web_APIs: List[str], lib_name: str, expand: Optional[bool] = None) -> Dict[str, Any]:
    """
    Retrieves documentation parameters from source code based on a list of web APIs.

    Parameters
    ----------
    web_APIs : List[str]
        A list of web API strings to process.
    lib_name : str
        The library name to use for context.
    expand : Optional[bool], optional
        Whether to expand the API search to include sub-modules. Default is None.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the processed API details.
    """
    success_count = 0
    failure_count = 0
    results = {}
    if expand:
        pass
    else:
        expand = are_most_strings_modules(web_APIs)
    print('==>Is api page html reliable: ', not expand, '!')
    for api_string in web_APIs:
        members = import_member(api_string, lib_name, expand)
        """print((len(members)>1), expand)
        if len(members)>1 and not expand:
            print(members, api_string)
        assert len(members)<=1 or expand"""
        if not members:
            print(f"Error import {api_string}: No members found!")
            failure_count += 1
            continue
        #print('!get '+str(len(members))+" members")
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
                    if short_desc and long_desc:
                        description = short_desc+long_desc
                    elif short_desc:
                        description = short_desc
                    elif long_desc:
                        description = long_desc
                    else:
                        description = ""
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
                    current_member_doc = pydoc.getdoc(member) or ""
                    """if not current_member_doc:
                        continue"""
                    params, returns, example = get_returnparam_docstring(current_member_doc)
                    api_short_desc = parse(current_member_doc).short_description or ""
                    api_long_desc = parse(current_member_doc).long_description or ""
                    if api_short_desc and api_long_desc:
                        api_desc = api_short_desc+api_long_desc
                    elif api_short_desc:
                        api_desc = api_short_desc
                    elif api_long_desc:
                        api_desc = api_long_desc
                    else:
                        api_desc = ""
                    result_params = {}
                    try:
                        signature = inspect.signature(member) if callable(member) else None
                    except:
                        # for built-in functions
                        signature = None
                    if signature:
                        for param, param_type in signature.parameters.items():
                            if (param in ['self', 'kwargs', 'options', 'params', 'parameters', 'kwds', 'args']) or any(keyword in param for keyword in ['kwargs', 'kwds', 'args']) or param.startswith('*'):
                                continue
                            else:
                                param_type_str = format_type(param_type.annotation)
                                if not param_type_str:
                                    param_type_str = params.get(param, {}).get('type', None)
                                result_params[param] = {
                                    'type': param_type_str,
                                    'default': str(param_type.default) if param_type.default != inspect.Parameter.empty else None,
                                    'optional': True if param_type.default != inspect.Parameter.empty else False,
                                    'description': params.get(param, {}).get('description', '')
                                }
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
                    return_type, return_desc = extract_return_type_and_description(returns)
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
                results[member_name] = result
                success_count += 1
            except Exception as e:
                print(f"Failed to process API: {member_name}, Exception: {str(e)}")
                failure_count += 1
    print('Success count:', success_count, 'Failure count:', failure_count)
    return results

def filter_optional_parameters(api_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters out optional parameters from the API data.

    Parameters
    ----------
    api_data : Dict[str, Any]
        The API data to filter.

    Returns
    -------
    Dict[str, Any]
        The filtered API data with optional parameters removed.
    """
    filtered_data = api_data.copy()
    for key, api_info in filtered_data.items():
        parameters = api_info.get("Parameters", {})
        filtered_parameters = {param_name: param_info for param_name, param_info in parameters.items() 
                               if not param_info.get("optional", False)
                               }
        api_info["Parameters"] = filtered_parameters
    return filtered_data

def generate_api_callings(results: Dict[str, Any], basic_types: List[str] = ['str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set', 'any', 'List', 'Dict']) -> Dict[str, Any]:
    """
    Generates API callings for each API in the results.

    Parameters
    ----------
    results : Dict[str, Any]
        The API results to process.
    basic_types : List[str], optional
        A list of basic type strings used for filtering. Default includes common data types.

    Returns
    -------
    Dict[str, Any]
        The results with updated API callings.
    """
    updated_results = {}
    for api_name, api_info in results.items():
        if api_info["api_type"]: #  in ['function', 'method', 'class', 'functools.partial']
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

def generate_api_calling_simple(api_name: str, parameters: Dict[str, Any]) -> List[str]:
    """
    Generates a simple API calling structure based on the API name and its parameters.

    Parameters
    ----------
    api_name : str
        The name of the API to generate calling for.
    parameters : Dict[str, Any]
        The parameters of the API.

    Returns
    -------
    List[str]
        A list of strings representing simple API callings.
    """
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

def merge_jsons(json_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merges multiple JSON dictionaries into a single dictionary.

    Parameters
    ----------
    json_list : List[Dict[str, Any]]
        A list of JSON dictionaries to merge.

    Returns
    -------
    Dict[str, Any]
        The merged JSON dictionary.
    """
    merged_json = {}
    for json_dict in json_list:
        merged_json.update(json_dict)
    return merged_json

def filter_specific_apis(data: Dict[str, Any], lib_name: str) -> Dict[str, Any]:
    """
    Filters out specific APIs from the data based on defined criteria.

    Parameters
    ----------
    data : Dict[str, Any]
        The API data to filter.
    lib_name : str
        The library name to use for filtering context.

    Returns
    -------
    Dict[str, Any]
        The filtered API data.
    """
    filtered_data = {}
    filter_counts = {
        "api_type_module_constant_property_getsetdescriptor": 0,
        "api_type_unwant_func": 0,
        "api_type_unknown": 0,
        "empty_docstring": 0,
        "empty_input_output":0,
        "external_lib_function":0,
    }
    filter_API = {"api_type_module_constant_property_getsetdescriptor": [],
        "api_type_unwant_func": [],
        "api_type_unknown": [],
        "empty_docstring": [],
        "empty_input_output":[],
        "external_lib_function":[],}
    for api, details in data.items():
        api_type = details['api_type']
        docstring = details['Docstring']
        parameters = details['Parameters']
        Returns_type = details['Returns']['type']
        Returns_description = details['Returns']['description']
        if api_type in ["module", "constant", "property", "getset_descriptor"]:
            filter_counts["api_type_module_constant_property_getsetdescriptor"] += 1
            filter_API["api_type_module_constant_property_getsetdescriptor"].append(api)
            continue
        # We filter out `cython` type API, because some are compiled functions.
        """if api_type in ["builtin", 'functools.partial', "rePattern", "cython"]:
            filter_counts["api_type_unwant_func"] += 1
            filter_API["api_type_unwant_func"].append(api)
            continue"""
        # We filter out the api_type that we can not parsed
        """if api_type in ["unknown"]:
            filter_counts["api_type_unknown"] += 1
            filter_API["api_type_unknown"].append(api)
            continue"""
        # Filter by empty docstring
        if (not docstring) or (not details['description']):
            filter_counts["empty_docstring"] += 1
            filter_API["empty_docstring"].append(api)
            continue
        # These API is not our targeted API. We filter it because there are too many `method` type API in some libs.
        # TODO: We might include them in future for robustness.
        """if (not parameters) and (not Returns_type) and (not Returns_description):
            filter_counts["empty_input_output"] += 1
            filter_API["empty_input_output"].append(api)
            continue"""
        # Remove API that imported from external lib 
        # (used for github repo 2 biomania app only)
        remove_extra_API = False
        if remove_extra_API:
            tmp_all_members = import_member(api, lib_name, expand=True)
            #print(api, tmp_all_members)
            try:
                tmp_all_members[0][1].__modules__
                print(lib_name, tmp_all_members[0][1].__modules__)
                if lib_name not in str(tmp_all_members[0][1].__modules__):
                    filter_counts["external_lib_function"] += 1
                    filter_API["external_lib_function"].append(api)
                    continue
            except:
                pass
        filtered_data[api] = details
    print('==>Filtering APIs!')
    print(json.dumps(filter_counts,indent=4))
    print(json.dumps(filter_API,indent=4))
    assert sum(filter_counts.values())+len(filtered_data)==len(data)
    return filtered_data

def parse_content_list(content_list: List[str]) -> List[str]:
    """
    Parses a list of content strings to handle various separators and formats.

    Parameters
    ----------
    content_list : List[str]
        The list of content strings to parse.

    Returns
    -------
    List[str]
        The parsed list of strings.
    """
    parsed_list = []
    for item in content_list:
        # Split by comma first, then split each resulting item by spaces and strip whitespace
        sub_items = item.strip().split(',')
        for sub_item in sub_items:
            parsed_list.extend(sub_item.strip().split())
    return [item for item in parsed_list if item]  # Remove any empty strings


def main_get_API_init(lib_name: str, lib_alias: str, analysis_path: str, api_html_path: Optional[str] = None, api_txt_path: Optional[str] = None) -> None:
    """
    Main function to initialize API data collection from either HTML or text files.

    Parameters
    ----------
    lib_name : str
        The library name.
    lib_alias : str
        The alias for the library.
    analysis_path : str
        The path to save analysis results.
    api_html_path : Optional[str], optional
        The path to HTML API documentation. Default is None.
    api_txt_path : Optional[str], optional
        The path to a text file containing API documentation. Default is None.
    """
    # STEP1
    if api_txt_path:
        content_list = []
        try:
            with open(api_txt_path, 'r', encoding='latin') as file:
                content_list = file.readlines()
            ori_content_keys = parse_content_list(content_list)
        except FileNotFoundError:
            print(f"Error: File '{api_txt_path}' not found.")
        except Exception as e:
            print(f"Error: {e}")
    elif api_html_path:  # if you provide API page
        if os.path.isdir(api_html_path):
            print('processing API subset dir')
            content = ''
            for file_name in os.listdir(api_html_path):
                print('sub API file: ', file_name)
                if file_name.endswith('html'):
                    file_path = os.path.join(api_html_path, file_name)
                    content += process_html(file_path)
        elif api_html_path.endswith('html'):
            content = process_html(api_html_path)
        else:
            print('Please double check the given html! File format error!')
            raise ValueError
        pattern = re.compile(r"(\b\w+(\.\w+)+\b)")
        ori_content_keys = list(set([match[0] for match in pattern.findall(content)]))
        ori_content_keys = [i for i in ori_content_keys if lib_alias in i]
    else: # if not, start from lib module
        ori_content_keys = [lib_alias]
    if len(ori_content_keys)==0:
        ori_content_keys = [lib_alias]
    print(ori_content_keys, ', # is', len(ori_content_keys))
    # STEP2
    print('Start getting docparam from source')
    results = get_docparam_from_source(ori_content_keys, lib_name)
    #results = filter_optional_parameters(results)
    results = generate_api_callings(results)
    print('Get API #numbers are: ', len(results))
    tmp_results = filter_specific_apis(results, lib_name)
    print('After filtering non-calling #numbers are: ', len(tmp_results))
    # output_file = os.path.join(analysis_path,lib_name,"API_init.json")
    os.makedirs(os.path.join('data','standard_process',lib_name), exist_ok=True)
    output_file = os.path.join('data','standard_process',lib_name,"API_init.json")
    for api_name, api_info in tmp_results.items():
        api_info['relevant APIs'] = []
        api_info['type'] = 'singleAPI'
    save_json(output_file, tmp_results)

def main_get_API_basic(cheatsheet: Dict[str, List[str]]) -> None:
    """
    Main function to extract basic API information from a cheatsheet.

    Parameters
    ----------
    analysis_path : str
        The path to save analysis results.
    cheatsheet : Dict[str, List[str]]
        A dictionary containing API information.
    """
    # STEP1: get API from cheatsheet, save to basic_API.json
    #output_file = os.path.join(analysis_path,"API_base.json")
    os.makedirs(os.path.join('data','standard_process',"base"), exist_ok=True)
    output_file = os.path.join('data','standard_process',"base", "API_init.json")
    result_list = []
    print('Start getting docparam from source')
    for api in cheatsheet:
        print('-'*10)
        print(f'Processing {api}!')
        results = get_docparam_from_source(cheatsheet[api], api)
        #results = filter_optional_parameters(results)
        results = generate_api_callings(results)
        results = {r: results[r] for r in results if r in cheatsheet[api]}
        result_list.append(results)
    outputs = merge_jsons(result_list)
    for api_name, api_info in outputs.items():
        api_info['relevant APIs'] = []
        api_info['type'] = 'singleAPI'
    with open(output_file, 'w') as file:
        file.write(json.dumps(outputs, indent=4))

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--LIB', type=str, help='PyPI tool')
    parser.add_argument('--api_txt_path', type=str, default=None, help='Your self-defined api txt path')
    args = parser.parse_args()
    info_json = get_all_variable_from_cheatsheet(args.LIB)
    LIB_ALIAS, API_HTML, TUTORIAL_GITHUB, API_HTML_PATH = [info_json[key] for key in ['LIB_ALIAS', 'API_HTML', 'TUTORIAL_GITHUB','API_HTML_PATH']]
    CHEATSHEET = get_all_basic_func_from_cheatsheet()
    main_get_API_init(args.LIB,LIB_ALIAS,ANALYSIS_PATH,API_HTML_PATH, api_txt_path=args.api_txt_path)
    # currently we do not need the API_base.json
    main_get_API_basic(CHEATSHEET)
