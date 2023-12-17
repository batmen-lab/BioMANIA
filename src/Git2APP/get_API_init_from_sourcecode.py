import pydoc, argparse, json, re, os, collections, inspect, importlib, typing, functools
from typing import Any, Type, List
from docstring_parser import parse
from langchain.document_loaders import BSHTMLLoader
from configs.model_config import ANALYSIS_PATH, get_all_variable_from_cheatsheet, get_all_basic_func_from_cheatsheet
from dataloader.extract_function_from_sourcecode import get_returnparam_docstring
from dataloader.get_API_init_from_sourcecode import get_dynamic_types, process_html, type_to_string, expand_types, is_outer_level_separator, resolve_forwardref, format_type_ori, format_type, is_valid_member, is_unwanted_api, is_from_external_module, are_most_strings_modules, recursive_member_extraction, get_api_type, import_member, extract_return_type_and_description, filter_optional_parameters, generate_api_callings, generate_api_calling_simple, merge_jsons, main_get_API_basic


# STEP1: get API from web
# STEP2: get docstring/parameters from source code, based on API
# STEP3: produce API calling

type_strings = get_dynamic_types()
typing_list = [type_to_string(t) for t in type_strings]

def get_docparam_from_source(web_APIs, lib_name, expand=None):
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
                            'code':str(inspect.getsource(member))
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
                            'code':str(inspect.getsource(member)) #### record code, because we need to modify the docstring for you later. If you have already added the docstring, then this step is ignorable.
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
                        'api_type': api_type,
                        'code':str(inspect.getsource(member))
                    }
                results[member_name] = result
                success_count += 1
            except Exception as e:
                print(f"Failed to process API: {member_name}, Exception: {str(e)}")
                failure_count += 1
    print('Success count:', success_count, 'Failure count:', failure_count)
    return results

def filter_specific_apis(data, lib_name):
    #### TODO
    # can modify the filtering here
    """
    Remove APIs from the data based on specific criteria:
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
        if api_type in ["builtin", 'functools.partial', "rePattern", "cython"]:
            filter_counts["api_type_unwant_func"] += 1
            filter_API["api_type_unwant_func"].append(api)
            continue
        if api_type in ["unknown"]:
            filter_counts["api_type_unknown"] += 1
            filter_API["api_type_unknown"].append(api)
            continue
        # Filter by empty docstring
        #if not docstring:
        #    filter_counts["empty_docstring"] += 1
        #    filter_API["empty_docstring"].append(api)
        #    continue
        # These API is not our targeted API. We filter it because there are too many `method` type API in some libs.
        # TODO: We might include them in future for robustness.
        #if (not parameters) and (not Returns_type) and (not Returns_description):
        #    filter_counts["empty_input_output"] += 1
        #    filter_API["empty_input_output"].append(api)
        #    continue
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

def main_get_API_init(lib_name,lib_alias,analysis_path,api_html_path=None,api_txt_path=None):
    # STEP1
    if api_txt_path:  # or you can provide a self-defined API path
        content_list = []
        try:
            with open(api_txt_path, 'r', encoding='latin') as file:
                content_list = file.readlines()
        except FileNotFoundError:
            print(f"Error: File '{api_txt_path}' not found.")
        except Exception as e:
            print(f"Error: {e}")
        if ',' in content_list[0]:
            ori_content_keys = content_list[0].strip().split(',')
        else:
            ori_content_keys = content_list
        ori_content_keys = [i.strip() for i in ori_content_keys]
    elif api_html_path:  # if you provide API page
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
            raise ValueError
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
    results = get_docparam_from_source(ori_content_keys, lib_name)
    #results = filter_optional_parameters(results)
    results = generate_api_callings(results)
    print('Get API #numbers are: ', len(results))
    tmp_results = filter_specific_apis(results, lib_name) #### This function is different from dataloader/get_API_init...py
    print('After filtering non-calling #numbers are: ', len(tmp_results))
    # output_file = os.path.join(analysis_path,lib_name,"API_init.json")
    os.makedirs(os.path.join('data','standard_process',lib_name), exist_ok=True)
    output_file = os.path.join('data','standard_process',lib_name,"API_init_prepare.json")
    for api_name, api_info in tmp_results.items():
        api_info['relevant APIs'] = []
        api_info['type'] = 'singleAPI'
    with open(output_file, 'w') as file:
        file.write(json.dumps(tmp_results, indent=4))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--LIB', type=str, help='PyPI tool')
    parser.add_argument('--api_txt_path', type=str, default=None, help='Your self-defined api txt path')
    args = parser.parse_args()
    info_json = get_all_variable_from_cheatsheet(args.LIB)
    LIB_ALIAS, API_HTML, TUTORIAL_GITHUB, API_HTML_PATH = [info_json[key] for key in ['LIB_ALIAS', 'API_HTML', 'TUTORIAL_GITHUB','API_HTML_PATH']]
    CHEATSHEET = get_all_basic_func_from_cheatsheet()
    main_get_API_init(args.LIB,LIB_ALIAS,ANALYSIS_PATH,API_HTML_PATH, api_txt_path=args.api_txt_path)
    # currently we do not need the API_base.json
    #main_get_API_basic(ANALYSIS_PATH,CHEATSHEET)
