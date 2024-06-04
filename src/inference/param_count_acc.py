"""
Author: Zhengyuan Dong
Date Created: May 10, 2024
Last Modified: May 28, 2024
Description: parameters correction prediction.
Conclusion: 
"""
import asyncio
from tqdm.asyncio import tqdm_asyncio
import re
import aiohttp
import pandas as pd
import pickle
import json, ast
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, openai
from dotenv import load_dotenv
import tenacity as T
import logging
def extract_api_name_from_query_code(query_code):
    """Extract the API name from a query code string."""
    return query_code.split('(')[0].strip()

from .param_count_acc_just_test import parse_json_safely, post_process_parsed_params

def json_to_docstring(api_name, description, parameters):
    params_list = ', '.join([
        f"{param}:{parameters[param]['type']}={parameters[param]['default']}" if parameters[param]['optional'] else f"{param}:{parameters[param]['type']}"
        for param in parameters
    ])
    function_signature = f"def {api_name}({params_list}):"
    docstring = f"\"\"\"{description}\n\n"
    if len(parameters) > 0:
        docstring += "Parameters\n----------\n"
    for parameter in parameters:
        info = parameters[parameter]
        if (info['description'] is not None) and (info['description'].strip()):
            docstring += f"{parameter}: {info['description']}\n"
    docstring += "\"\"\""
    return function_signature + "\n" + docstring.strip()

def common_prompt():
    return """
Task Description: Determine from the API DOCSTRING and USER QUERY whether each parameter needs a predicted value. Based on the parameter type and meaning, infer if it needs to fill in a value or modify the default value, and return all the parameters that need prediction or modification with their predicted values.
Instructions:
1. Extract values and assign to parameters: If there are values explicitly appearing in the USER QUERY, such as those quoted with '', values with underscores like value_value, or list/tuple/dict like ['A', 'B'], they are likely to be actual parameter values. When handling these values:
Boolean values: If we are predicting boolean values, you can ignore these explicitly mentioned values.
Integer/float/str/list/tuple values: If we are predicting these types of values, you can assign them to the most corresponding parameters based on the modifiers of the extracted values.
2. Extract and convert values if necessary: Some values might not be obvious and require conversion (e.g., converting "Diffusion Map" to "diffmap" if "Diffusion Map" appears in the user query, as this is the correct usage for some API. Another example is converting "blank space delimiter" to "delimiter=' '"). Ensure that the returned values are correctly formatted, case-sensitive, and directly usable in API calls.
3. Identify relevant parameters: From the USER QUERY and API DOCSTRING, identify which parameters require value extraction or default value modification based on their type and meaning.
4. Never generate fake values: Only include values explicitly extracted in the USER QUERY or that can be clearly inferred. Do not generate fake values for remaining parameters.
5. Response format: Format results as {{"param name": "extracted value"}}, ensuring they are executable and match the correct type from the API DOCSTRING. For example, return ['a', 'b'] for list type instead of Tuple ('a', 'b'). Only return parameters with non-empty values (i.e. not None).
6. Avoid complexity: Avoid using 'or' in parameter values and ensure no overly complex escape characters. The format should be directly loadable by json.loads.
7. Distinguish similar parameters: For similar parameters (e.g., 'knn' and 'knn_max'), distinguish them in meaning and do not mix them up. If there are two parameter candidates with the same meaning for one value, fillin them both with the extracted value.
8. Maintain Format and transfer format if necessary: Ensure the final returned values match the expected parameter type, including complex structures like [("keyxx", 1)] for parameters like "obsm_keys" with types like "Iterable[tuple[str, int]]". If USER QUERY contains list, tuple, or dictionary values, extract them as a whole and keep the format as a value.
"""
def prepare_parameters_prompt(user_query, api_docstring, parameters_name_list, param_type):
    common_part = common_prompt()
    if param_type=='boolean':
        return f"""
{common_part}
Boolean:
- Determine the state (True/False) based on the full USER QUERY and the parameter description, rather than just a part of it. If the 'True/False' value is not explicitly mentioned but the meaning is implied in the USER QUERY, still predict the parameter.

Examples:
USER QUERY: "... with logarithmic axes" => {{"log":"True"}}
USER QUERY: "... without logarithmic axes" => {{"log":"False"}}
USER QUERY: "... hide the default legends and the colorbar legend" => {{"show":"False", "show_colorbar":"False"}}
USER QUERY: "... and create a copy of the data structure" => {{"copy": "True"}}, do not predict "data"

Now start extracting the values with parameters from USER QUERY based on parameters descriptions and default value from API DOCSTRING
USER QUERY: {user_query}
API DOCSTRING: {api_docstring}
param_list: {parameters_name_list}
Return parameter with extracted value in format: {{"param1":"value1", "param2":"value2", ...}}. Only return json format answer in one line, do not return other descriptions.
"""
    elif param_type=='int':
        return f"""
{common_part}
int, float, str, list[str], tuple[str]:
- Choose only parameters with explicitly mentioned values in the USER QUERY. If a parameter name is mentioned without a value (e.g., "a specific group/key" where "group/key" is the parameter name but no value is given), ignore that parameter.

Examples:
USER QUERY: "... with chunk size 6000" => {{"chunked":"True", "chunk_size":"6000"}}
USER QUERY: "... with groups ['A', 'B', 'C']" => {{"groups":['A', 'B', 'C']}}
USER QUERY: "... with at least 100 counts." => {{"min_counts": 100}}
USER QUERY: "... for a specific layer 'X_new'?" => {{"layer": "X_new"}}, do not assign "X_new" to "X"

Now start extracting the values with parameters from USER QUERY based on parameters descriptions and default value from API DOCSTRING
USER QUERY: {user_query}
API DOCSTRING: {api_docstring}
param_list: {parameters_name_list}
Return parameter with extracted value in format: {{"param1":"value1", "param2":"value2", ...}}. Only return json format answer in one line, do not return other descriptions.
"""
    elif param_type=='literal':
        return f"""
{param_type}
Literal:
- Find the candidate with the same level of informativeness. If there are multiple candidates containing the keyword, but the query provides only minimal information, select the candidate with the least information content. If a parameter name is mentioned without a value, skip it.

Examples:
USER QUERY: "... with basis 'Diffusion Map'" => {{"basis":'diffmap'}}
USER QUERY: "... blank space delimiter" => {{"delimiter":' '}}

Now start extracting the values with parameters from USER QUERY based on parameters descriptions and default value from API DOCSTRING
USER QUERY: {user_query}
API DOCSTRING: {api_docstring}
param_list: {parameters_name_list}
Return parameter with extracted value in format: {{"param1":"value1", "param2":"value2", ...}}. Only return json format answer in one line, do not return other descriptions.
"""

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def adjust_optional_parameters(parameters):
    # After finding the first optional parameter, all subsequent parameters are considered optional
    first_optional_found = False
    adjusted_params = {}
    for name, details in parameters.items():
        if first_optional_found:
            adjusted_details = details.copy()
            if not adjusted_details['optional']:
                adjusted_details['optional'] = True
            adjusted_params[name] = adjusted_details
        else:
            adjusted_params[name] = details
            if details['optional']:
                first_optional_found = True
    return adjusted_params

def adjust_adata_parameters(parameters):
    adjusted_params = {}
    for name, details in parameters.items():
        adjusted_details = details.copy()
        if name == 'adata' and details.get('type') is None:
            print('adjusted type of adata')
            adjusted_details['type'] = 'AnnData'
        adjusted_params[name] = adjusted_details
    return adjusted_params

def patch_query_code(item):
    full_api_call = item['api_calling'][0].split('(')[0].strip()
    query_code = item['query_code'].strip()
    function_name = full_api_call.split('.')[-1]
    if query_code.startswith(function_name + '(') and not query_code.startswith(full_api_call):
        new_query_code = full_api_call + query_code[len(function_name):]
        item['query_code'] = new_query_code
    return item['query_code']

def generate_api_calling(api_name, params, param_details, io_param_names = {'filename'}, io_types = {'PathLike', 'Path'}, special_types = {'AnnData', 'ndarray', 'spmatrix', 'DataFrame', 'recarray', 'Axes'}):
    params_call = []
    for param in params:
        type_info = param_details[param]['type']
        if param in io_param_names or any(t in type_info for t in io_types):
            params_call.append(param + '=@')
        elif any(t in type_info for t in special_types):
            params_call.append(param + '=$')
        else:
            params_call.append(param + '=@')
    return f"{api_name}(" + ', '.join(params_call) + ")"

special_types = {'AnnData', 'ndarray', 'spmatrix', 'DataFrame', 'recarray', 'Axes'}
io_types = {'PathLike', 'Path'}
io_param_names = {'filename'}

def check_overall_parameter_accuracy(data, total_inquiries):
    correct_inquiries = 0
    correct_params_name = 0
    correct_params_value = 0
    for item in data:
        params_list = list(item['Parameters'].keys())
        api_name = item['api_calling'][0].split('(')[0].strip()
        query_params = parse_true_parameters(item['query_code'], params_list)
        all_params_name_correct = True
        all_params_value_correct = True
        for param, value in query_params.items():
            if param not in params_list:
                all_params_name_correct = False
                print(f"{item['api_calling'][0].split('(')[0].strip()}: {param} not in {params_list}")
                break
            value = value.strip().replace('"', '').replace("'", "") if value is not None else None
            if value not in item['query']:
                all_params_value_correct = False
                break
        if all_params_name_correct and all_params_value_correct:
            correct_inquiries += 1
        if all_params_name_correct:
            correct_params_name += 1
        if all_params_value_correct:
            correct_params_value += 1
    accuracy = correct_inquiries / total_inquiries if total_inquiries > 0 else 0
    param_name_acc = correct_params_name / total_inquiries if total_inquiries > 0 else 0
    param_value_acc = correct_params_value / total_inquiries if total_inquiries > 0 else 0
    return correct_inquiries, param_name_acc, param_value_acc, accuracy

def evaluate_parameters(query, query_code, query_id, api_name, Parameters, true_params, pred_params, api_data):
    # Convert all values to strings for comparison
    str_true_params = {k: str(v) for k, v in true_params.items()}
    str_pred_params = {k: str(v) for k, v in pred_params.items()}
    # Determine non-default parameters in true_params
    non_default_true_params = {k: v for k, v in true_params.items() if api_data[api_name]['Parameters'][k]['default'] != v}
    # Determine non-None parameters in pred_params
    non_none_pred_params = {k: v for k, v in str_pred_params.items() if v != 'None'}
    # Calculate correct names and values
    correct_names = {k: k in non_none_pred_params for k in non_default_true_params.keys()}
    correct_values = {k: str(non_default_true_params[k]) == non_none_pred_params.get(k) for k in non_default_true_params if k in non_none_pred_params}
    # Initialize counts for confusion matrix
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    # Calculate confusion matrix values based on parameter values
    for key in non_default_true_params:
        if key in non_none_pred_params:
            if str(non_default_true_params[key]) == non_none_pred_params[key]:
                true_positive += 1  # Correctly predicted
            else:
                false_negative += 1  # Wrong value predicted
        else:
            false_negative += 1  # Missed true parameter
    for key in non_none_pred_params:
        if key not in non_default_true_params:
            false_positive += 1  # Incorrectly predicted parameter
    true_negative = len(Parameters) - (true_positive + false_negative + false_positive)  # Remaining parameters are true negatives
    return {
        "query": query,
        "query_code": query_code,
        "query_id": query_id,
        "api_name": api_name,
        "api_params": Parameters,
        "correct_names": correct_names,
        "correct_values": correct_values,
        "true_params": true_params,
        "pred_params": pred_params,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative
    }

def accumulate_metrics(metrics, item_results):
    metrics['total_correct_params'] += sum(item_results['correct_names'].values())
    metrics['total_correct_values'] += sum(item_results['correct_values'].values())
    metrics['total_params'] += len(item_results['api_params'])

def calculate_final_metrics(metrics):
    param_name_acc_based_params = metrics['total_correct_params'] / metrics['total_params'] if metrics['total_params'] > 0 else 0
    param_value_acc_based_params = metrics['total_correct_values'] / metrics['total_params'] if metrics['total_params'] > 0 else 0
    param_value_acc_based_non_empty_params = metrics['total_correct_non_empty_values'] / metrics['total_non_empty_params'] if metrics['total_non_empty_params'] > 0 else 0
    return {
        "param_name_acc_based_all_params": param_name_acc_based_params,
        "param_value_acc_based_all_params": param_value_acc_based_params,
        "param_value_acc_based_non_empty_params": param_value_acc_based_non_empty_params
    }

def analyze_api_data(data, key='query_code_params'):
    api_count = 0
    total_parameters = 0
    non_empty_parameter_values = 0
    non_empty_apis = 0
    for item in data:
        api_count += 1
        api_has_non_empty_param = False
        params = item[key]
        total_parameters += len(params)
        # Count non-empty parameter values and check if any parameter is non-empty
        for value in params.values():
            if value is not None:
                non_empty_parameter_values += 1
                api_has_non_empty_param = True
        # Check if the API query code is non-empty
        if item['query_code'].strip() or api_has_non_empty_param:
            non_empty_apis += 1
    return {
        'total_inquiries': api_count,
        'total_parameters': total_parameters,
        'non_empty_parameter_values': non_empty_parameter_values,
        'non_empty_inquiries': non_empty_apis
    }

def parse_true_parameters(query_code, params_list):
    # Extract the parameters string from the function call
    params_str = query_code[query_code.find('(') + 1:query_code.rfind(')')].strip()
    # Initialize a dictionary to store the parameters
    params_dict = {}
    # Current index for unnamed parameters (positional)
    index = 0
    # Initialize a stack to handle nested structures like tuples, lists, and dictionaries
    stack = []
    current_param = []
    # Process each character in the parameter string
    for char in params_str:
        if char in '([{' :
            # Entering a new nested level
            if current_param:  # Include '(', '[', '{' if part of a parameter (e.g., in a tuple, list, or dictionary)
                current_param.append(char)
            stack.append(current_param)
            current_param = []
        elif char in ')]}' :
            # Leaving a nested level
            nested = ''.join(current_param)
            if stack:
                current_param = stack.pop()
            current_param.append(nested + char)
        elif char == ',' and not stack:
            # End of a parameter
            param = ''.join(current_param).strip()
            if '=' in param:
                key, value = map(str.strip, param.split('=', 1))
                params_dict[key] = value
            else:
                if index < len(params_list):
                    params_dict[params_list[index]] = param
                index += 1
            current_param = []
        else:
            # Accumulate characters in the current parameter
            current_param.append(char)
    # Check for any remaining parameter after the last comma
    if current_param:
        param = ''.join(current_param).strip()
        if '=' in param:
            key, value = map(str.strip, param.split('=', 1))
            params_dict[key] = value
        else:
            if index < len(params_list):
                params_dict[params_list[index]] = param
    return params_dict

def parse_api_calling_parameters(new_api_calling):
    params_pattern = re.compile(r'\((.*?)\)')
    params_str = params_pattern.search(new_api_calling).group(1)
    params = [param.strip().strip('$@').replace('=','') for param in params_str.split(',')]
    return params

def classify_apis(inquiry_data, special_types, io_types, io_param_names):
    data_for_df = []
    class1, uncategorized, uncategorized_item = [], [], []
    for item in inquiry_data:
        query_id = item['query_id']
        params_list = list(item['Parameters'].keys())
        api_name = item['api_calling'][0].split('(')[0].strip()
        item['query_code'] = patch_query_code(item)
        extracted_api_name = extract_api_name_from_query_code(item['query_code'])
        if '(' in item['query_code']:
            query_params = parse_true_parameters(item['query_code'], params_list)
        else:
            query_params = {}
        polished_parameters = adjust_optional_parameters(item['Parameters'])
        new_polished_parameters = adjust_adata_parameters(polished_parameters)
        required_params = {name: details for name, details in new_polished_parameters.items() if not details['optional']}
        optional_params = {name: details for name, details in new_polished_parameters.items() if details['optional']}
        new_api_calling = generate_api_calling(api_name, required_params.keys(), required_params, io_param_names = io_param_names, io_types = io_types, special_types = special_types)
        item['new_api_calling'] = new_api_calling
        #item['required_params'] = required_params
        # Step1: count the special type
        def process_type(details_type):
            if not details_type:
                return [details_type]
            # remove optional[]
            if details_type.startswith("Optional[") and details_type.endswith("]"):
                details_type = details_type[len("Optional["):-1].strip()
            if details_type.startswith("Literal"):
                return ["Literal"]
            # remove Union[]
            if details_type.startswith("Union[") and details_type.endswith("]"):
                details_type = details_type[len("Union["):-1].strip()
            if not any(char in details_type for char in "[]()"):
                if ',' in details_type:
                    details_type = details_type.split(',')
                    return details_type
                else:
                    return [details_type]
            else:
                return [details_type]
        
        for param, details in required_params.items():
            param_types = process_type(details['type'])
            for param_type in param_types:
                data_for_df.append({
                    'query_id': query_id,
                    'parameter': param_types,
                    'type': param_type,
                    'description': details['description']
                })
        for param, details in optional_params.items():
            param_types = process_type(details['type'])
            for param_type in param_types:
                data_for_df.append({
                    'query_id': query_id,
                    'parameter': param_types,
                    'type': param_type,
                    'description': details['description']
                })
        # Step2: filter the parameters which are in special types and io types and io_param_names
        ground_truth_params = {
            param: details
            for param, details in required_params.items()
            if not any(t in str(details['type']) for t in special_types.union(io_types)) and param not in io_param_names
        }
        item['ground_truth_params'] = ground_truth_params
        
        ground_truth_params_optional = {
            param: details
            for param, details in optional_params.items()
            if not any(t in str(details['type']) for t in special_types.union(io_types)) and param not in io_param_names
        }
        item['ground_truth_params_optional'] = ground_truth_params_optional
        
        all_params_name_correct = True
        all_params_value_correct = True
        api_name_correct = extracted_api_name == api_name
        # Step3: filter the parameters from query_code
        query_code_params = {
            param: value.strip().replace('"', '').replace("'", "")
            for param, value in query_params.items()
            if param in ground_truth_params or param in ground_truth_params_optional
        }
        query_code_params_optional = {
            param: value.strip().replace('"', '').replace("'", "")
            for param, value in query_params.items()
            if param in ground_truth_params_optional
        }
        item['query_code_params'] = query_code_params
        item['query_code_params_optional'] = query_code_params_optional
        # for inquiry
        for param, details in query_code_params.items():
            if param not in params_list:
                all_params_name_correct = False
            value = str(query_code_params.get(param, "")).strip().replace('"', '').replace("'", "")
            if value and value not in item['query']:
                all_params_value_correct = False
        # Step4: classify the queries
        if not ground_truth_params and not ground_truth_params_optional:
            class1.append(query_id)
        elif not query_code_params and not query_code_params_optional:
            class1.append(query_id)
        else:
            uncategorized.append(query_id)
            uncategorized_item.append(item)
    df = pd.DataFrame(data_for_df)
    print("Distribution:", df['type'].value_counts())
    for item in uncategorized_item:
        del item['Docstring']
        del item['relevant APIs']
        #del item['example']
        del item['Returns']
        del item['type']
        del item['api_calling']
        del item['description']
        del item['Parameters']
    return uncategorized_item

def replace_ellipsis(data):
    if isinstance(data, dict):
        return {k: replace_ellipsis(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_ellipsis(i) for i in data]
    elif data is ...:
        return None
    return data

def process_results(results, api_data):
    metrics = {
        'total_correct_params': 0,
        'total_correct_values': 0,
        'total_params': 0,
        'total_api': 0,
        'total_non_empty_params': 0,
        'total_correct_non_empty_values': 0
    }
    for item_results in results:
        corrected_pred_params = {}
        for param_name, value in item_results['pred_params'].items():
            corrected_pred_params[param_name] = None if value == 'None' else value
            try:
                if value is not None and '"' in value:
                    corrected_pred_params[param_name] = value.replace('"', '')
                if value is not None and "'" in value:
                    corrected_pred_params[param_name] = value.replace("'", '')
            except Exception as e:
                print('error:', e)
                pass
        item_results['pred_params'] = corrected_pred_params
        # Use the evaluate_parameters function
        eval_result = evaluate_parameters(
            item_results['query'],
            item_results['query_code'],
            item_results['query_id'],
            item_results['api_name'],
            item_results['api_params'],
            item_results['true_params'],
            item_results['pred_params'],
            api_data
        )
        item_results.update(eval_result)
        metrics['total_correct_params'] += sum(eval_result['correct_names'].values())
        metrics['total_correct_values'] += sum(eval_result['correct_values'].values())
        metrics['total_params'] += len(item_results['api_params'])
        metrics['total_api'] += 1
        has_non_empty_params = False
        non_empty_params_count = 0
        correct_non_empty_values_count = 0
        for param, value in item_results['true_params'].items():
            if value is not None:
                has_non_empty_params = True
                non_empty_params_count += 1
                metrics['total_non_empty_params'] += 1
                if item_results['pred_params'].get(param) == str(value):
                    metrics['total_correct_non_empty_values'] += 1
                    correct_non_empty_values_count += 1
        item_results['has_non_empty_params'] = has_non_empty_params
        item_results['non_empty_params_count'] = non_empty_params_count
        item_results['correct_non_empty_values_count'] = correct_non_empty_values_count
    return metrics

async def predict_parameters(doc, instruction, parameters_name_list, api_name, api_data, param_type, model="gpt-3.5-turbo-0125"):
    prompt = prepare_parameters_prompt(instruction, doc, parameters_name_list, param_type=param_type)
    #print('prompt: ', prompt)
    response = await query_openai(prompt, "openai", model=model)  # gpt-4  # 
    returned_content_str_new = response.replace('null', 'None').replace('None', '"None"')
    #factory = PromptFactory()
    #prompt = factory.create_prompt('parameters', instruction, doc, parameters_name_list)
    #print("returned_content_str_new: ", returned_content_str_new)
    try:
        predicted_params, success = parse_json_safely(returned_content_str_new)
        parsed_data = post_process_parsed_params(predicted_params, api_name, api_data)
    except Exception as e:
        parsed_data = {}
    #if isinstance(parsed_data, dict):
    #    parsed_data = list(parsed_data.keys())
    print('parsed_data: ', parsed_data)
    if success:
        return parsed_data, returned_content_str_new
    else:
        return {}, returned_content_str_new

async def process_item(item, api_data, special_types = {'AnnData', 'ndarray', 'spmatrix', 'DataFrame', 'recarray', 'Axes'}, io_types = {'PathLike', 'Path'}):
    instruction = item['query']
    api_name = item['new_api_calling'].split('(')[0]
    # remove special types and io types and io_param_names and parameters without description
    param_tmp = {i:api_data[api_name]['Parameters'][i] for i in api_data[api_name]['Parameters'] if (api_data[api_name]['Parameters'][i]['description'] is not None) and (api_data[api_name]['Parameters'][i]['type'] not in special_types) and (api_data[api_name]['Parameters'][i]['type'] not in io_types) and (i not in io_param_names)}
    boolean_params = {k: v for k, v in param_tmp.items() if 'boolean' in str(v['type']) or 'bool' in str(v['type'])}
    literal_params = {k: v for k, v in param_tmp.items() if 'literal' in str(v['type']) or 'Literal' in str(v['type'])}
    #int_params = {k: v for k, v in param_tmp.items() if v['type'] in {'int', 'float', 'str', 'list[str]', 'tuple[str]'}}
    int_params = {k: v for k, v in param_tmp.items() if k not in boolean_params and k not in literal_params}
    boolean_document = json_to_docstring(api_name, api_data[api_name]["description"], boolean_params)
    literal_document = json_to_docstring(api_name, api_data[api_name]["description"], literal_params)
    int_document = json_to_docstring(api_name, api_data[api_name]["description"], int_params)
    
    #document = json_to_docstring(api_name, api_data[api_name]["description"], param_tmp)
    problem_type = 'multiple'
    predicted_params = {}
    if boolean_params:
        boolean_predicted, response1 = await predict_parameters(boolean_document, instruction, list(boolean_params.keys()), api_name, api_data, 'boolean')
        predicted_params.update(boolean_predicted)
    else:
        response1 = ""
    if int_params:
        int_predicted, response2 = await predict_parameters(int_document, instruction, list(int_params.keys()), api_name, api_data, 'int')
        predicted_params.update(int_predicted)
    else:
        response2 = ""
    if literal_params:
        literal_predicted, response3 = await predict_parameters(literal_document, instruction, list(literal_params.keys()), api_name, api_data, 'literal')
        predicted_params.update(literal_predicted)
    else:
        response3 = ""
    #predicted_params, response = await predict_parameters(document, instruction, list(param_tmp.keys()), api_name, api_data)
    print(f'for api {api_name}, predicted_params: {predicted_params}') # the document is {document}, 
    if predicted_params:
        pass
    else:
        predicted_params = {}
    #item_results = evaluate_parameters(instruction, item['query_code'], item['query_id'], api_name, param_tmp, item['query_code_params'], predicted_params, api_data)
    # evaluate
    param = {i: api_data[api_name]['Parameters'][i] for i in api_data[api_name]['Parameters'] if i in item["query_code_params"]}
    item_results = evaluate_parameters(instruction, item['query_code'], item['query_id'], api_name, param, item['query_code_params'], predicted_params, api_data)
    try:
        item_results['gpt_response1'] = response1
    except:
        item_results['gpt_response1'] = ""
    try:
        item_results['gpt_response2'] = response2
    except:
        item_results['gpt_response2'] = ""
    try:
        item_results['gpt_response3'] = response3
    except:
        item_results['gpt_response3'] = ""
    return item_results

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
async def query_openai(prompt, mode='azure', model='gpt-35-turbo', max_tokens = 1200, **kwargs):
    # 240127: update openai version
    if mode == 'openai':
        response = openai.chat.completions.create(model=model,
                                            messages=[{'role': 'user', 'content': prompt}],
                                            max_tokens=max_tokens,
                                            **kwargs
                                            )
    else:
        response = openai.chat.completions.create(
            deployment_id=model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            **kwargs,
        )
    return response.choices[0].message.content

def transfer_type(results):
    for items in results:
        for key in items['pred_params']:
            if isinstance(items['pred_params'][key], bool) or isinstance(items['pred_params'][key], int) or isinstance(items['pred_params'][key], float):
                items['pred_params'][key] = str(items['pred_params'][key])
    return results

def save_with_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--LIB', type=str, required=True, help='Library name')
    args = parser.parse_args()
    with open(f"./data/standard_process/{args.LIB}/API_inquiry.json", "r") as file:
        inquiry_data = json.load(file)
        
    with open(f"./data/standard_process/{args.LIB}/API_init.json", "r") as file:
        api_data = json.load(file)

    #with open(f"api_parameters_{args.LIB}_class3_final.json", 'r') as f:
    #    uncategorized_item = json.load(f)
    #print('uncategorized_item: ', len(uncategorized_item))
    #print(analyze_api_data(uncategorized_item))

    # add optional items
    with open(f"api_parameters_{args.LIB}_class3_final_modified_v3.json", 'r') as f:
        uncategorized_item = json.load(f)
    # filter out the samples with some parameters
    uncategorized_item = [item for item in uncategorized_item if item['query_code_params']]
    # filter out the samples with non null parameters
    uncategorized_item = [item for item in uncategorized_item if any(item['query_code_params'].values())]
    print('after adding optional items:')
    print(analyze_api_data(uncategorized_item))
    def filter_uncategorized_items(uncategorized_item, api_data):
        filtered_items = []
        for item in uncategorized_item:
            api_name = item['new_api_calling'].split('(')[0].strip()
            query_code_params = item['query_code_params']
            # Check if all parameters have empty descriptions
            all_params_empty_desc = all(
                api_data[api_name]['Parameters'][param_name]['description'] is None or
                api_data[api_name]['Parameters'][param_name]['description'].strip() == ''
                for param_name in query_code_params
            )
            # Filter out items with non-null parameters
            has_non_null_params = any(query_code_params.values())
            # Add item to filtered list if it doesn't have all empty descriptions and has non-null parameters
            if not all_params_empty_desc and has_non_null_params:
                filtered_items.append(item)
        return filtered_items
    uncategorized_item = filter_uncategorized_items(uncategorized_item, api_data)
    print('after removing non description parameters items:')
    print(analyze_api_data(uncategorized_item))
    
    #### Prediction:
    # query_code parameters name and value check:
    #from BioMANIA.inference.utils import json_to_docstring
    metrics = {
        'total_correct_params': 0,
        'total_correct_values': 0,
        'total_params': 0,
        'total_api': 0
    }
    results = []
    
    tasks = [process_item(item, api_data) for item in uncategorized_item]
    for item_results in await tqdm_asyncio.gather(*tasks):
        results.append(item_results)
        accumulate_metrics(metrics, item_results)
        metrics['total_api'] += 1

    with open(f'api_parameters_{args.LIB}_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    # transfer type
    results = transfer_type(results)
    print(analyze_api_data(results, key='true_params'))

    metrics = process_results(results, api_data)
    final_metrics = calculate_final_metrics(metrics)
    print("Final Metrics across all API calls:")
    print(final_metrics)
    results = replace_ellipsis(results)
    
    save_with_pickle(results, 'final_results.pkl')
    with open(f'api_parameters_{args.LIB}_final_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())