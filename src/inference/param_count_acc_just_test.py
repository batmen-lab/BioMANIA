"""
Author: Zhengyuan Dong
Date Created: May 10, 2024
Last Modified: Sep 12, 2024
Description: parameters correction prediction.
Conclusion: 
"""
import json, ast, re, pickle

def plot_confusion_matrix(metrics, lib_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import matplotlib.colors as mcolors
    
    conf_matrix = np.array([
        [metrics['true_positive_ratio']*100, (1-metrics['true_positive_ratio'])*100],
        [(1-metrics['false_positive_ratio_really'])*100, metrics['false_positive_ratio_really']*100]
    ])
    
    colors = [(0.6, 0.6, 0.6, 0.6) if i == 'None' else mcolors.to_rgba(i, alpha=0.6) for i in ['#f4b26e', '#89a7bf']]
    cmap = mcolors.LinearSegmentedColormap.from_list('DeepBlueOrange', colors)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap=cmap, cbar=True, annot_kws={"size": 16})

    plt.title('Confusion Matrix', fontsize=18)
    plt.xlabel(f'Predicted Labels for {lib_name}', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)

    ax.set_xticklabels(['Predicted Correctly', 'Predicted Wrongly'], fontsize=12)
    ax.set_yticklabels(['Should be Predicted', 'Should not be Predicted'], fontsize=12)

    plt.savefig(f'confusion_matrix_{lib_name}.pdf')
    #plt.savefig(f'confusion_matrix_{lib_name}.png')

def post_process_parsed_params(predicted_params, api_name, api_data):
    if not predicted_params:
        predicted_params = {}
    corrected_pred_params = {}
    if isinstance(predicted_params, list):
        predicted_params = predicted_params[0]
    for param_name, value in predicted_params.items():
        if value in [None, "None", "null"]:
            corrected_pred_params[param_name] = None
            continue
        # Convert "true"/"false" to "True"/"False"
        if value == "true":
            value = "True"
        elif value == "false":
            value = "False"
        # Convert "\t" to "\\t"
        if value=='\t':
            value = '\\t'
        #try:
        #    value = json.loads(value)  # Try to parse JSON
        #except (json.JSONDecodeError, TypeError):
        #    pass  # Keep value as is if JSON parsing fails
        value = correct_param_type(param_name, value, api_data, api_name)
        corrected_pred_params[param_name] = None if value == 'None' else value
        if value is not None:
            value = str(value)
            if '"' in value:
                value = value.replace('"', '')
            if "'" in value:
                value = value.replace("'", '')
            corrected_pred_params[param_name] = value
    return corrected_pred_params

def correct_param_type(param_name, param_value, api_data, api_name):
    if param_name not in api_data[api_name]['Parameters']:
        return param_value
    param_type = api_data[api_name]['Parameters'][param_name]['type']
    if param_type in [None, "None"]:
        return param_value
    if 'List' in param_type or 'list' in param_type:
        if str(param_value).startswith('(') and str(param_value).endswith(')'):
            param_value = str(param_value).replace('(', '[').replace(')', ']')
        # Convert string representation of numbers to list
        elif re.match(r'^\d+(,\d+)*$', str(param_value)):
            param_value = f'[{param_value}]'
    elif 'Tuple' in param_type or 'tuple' in param_type:
        if str(param_value).startswith('[') and str(param_value).endswith(']'):
            param_value = str(param_value).replace('[', '(').replace(']', ')')
        elif re.match(r'^\(.*\)$', str(param_value)):
            # Convert string representation of tuple to actual tuple
            #param_value = eval(param_value)
            param_value = tuple(float(x) if x.isdigit() else float(x) for x in re.findall(r'[-+]?\d*\.?\d+', param_value))
        # Convert string representation of numbers to tuple
        elif re.match(r'^\d+(,\d+)*$', str(param_value)):
            param_value = f'({param_value})'
    if 'Iterable' in param_type or 'List' in param_type or 'Tuple' in param_type:
        if not str(param_value).startswith('[') and not str(param_value).startswith('('):
            param_value = '[' + str(param_value) + ']'
    return param_value

import json
import re
import ast

def parse_json_safely(input_data):
    def clean_json_string(s):
        s = s.replace('""', '"').replace('\\"', '"').replace("\\'", "'")
        s = s.replace('"None"', 'null').replace("'None'", 'null')
        s = s.replace('True', 'true').replace('False', 'false')
        s = s.replace('"\'', '"').replace('\'"', '"')
        s = re.sub(r'\'([^\']+)\'', r'"\1"', s)
        s = re.sub(r'"\s*\'', '"', s)
        s = re.sub(r'\'\s*"', '"', s)
        s = re.sub(r'\(([^()]+)\)', r'[\1]', s)
        s = re.sub(r'"\[([^]]+)\]"', r'[\1]', s)
        return s
    def try_parse_json(s):
        try:
            return json.loads(s), True
        except json.JSONDecodeError:
            return None, False
    def try_literal_eval(s):
        try:
            return ast.literal_eval(s), True
        except (ValueError, SyntaxError):
            return None, False
    def convert_values_to_str(data):
        if isinstance(data, dict):
            return {k: str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [str(item) for item in data]
        else:
            return str(data)
    def parse_mixed_json_str(s):
        parts = [part.strip() for part in s.split(',')]
        parsed_parts = []
        for part in parts:
            if not part.startswith('{'):
                part = '{' + part
            if not part.endswith('}'):
                part = part + '}'
            cleaned_part = clean_json_string(part)
            parsed_data, success = try_parse_json(cleaned_part)
            if not success:
                parsed_data, success = try_literal_eval(cleaned_part)
            if success:
                parsed_parts.append(parsed_data)
        return parsed_parts
    if isinstance(input_data, str):
        cleaned_json_str = clean_json_string(input_data)
        multiple_objects = cleaned_json_str.count('{') > 1 and cleaned_json_str.count('}') > 1
        if multiple_objects:
            parsed_data = parse_mixed_json_str(cleaned_json_str)
            success = True if parsed_data else False
            if success:
                return parsed_data, True
        parsed_data, success = try_parse_json(cleaned_json_str)
        if not success:
            parsed_data, success = try_literal_eval(cleaned_json_str)
        if success:
            parsed_data = convert_values_to_str(parsed_data)
            if isinstance(parsed_data, dict):
                return [parsed_data], True
            elif isinstance(parsed_data, list):
                return parsed_data, True
    elif isinstance(input_data, list):
        parsed_data_list = []
        for item in input_data:
            if isinstance(item, str):
                cleaned_json_str = clean_json_string(item)
                parsed_data, success = try_parse_json(cleaned_json_str)
                if not success:
                    parsed_data, success = try_literal_eval(cleaned_json_str)
                if success:
                    parsed_data_list.append(convert_values_to_str(parsed_data))
            elif isinstance(item, dict):
                parsed_data_list.append(convert_values_to_str(item))
        return parsed_data_list, True
    return [], False

def standardize_param_value(value):
    try:
        # Try to parse as JSON and convert to standardized string format
        parsed_value = json.loads(str(value))
        if isinstance(parsed_value, list):
            parsed_value = [str(i).replace(' ', '').replace(' ', '').replace('"', '').replace("'", '') for i in parsed_value]
            return '[' + ','.join(parsed_value) + ']'
        if isinstance(parsed_value, tuple):
            parsed_value = [str(i).replace(' ', '').replace(' ', '').replace('"', '').replace("'", '') for i in parsed_value]
            return '(' + ','.join(parsed_value) + ')'
        return str(parsed_value).replace(' ', '').replace(' ', '').replace('"', '').replace("'", '')
    except (json.JSONDecodeError, TypeError):
        # Convert lists to standardized string format
        if isinstance(value, list):
            value = [str(i).replace(' ', '').replace(' ', '').replace('"', '').replace("'", '') for i in value]
            return '[' + ','.join(value) + ']'
        if isinstance(value, tuple):
            value = [str(i).replace(' ', '').replace(' ', '').replace('"', '').replace("'", '') for i in value]
            return '(' + ','.join(value) + ')'
        return str(value).replace(' ', '').replace(' ', '').replace('"', '').replace("'", '')

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

special_types = {'AnnData', 'ndarray', 'spmatrix', 'DataFrame', 'recarray', 'Axes'}
io_types = {'PathLike', 'Path'}
io_param_names = {'filename'}

def compare_values(true_value, pred_value):
    # Convert to lower case for comparison, remove spaces and quotes
    true_value = str(true_value).strip().lower().replace(' ', '').replace('"', '').replace("'", '').replace('[', '').replace(']', '').replace('(', '').replace(')', '')
    pred_value = str(pred_value).strip().lower().replace(' ', '').replace('"', '').replace("'", '').replace('[', '').replace(']', '').replace('(', '').replace(')', '')
    # Handle numeric values comparison
    try:
        true_value_num = float(true_value)
        pred_value_num = float(pred_value)
        if true_value_num == pred_value_num:
            return True
    except ValueError:
        pass
    try:
        true_value_num = ast.literal_eval(true_value)
        pred_value_num = ast.literal_eval(pred_value)
        if true_value_num == pred_value_num:
            return True
    except:
        pass
    # Handle \\t case
    if true_value == '\t' and pred_value == '\\t':
        return True
    return true_value == pred_value

def evaluate_parameters(query, query_code, query_id, api_name, Parameters, true_params, pred_params, api_data):
    category_stats = {
        'Literal': {
            'true_positive': 0,
            'true_negative': 0,
            'true_negative_wrong': 0,
            'true_negative_none': 0,
            'false_positive': 0,
            'false_negative': 0,
        },
        'Boolean': {
            'true_positive': 0,
            'true_negative': 0,
            'true_negative_wrong': 0,
            'true_negative_none': 0,
            'false_positive': 0,
            'false_negative': 0,
        },
        'Others': {
            'true_positive': 0,
            'true_negative': 0,
            'true_negative_wrong': 0,
            'true_negative_none': 0,
            'false_positive': 0,
            'false_negative': 0,
        }
    }
    
    # Convert all values to strings for comparison
    str_true_params = {k: standardize_param_value(v) for k, v in true_params.items() if v is not None} #  and Parameters[k]['description']
    #str_pred_params = {k: standardize_param_value(v) for k, v in pred_params.items() if k in Parameters and ((api_data[api_name]['Parameters'][k]['type'] not in special_types) and (api_data[api_name]['Parameters'][k]['type'] not in io_types) and (k not in io_param_names))}#  and Parameters[k]['description']
    str_pred_params = {k: standardize_param_value(v) for k, v in pred_params.items() if k in Parameters and (not any(t in str(api_data[api_name]['Parameters'][k]['type']) for t in special_types.union(io_types)) and k not in io_param_names)}#  and Parameters[k]['description']
    
    # Determine non-default parameters in true_params
    #non_default_true_params = {k: standardize_param_value(v) for k, v in true_params.items() if standardize_param_value(api_data[api_name]['Parameters'][k]['default']) != standardize_param_value(v) and k in Parameters}
    non_default_true_params = {k: v for k, v in str_true_params.items() if standardize_param_value(api_data[api_name]['Parameters'][k]['default']) != v and k in Parameters}
    non_default_pred_params = {k: v for k, v in str_pred_params.items() if standardize_param_value(api_data[api_name]['Parameters'][k]['default']) != v and k in Parameters}
    # Determine non-None parameters in pred_params
    non_none_pred_params = {k: v for k, v in str_pred_params.items() if v != 'None' and k in Parameters}
    non_none_true_params = {k: v for k, v in str_true_params.items() if v != 'None' and k in Parameters}
    # Calculate correct names and values
    #correct_names = {k: k in non_none_pred_params for k in non_default_true_params.keys() if k in Parameters}
    #correct_values = {k: non_default_true_params[k] == non_none_pred_params.get(k) for k in non_default_true_params if k in non_none_pred_params and k in Parameters}
    # Initialize counts for confusion matrix
    true_positive = 0
    true_negative_wrong = 0
    true_negative_none = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    previous_list = []
    # Calculate confusion matrix values
    for key in non_default_true_params:
        param_type = api_data[api_name]['Parameters'][key]['type']
        processed_type = process_type(param_type)  # 使用你的 process_type 函数分类
        
        if 'Literal' in processed_type:
            category = 'Literal'
        elif any('bool' in p_type.lower() for p_type in processed_type):
            category = 'Boolean'
        else:
            category = 'Others'

        if key in non_none_true_params and key in non_none_pred_params:
            if compare_values(str(non_default_true_params[key]), str(non_none_pred_params[key])):
                true_positive += 1  # Correctly predicted
                category_stats[category]['true_positive'] += 1 
                previous_list.append(key)
            else:
                true_negative_wrong+=1
                true_negative += 1  # Wrong value predicted
                category_stats[category]['true_negative_wrong'] += 1
                category_stats[category]['true_negative'] += 1  
                previous_list.append(key)
        else:
            true_negative_none+=1
            true_negative += 1  # Missed true parameter
            category_stats[category]['true_negative_none'] += 1
            category_stats[category]['true_negative'] += 1  
            previous_list.append(key)
    
    for key in non_default_pred_params:
        param_type = api_data[api_name]['Parameters'][key]['type']
        processed_type = process_type(param_type)
        if 'Literal' in processed_type:
            category = 'Literal'
        elif any('bool' in p_type.lower() for p_type in processed_type):
            category = 'Boolean'
        else:
            category = 'Others'

        if key not in non_default_true_params:
            category_stats[category]['false_positive'] += 1 
            false_positive += 1  # Incorrectly predicted parameter
            previous_list.append(key)
        # No need for else clause here because it's covered in the first loop
    #for key in non_default_true_params:
    #    if key not in non_none_pred_params:
    #        false_negative += 1  # Should not have predicted but did
    for key in api_data[api_name]['Parameters']:
        #if ((api_data[api_name]['Parameters'][key]['type'] not in special_types) and (api_data[api_name]['Parameters'][key]['type'] not in io_types) and (key not in io_param_names)):
        if (not any(t in str(api_data[api_name]['Parameters'][key]['type']) for t in special_types.union(io_types)) and key not in io_param_names):
            if key not in previous_list:
                param_type = api_data[api_name]['Parameters'][key]['type']
                processed_type = process_type(param_type)
                if 'Literal' in processed_type:
                    category = 'Literal'
                elif any('bool' in p_type.lower() for p_type in processed_type):
                    category = 'Boolean'
                else:
                    category = 'Others'
                category_stats[category]['false_negative'] += 1

                false_negative += 1  # Should not have predicted but did
    return {
        "query": query,
        "query_code": query_code,
        "query_id": query_id,
        "api_name": api_name,
        "api_params": Parameters,
        #"correct_names": correct_names,
        #"correct_values": correct_values,
        "true_params": true_params,
        "pred_params": pred_params,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "true_negative_wrong": true_negative_wrong,
        "true_negative_none": true_negative_none,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "category_stats": category_stats
    }

def replace_ellipsis(data):
    if isinstance(data, dict):
        return {k: replace_ellipsis(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_ellipsis(i) for i in data]
    elif data is ...:
        return None
    return data

def calculate_final_metrics_confusion(metrics):
    tp = metrics['true_positive']
    tn_wrong = metrics['true_negative_wrong']
    tn_none = metrics['true_negative_none']
    tn = metrics['true_negative']
    fp = metrics['false_positive']
    fn = metrics['false_negative']
    true_positive_ratio = tp / (tp+tn) if (tp+tn) > 0 else 0
    false_positive_ratio = fp / (tp+fp) if (tp+fp) > 0 else 0
    false_positive_ratio_really = fn / (fp+fn) if (fp+fn) > 0 else 0
    true_negative_wrong_ratio = tn_wrong / (tp+tn) if (tp+tn) > 0 else 0
    true_negative_none_ratio = tn_none / (tp+tn) if (tp+tn) > 0 else 0
    epsilon = 1e-9
    print('fn:', fn, 'tn:', tn, 'tp:', tp, 'fp:', fp)
    total_ratio = true_positive_ratio + true_negative_wrong_ratio + true_negative_none_ratio

    if abs(total_ratio - 1) > epsilon:
        raise AssertionError(f"Ratios do not sum to 1: {total_ratio}")
    
    # Precision: tp / (tp + fp)
    #precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # Recall: tp / (tp + fn)
    #recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # F1 Score: 2 * (precision * recall) / (precision + recall)
    #f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Accuracy: (tp + tn) / (tp + tn + fp + fn)
    #accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    # Param name accuracy based on all params
    #param_name_acc_based_params = metrics['total_correct_params'] / metrics['total_params'] if metrics['total_params'] > 0 else 0
    # Param value accuracy based on all params
    #param_value_acc_based_params = metrics['total_correct_values'] / metrics['total_params'] if metrics['total_params'] > 0 else 0
    # Param value accuracy based on non-empty params
    #param_value_acc_based_non_empty_params = metrics['total_correct_non_empty_values'] / metrics['total_non_empty_params'] if metrics['total_non_empty_params'] > 0 else 0
    return {
        "true_positive_ratio": true_positive_ratio,
        "false_positive_ratio": false_positive_ratio,
        "true_negative_wrong_ratio": true_negative_wrong_ratio,
        "true_negative_none_ratio": true_negative_none_ratio,
        "false_positive_ratio_really": false_positive_ratio_really,
        #"precision": precision,
        #"recall": recall,
        #"f1_score": f1_score,
        #"accuracy": accuracy,
        #"param_name_acc_based_all_params": param_name_acc_based_params,
        #"param_value_acc_based_all_params": param_value_acc_based_params,
        #"param_value_acc_based_non_empty_params": param_value_acc_based_non_empty_params
    }


def process_type(details_type):
    details_type = str(details_type)
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

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def process_results(results, api_data):
    metrics = {
        #'total_correct_params': 0,
        #'total_correct_values': 0,
        'total_params': 0,
        'total_api': 0,
        'total_non_empty_params': 0,
        'total_correct_non_empty_values': 0,
        'true_positive': 0,
        'true_negative': 0,
        'false_positive': 0,
        'false_negative': 0,
        'true_negative_wrong': 0,
        'true_negative_none': 0,
        'total_queries': 0,
        'correct_queries': 0,
        'Literal': {
            'true_positive': 0,
            'true_negative': 0,
            'true_negative_wrong': 0,
            'true_negative_none': 0,
            'false_positive': 0,
            'false_negative': 0,
        },
        'Boolean': {
            'true_positive': 0,
            'true_negative': 0,
            'true_negative_wrong': 0,
            'true_negative_none': 0,
            'false_positive': 0,
            'false_negative': 0,
        },
        'Others': {
            'true_positive': 0,
            'true_negative': 0,
            'true_negative_wrong': 0,
            'true_negative_none': 0,
            'false_positive': 0,
            'false_negative': 0,
        }
    }
    for item_results in results:
        # re-parse the pred_params
        corrected_pred_params = {}
        api_name = item_results['api_name'] ###
        
        returned_content_str_new = item_results['gpt_response1']
        api_name = item_results['api_name']
        predicted_params, success = parse_json_safely(returned_content_str_new)
        #print(returned_content_str_new, predicted_params)
        corrected_pred_params_tmp = post_process_parsed_params(predicted_params, api_name, api_data)
        corrected_pred_params.update(corrected_pred_params_tmp)
        
        returned_content_str_new = item_results['gpt_response2']
        api_name = item_results['api_name']
        predicted_params, success = parse_json_safely(returned_content_str_new)
        #print(returned_content_str_new, predicted_params)
        corrected_pred_params_tmp = post_process_parsed_params(predicted_params, api_name, api_data)
        corrected_pred_params.update(corrected_pred_params_tmp)
        
        returned_content_str_new = item_results['gpt_response3']
        api_name = item_results['api_name']
        predicted_params, success = parse_json_safely(returned_content_str_new)
        #print(returned_content_str_new, predicted_params)
        corrected_pred_params_tmp = post_process_parsed_params(predicted_params, api_name, api_data)
        corrected_pred_params.update(corrected_pred_params_tmp)
        
        item_results['pred_params'] = corrected_pred_params
        # Use the evaluate_parameters function
        eval_result = evaluate_parameters(
            item_results['query'],
            item_results['query_code'],
            item_results['query_id'],
            item_results['api_name'],
            #item_results['api_params'],
            api_data[item_results['api_name']]['Parameters'],
            item_results['true_params'],
            item_results['pred_params'],
            api_data
        )
        item_results.update(eval_result)
        #metrics['total_correct_params'] += sum(eval_result['correct_names'].values())
        #metrics['total_correct_values'] += sum(eval_result['correct_values'].values())
        #metrics['total_params'] += len(item_results['api_params'])
        metrics['total_params'] += len(api_data[item_results['api_name']]['Parameters'])
        metrics['total_api'] += 1
        has_non_empty_params = False
        non_empty_params_count = 0
        correct_non_empty_values_count = 0
        query_correct = True
        
        for param, value in item_results['true_params'].items():
            if value is not None:
                has_non_empty_params = True
                non_empty_params_count += 1
                ############
                if api_data[item_results['api_name']]['Parameters'][param]['description']:
                    #if str(value)==str(api_data[item_results['api_name']]['Parameters'][param]['default']):
                    if compare_values(str(value),str(api_data[item_results['api_name']]['Parameters'][param]['default'])):
                        #raise AssertionError(f"Default value should not be in true_params: {param}={value}")
                        pass
                    else:
                        metrics['total_non_empty_params'] += 1
                        if str(item_results['pred_params'].get(param)) == str(value):
                            metrics['total_correct_non_empty_values'] += 1
                            correct_non_empty_values_count += 1
                        else:
                            query_correct = False
                #else:
                #    print(param)
        if has_non_empty_params:
            metrics['total_queries'] += 1
            if query_correct:
                metrics['correct_queries'] += 1
        
        item_results['has_non_empty_params'] = has_non_empty_params
        item_results['non_empty_params_count'] = non_empty_params_count
        item_results['correct_non_empty_values_count'] = correct_non_empty_values_count
        
        for category in ['Literal', 'Boolean', 'Others']:
            metrics[category]['true_positive'] += eval_result['category_stats'][category]['true_positive']
            metrics[category]['true_negative'] += eval_result['category_stats'][category]['true_negative']
            metrics[category]['true_negative_wrong'] += eval_result['category_stats'][category]['true_negative_wrong']
            metrics[category]['true_negative_none'] += eval_result['category_stats'][category]['true_negative_none']
            metrics[category]['false_positive'] += eval_result['category_stats'][category]['false_positive']
            metrics[category]['false_negative'] += eval_result['category_stats'][category]['false_negative']

        metrics['true_positive'] += eval_result['true_positive']
        metrics['true_negative'] += eval_result['true_negative']
        metrics['true_negative_wrong'] += eval_result['true_negative_wrong']
        metrics['true_negative_none'] += eval_result['true_negative_none']
        metrics['false_positive'] += eval_result['false_positive']
        metrics['false_negative'] += eval_result['false_negative']
    metrics['query_correct_ratio'] = metrics['correct_queries'] / metrics['total_queries'] if metrics['total_queries'] > 0 else 0
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--LIB', type=str, required=True, help='Library name')
    args = parser.parse_args()
    #inquiry_data = load_json(f"./data/standard_process/{args.LIB}/API_inquiry.json")
    api_data = load_json(f"./data/standard_process/{args.LIB}/API_init.json")
    # api_parameters_{args.LIB}_class3_final.json
    # api_parameters_{args.LIB}_class3_final_modified_v3.json
    uncategorized_item = load_json(f'api_parameters_{args.LIB}_final_results.json')
    #uncategorized_item = load_pickle('final_results.pkl')
    metrics = process_results(uncategorized_item, api_data)
    final_metrics = calculate_final_metrics_confusion(metrics)
    plot_confusion_matrix(final_metrics, args.LIB)
    
    print(json.dumps(metrics,indent=4))
    print("Final Metrics across all API calls:")
    print(json.dumps(final_metrics,indent=4))
    results = replace_ellipsis(uncategorized_item)
    with open(f'api_parameters_{args.LIB}_final_final_results.json', 'w') as f:
        json.dump(results, f, indent=4)