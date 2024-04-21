"""
Author: Zhengyuan Dong
Created Date: 2024-04-19
Last Edited Date: 2024-04-19
Description: 
    Check the API dependency between different sections in the tutorial.
"""

import json
import matplotlib.pyplot as plt
from gpt.utils import save_json, load_json

def filter_apis(summarized_data, api_init):
    tut_apis = {}
    valid_apis = set(api_init.keys())  # Extract all valid API names from API_init data
    for section in summarized_data:
        if section not in tut_apis:
            tut_apis[section] = []
        for item in summarized_data[section]:
            if 'ori_relevant_API' in item:
                # Only add APIs that are present in the API_init data
                filtered_apis = [api for api in item['ori_relevant_API'] if api in valid_apis]
                tut_apis[section].extend(filtered_apis)
    return tut_apis

def generate_api_pairs(apis, window_size=3):
    """
    Generate pairs of API sequences and the next API based on a specified window size.
    
    Args:
    apis (dict): A dictionary where the key is the section and the value is a list of APIs.
    window_size (int): The number of APIs to include in the current sequence before the next API.

    Returns:
    dict: A dictionary with the same keys as the input, where the value is a list of tuples.
          Each tuple contains a list of current APIs and the next API.
    """
    api_pairs = {}
    for section, api_list in apis.items():
        api_pairs[section] = [(api_list[i:i+window_size], api_list[i+window_size]) for i in range(len(api_list) - window_size)]
    return api_pairs

def match_types(return_types, input_types, specific_type=['AnnData', 'DataFrame', 'ndarray', 'spmatrix']): # =
    # Simplifying the check: any return type in the input types list, considering possible optional prefixes
    for r_type in return_types:
        for i_type in input_types:
            if r_type in i_type or (r_type.startswith('Optional[') and r_type[9:-1] in i_type):
                if specific_type:
                    if (r_type in specific_type) or (r_type[9:-1] in specific_type):
                        return True
                else:
                    return True
    return False

def process_api_sequences(api_pairs, api_data):
    results = []
    for section, pairs in api_pairs.items():
        for current_apis, next_api in pairs:
            # Collect all return types from current APIs
            current_return_types = [api_data[api].get('return_type', []) for api in current_apis]
            # Flatten the list of return types
            current_return_types = [rtype for sublist in current_return_types for rtype in sublist]
            # Get input types for the next API
            next_input_types = api_data[next_api].get('parameter_type', [])
            next_parameters = [param for param_details in api_data[next_api].get('Parameters', {}).values() for param in param_details.keys()]
            # Check if there's a match
            match = match_types(current_return_types, next_input_types)
            results.append({
                'section': section,
                'current_apis': current_apis,
                'next_api': next_api,
                'current_return_types': current_return_types,
                'next_input_types': next_input_types,
                'next_parameters': next_parameters,
                'match': match
            })
    return results

def transform_api_data(api_init):
    api_data = {}
    for api_name, details in api_init.items():
        # Initialize the structure for each API
        if api_name not in api_data:
            api_data[api_name] = {'return_type': [], 'parameter_type': []}
        # Process return types
        if 'Returns' in details and 'type' in details['Returns']:
            return_type = details['Returns']['type']
            if return_type:
                api_data[api_name]['return_type'].append(return_type)
        # Process parameter types
        if 'Parameters' in details:
            for param, param_details in details['Parameters'].items():
                param_type = param_details.get('type', None)
                if param_type:
                    api_data[api_name]['parameter_type'].append(param_type)
    return api_data

def calculate_match_rate(results):
    total = len(results)
    matches = sum(1 for result in results if result['match'])
    return {
        'total_pairs': total,
        'matches': matches,
        'match_rate': matches / total if total > 0 else 0
    }

def deduplicate_tutorials(tutorials):
    keys_to_remove = set()
    keys = list(tutorials.keys())
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            if i != j and key_i in key_j:
                # Check if the APIs lists are the same
                if tutorials[key_i] == tutorials[key_j]:
                    # Mark the more complex key for removal
                    more_complex_key = key_i if 'dot' in key_i else key_j
                    keys_to_remove.add(more_complex_key)
    # Remove the marked keys
    for key in keys_to_remove:
        del tutorials[key]
    return tutorials

def main():
    summarized_data = load_json('./data/autocoop/scanpy/summarized_responses.json')
    API_init = load_json('./data/standard_process/scanpy/API_init.json')
    # Filter the APIs from the summarized data
    tut_apis = filter_apis(summarized_data, API_init)
    print('tut_apis:', len(tut_apis))
    # there are some duplicated tutorials, we need to remove them
    tut_apis = deduplicate_tutorials(tut_apis)
    print('tut_apis:', len(tut_apis))
    print(tut_apis.keys())
    # Transform the API data
    api_data = transform_api_data(API_init)
    match_rates = []
    window_sizes = range(1, 10)
    for window_size in window_sizes:
        # Generate API pairs
        api_pairs = generate_api_pairs(tut_apis, window_size)
        # Process the API sequences
        results = process_api_sequences(api_pairs, api_data)
        # Save the results
        save_json(f'api_results_win_{window_size}.json', results)
        # Calculate the match rate
        match_stats = calculate_match_rate(results)
        save_json(f'api_match_stats_win_{window_size}.json', match_stats)
        match_rates.append(match_stats['match_rate'])
    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(list(window_sizes), match_rates, marker='o', linestyle='-', color='b')
    plt.title('Window Size vs Match Rate')
    plt.xlabel('Window Size')
    plt.ylabel('Match Rate')
    plt.grid(True)
    plt.xticks(list(window_sizes))
    plt.savefig('window_size_vs_match_rate.png')
    plt.show()

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == '__main__':
    main()
