
"""
Author: Zhengyuan Dong
Created Date: 2024-01-17
Last Edited Date: 2024-01-23
Email: zydong122@gmail.com
Description: 
    Check whether there exist data leakage, duplicate queries, missing API for annotated data.
"""
import inspect, os
from typing import Tuple
from ..gpt.utils import load_json

def get_training_and_test_sets(inquiry_data: list, annotated_data: list) -> Tuple[list, list]:
    """
    Separates the annotated data into training and test sets based on the presence of their query IDs in the inquiry data.

    Parameters
    ----------
    inquiry_data : list
        The list of inquiry data dictionaries, each containing a query_id.
    annotated_data : list
        The list of annotated data dictionaries, each containing a query_id.

    Returns
    -------
    Tuple[list, list]
        A tuple containing two lists: the training data and the test data.
    """
    inquiry_query_ids = set(item["query_id"] for item in inquiry_data)
    train_data = [item for item in annotated_data if item["query_id"] in inquiry_query_ids]
    test_data = [item for item in annotated_data if item["query_id"] not in inquiry_query_ids]
    return train_data, test_data

def check_api_coverage_and_uniqueness(train_data: list, test_data: list, LIB: str) -> None:
    """
    Ensures that each API from the training data appears only once in the test data and that all APIs in the test data
    are present in the training data. It also filters APIs by a given library prefix.

    Parameters
    ----------
    train_data : list
        List of dictionaries representing the training data, each containing an "api_name".
    test_data : list
        List of dictionaries representing the test data, each containing an "api_name".
    LIB : str
        The library prefix to filter API names by.
    """
    train_apis = {item["api_name"] for item in train_data}
    # filter class and composite API
    train_apis = {i for i in list(train_apis) if i.startswith(LIB)}
    test_apis = {api_name: 0 for api_name in train_apis}

    for item in test_data:
        if item["api_name"] in test_apis:
            test_apis[item["api_name"]] += 1
        else:
            print(f"API {item['api_name']} in test data is not in training data.")

def check_for_query_text_overlap(train_data: list, test_data: list) -> None:
    """
    Checks for any overlap in query texts between the training and test datasets to ensure no data leakage.

    Parameters
    ----------
    train_data : list
        List of dictionaries representing the training data, each containing a "query".
    test_data : list
        List of dictionaries representing the test data, each containing a "query".
    """
    train_queries = {item["query"] for item in train_data}
    test_queries = {item["query"] for item in test_data}
    overlapping_queries = train_queries.intersection(test_queries)
    if overlapping_queries:
        print(f"Data leakage detected: Overlapping query texts in training and test datasets: {overlapping_queries}")

def check_all_queries_unique(annotated_data: list) -> None:
    """
    Ensures that all query texts in the annotated data are unique.

    Parameters
    ----------
    annotated_data : list
        A list of annotated data dictionaries, each containing a "query".
    """
    queries = [item["query"] for item in annotated_data]
    unique_queries = set(queries)
    if len(queries) != len(unique_queries):
        duplicated_queries = {q for q in queries if queries.count(q) > 1}
        print(f"Duplicated queries detected: {duplicated_queries}")
    else:
        print("All queries are unique.")

def check_api_presence_in_inquiry(composite_data: dict, inquiry_data: list) -> None:
    """
    Checks if all APIs listed in the composite dataset are present in the inquiry dataset.

    Parameters
    ----------
    composite_data : dict
        A dictionary representing composite data where keys are API names.
    inquiry_data : list
        A list of inquiry data dictionaries, each containing an "api_calling" entry with API names.
    """
    composite_apis = {item for item in composite_data}
    inquiry_apis = {item['api_calling'][0].split('(')[0] for item in inquiry_data}
    print(f'length of composite/inquiry is {len(composite_apis)}, {len(inquiry_apis)}')
    missing_apis = composite_apis - inquiry_apis
    if missing_apis:
        print(f"Missing APIs in inquiry dataset: {missing_apis}")
    else:
        print("All APIs in composite dataset are present in inquiry dataset.")

def compare_inquiries_in_datasets(inquiry_data: list, annotated_data: list) -> None:
    """
    Compares the inquiries between the inquiry data and annotated data based on their query IDs to ensure consistency.

    Parameters
    ----------
    inquiry_data : list
        A list of inquiry data dictionaries, each containing a "query_id" and "query".
    annotated_data : list
        A list of annotated data dictionaries, each containing a "query_id" and "query".
    """
    inquiry_dict = {item["query_id"]: item["query"] for item in inquiry_data}
    annotated_dict = {item["query_id"]: item["query"] for item in annotated_data}
    common_query_ids = set(inquiry_dict.keys()) & set(annotated_dict.keys())
    print('common_query_ids: ', len(common_query_ids))
    inconsistent_inquiries = []
    for query_id in common_query_ids:
        if inquiry_dict[query_id] != annotated_dict[query_id]:
            inconsistent_inquiries.append(query_id)
    if inconsistent_inquiries:
        print(f"Inconsistent inquiries detected for query_id(s): {inconsistent_inquiries}")
    else:
        print("All matching query_ids have consistent inquiries between inquiry_data and annotated_data.")

def main(lib_data_path:str, LIB: str) -> None:
    """
    Main function that loads data and performs checks to ensure data integrity for training and testing datasets.

    Parameters
    ----------
    lib_data_path: str
        The path to the library data.
    LIB : str
        The library name for the JSON data to be processed.
    """
    inquiry_data = load_json(os.path.join(lib_data_path, 'API_inquiry.json'))
    annotated_data = load_json(os.path.join(lib_data_path, 'API_inquiry_annotate.json'))
    single_data = load_json(os.path.join(lib_data_path, 'API_init.json'))
    train_data, test_data = get_training_and_test_sets(inquiry_data, annotated_data)
    inquiry_api_names = set()
    for inquiry in inquiry_data:
        for api_call in inquiry["api_calling"]:
            # Extracting API name before parentheses
            api_name = api_call.split("(")[0]
            inquiry_api_names.add(api_name)
    # Extract API names from API_init.json
    init_api_names = set(single_data.keys())
    # Find the API names that are in inquiry but not in init (the ones missing)
    missing_apis = init_api_names - inquiry_api_names
    print('missing_apis: ', missing_apis)
    # Apply the checks
    check_api_coverage_and_uniqueness(train_data, test_data, LIB)
    check_for_query_text_overlap(train_data, test_data)
    check_all_queries_unique(annotated_data)
    check_api_presence_in_inquiry(single_data, inquiry_data)
    compare_inquiries_in_datasets(inquiry_data, annotated_data)
    print("All checks passed successfully.")

__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check data integrity for training and testing datasets.")
    parser.add_argument("--LIB", type=str, help="Library name for the JSON data.")
    args = parser.parse_args()
    
    from ..configs.model_config import get_all_variable_from_cheatsheet
    info_json = get_all_variable_from_cheatsheet(args.LIB)
    LIB_DATA_PATH = info_json['LIB_DATA_PATH']
    
    main(LIB_DATA_PATH, args.LIB)
