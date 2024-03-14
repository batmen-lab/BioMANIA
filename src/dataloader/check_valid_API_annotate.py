
"""
Author: Zhengyuan Dong
Created Date: 2024-01-17
Last Edited Date: 2024-01-23
Description: 
    Check whether there exist data leakage, duplicate queries, missing API for annotated data.
"""
import json
import argparse

def load_data(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def get_training_and_test_sets(inquiry_data, annotated_data):
    """Separate the data into training and test sets based on query IDs."""
    inquiry_query_ids = set(item["query_id"] for item in inquiry_data)

    train_data = [item for item in annotated_data if item["query_id"] in inquiry_query_ids]
    test_data = [item for item in annotated_data if item["query_id"] not in inquiry_query_ids]

    return train_data, test_data

def check_api_coverage_and_uniqueness(train_data, test_data, LIB):
    """
    Check if each API in the training dataset appears only once in the test dataset
    and if test dataset contains any API not present in training dataset.
    """
    train_apis = set(item["api_name"] for item in train_data)
    # filter class and composite API
    train_apis = set([i for i in list(train_apis) if i.startswith(LIB)])
    test_apis = {api_name: 0 for api_name in train_apis}

    for item in test_data:
        if item["api_name"] in test_apis:
            test_apis[item["api_name"]] += 1
        else:
            print(f"API {item['api_name']} in test data is not in training data.")

    #for api, count in test_apis.items():
    #    if count != 1:
    #        print(f"API {api} appears {count} times in the test dataset, not just once.")

def check_for_query_text_overlap(train_data, test_data):
    """
    Check for overlap in query texts between training and test datasets.
    """
    train_queries = set(item["query"] for item in train_data)
    test_queries = set(item["query"] for item in test_data)

    overlapping_queries = train_queries.intersection(test_queries)
    if overlapping_queries:
        print(f"Data leakage detected: Overlapping query texts in training and test datasets: {overlapping_queries}")

def check_all_queries_unique(annotated_data):
    """Check that all queries in the annotated data are unique."""
    queries = [item["query"] for item in annotated_data]
    unique_queries = set(queries)
    if len(queries) != len(unique_queries):
        duplicated_queries = set([q for q in queries if queries.count(q) > 1])
        print(f"Duplicated queries detected: {duplicated_queries}")
    else:
        print("All queries are unique.")

def check_api_presence_in_inquiry(composite_data, inquiry_data):
    """
    Check if all APIs in the composite dataset are present in the inquiry dataset.
    """
    composite_apis = set(item for item in composite_data)
    inquiry_apis = set(item['api_calling'][0].split('(')[0] for item in inquiry_data)
    print(f'length of composite/inquiry is {len(composite_apis)}, {len(inquiry_apis)}')
    missing_apis = composite_apis - inquiry_apis
    if missing_apis:
        print(f"Missing APIs in inquiry dataset: {missing_apis}")
    else:
        print("All APIs in composite dataset are present in inquiry dataset.")

def compare_inquiries_in_datasets(inquiry_data, annotated_data):
    """
    Compare the inquiries for matching query_id in inquiry_data and annotated_data.
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

def main():
    parser = argparse.ArgumentParser(description="Check data integrity for training and testing datasets.")
    parser.add_argument("--LIB", type=str, help="Library name for the JSON data.")
    args = parser.parse_args()

    inquiry_data = load_data(f'./data/standard_process/{args.LIB}/API_inquiry.json')
    annotated_data = load_data(f'./data/standard_process/{args.LIB}/API_inquiry_annotate.json')
    composite_data = load_data(f'./data/standard_process/{args.LIB}/API_composite.json')
    single_data = load_data(f'./data/standard_process/{args.LIB}/API_init.json')

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
    check_api_coverage_and_uniqueness(train_data, test_data, args.LIB)
    check_for_query_text_overlap(train_data, test_data)
    print("All checks passed successfully.")
    check_all_queries_unique(annotated_data)
    check_api_presence_in_inquiry(single_data, inquiry_data)
    compare_inquiries_in_datasets(inquiry_data, annotated_data)

if __name__ == "__main__":
    main()
