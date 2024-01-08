"""
Author: Zhengyuan Dong
Date Created: January 08, 2024
Last Modified: January 08, 2024
Description: Add validation check for API_inquiry_annotate.json.
"""


import json
import argparse

def load_data(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def get_split_query_id(inquiry_data):
    """Get the split query id as the last query id in inquiry data plus one."""
    last_query_id = inquiry_data[-1]["query_id"]
    return last_query_id + 1

def check_api_uniqueness(train_data, test_data):
    """Check if each API in the training dataset appears only once in the test dataset."""
    train_apis = set(item["api_name"] for item in train_data)
    test_apis = {item["api_name"]: 0 for item in train_data}

    for item in test_data:
        if item["api_name"] in test_apis:
            test_apis[item["api_name"]] += 1

    for api, count in test_apis.items():
        assert count == 1, f"API {api} appears {count} times in the test dataset, not just once"
    print("Each API in the training dataset appears only once in the test dataset.")

def check_for_data_leakage(train_data, test_data):
    """Check for data leakage: test queries should not appear in the training dataset."""
    train_query_ids = set(item["query_id"] for item in train_data)
    data_leakage = any(item["query_id"] in train_query_ids for item in test_data)
    assert not data_leakage, "Data leakage detected: Some test query_ids are present in the training dataset"
    print("No data leakage detected.")

def check_query_uniqueness(data):
    """Check that the same query is not repeated for the same API."""
    api_queries = {}
    for item in data:
        api_queries.setdefault(item["api_name"], set()).add(item["query_id"])

    for api, queries in api_queries.items():
        assert len(queries) == len([item for item in data if item["api_name"] == api]), f"Query IDs for API {api} are not unique."
    print("Query IDs are unique for each API.")

def main():
    parser = argparse.ArgumentParser(description="Check data integrity for training and testing datasets.")
    parser.add_argument("lib", type=str, help="Library name for the JSON data.")
    args = parser.parse_args()

    inquiry_data = load_data(f'./data/standard_process/{args.lib}/API_inquiry.json')
    annotated_data = load_data(f'./data/standard_process/{args.lib}/API_inquiry_annotate.json')

    split_query_id = get_split_query_id(inquiry_data)

    train_data = [item for item in annotated_data if item["query_id"] < split_query_id]
    test_data = [item for item in annotated_data if item["query_id"] >= split_query_id]

    check_api_uniqueness(train_data, test_data)
    check_for_data_leakage(train_data, test_data)
    check_query_uniqueness(annotated_data)

    print("All checks passed successfully.")

if __name__ == "__main__":
    main()
