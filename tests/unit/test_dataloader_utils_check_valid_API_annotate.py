from unittest.mock import patch
from src.dataloader.check_valid_API_annotate import (
    get_training_and_test_sets,
    check_api_coverage_and_uniqueness,
    check_for_query_text_overlap,
    check_all_queries_unique,
    check_api_presence_in_inquiry,
    compare_inquiries_in_datasets
)

inquiry_data = [
    {'query_id': '1', 'api_calling': ['lib.functionA()'], 'query': 'Different'},
    {'query_id': '2', 'api_calling': ['lib.functionB()'], 'query': 'Different'}
]

annotated_data = [
    {'query_id': '1', 'query': 'Function A', 'api_name': 'lib.functionA'},
    {'query_id': '2', 'query': 'Function B', 'api_name': 'lib.functionB'},
    {'query_id': '3', 'query': 'Function C', 'api_name': 'lib.functionC'}
]

composite_data = {
    'lib.functionA': {},
    'lib.functionB': {}
}

single_data = {
    'lib.functionA': {},
    'lib.functionB': {}
}

def test_get_training_and_test_sets():
    train_data, test_data = get_training_and_test_sets(inquiry_data, annotated_data)
    print(len(train_data), len(test_data))
    assert len(train_data) == 2, "Should have 2 training data entry"
    assert len(test_data) == 1, "Should have 3 test data entry"
    assert train_data[0]['query_id'] == '1', "Training data startswith query_id 1"
    assert test_data[0]['query_id'] == '3', "Test data startswith query_id 3"

def test_check_api_coverage_and_uniqueness(capsys):
    train_data = [{'api_name': 'lib.functionA'}]
    test_data = [{'api_name': 'lib.functionA'}, {'api_name': 'lib.functionC'}]
    check_api_coverage_and_uniqueness(train_data, test_data, 'lib')
    captured = capsys.readouterr()
    assert "API lib.functionC in test data is not in training data." in captured.out

def test_check_for_query_text_overlap(capsys):
    train_data = [{'query': 'Function A'}]
    test_data = [{'query': 'Function A'}]
    check_for_query_text_overlap(train_data, test_data)
    captured = capsys.readouterr()
    assert "Data leakage detected" in captured.out

def test_check_all_queries_unique(capsys):
    queries_data = [{'query': 'Function A'}, {'query': 'Function A'}, {'query': 'Function B'}]
    check_all_queries_unique(queries_data)
    captured = capsys.readouterr()
    assert "Duplicated queries detected" in captured.out

def test_check_api_presence_in_inquiry(capsys):
    inquiry_data = [{'api_calling': ['lib.functionA()'], 'query': 'Function A'}]
    check_api_presence_in_inquiry(single_data, inquiry_data)
    captured = capsys.readouterr()
    assert "All APIs in composite dataset are present in inquiry dataset." not in captured.out

def test_compare_inquiries_in_datasets(capsys):
    compare_inquiries_in_datasets(inquiry_data, annotated_data)
    captured = capsys.readouterr()
    print(captured)
    assert "Inconsistent inquiries detected" in captured.out

@patch('src.dataloader.check_valid_API_annotate.load_json')
def test_main(mock_load_json):
    mock_load_json.side_effect = [
        inquiry_data,
        annotated_data,
        composite_data,
        single_data
    ]
    with patch('sys.argv', ['--LIB', 'lib']):
        from src.dataloader.check_valid_API_annotate import main
        main('scanpy')