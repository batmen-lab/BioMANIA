import pytest
from src.dataloader.preprocess_retriever_data import unify_response_format, preprocess_json_string, parse_json_response

def test_unify_response_format_valid_json():
    input_json = '[{"key": "value"}, {"key2": "value2"}]'
    expected_output = [{"key": "value"}, {"key2": "value2"}]
    assert unify_response_format(input_json) == expected_output

def test_unify_response_format_invalid_json():
    input_json = '[{key: "value"}, {key2: "value2"}]'
    expected_output = []
    assert unify_response_format(input_json) == expected_output

def test_unify_response_format_partially_invalid_json():
    assert len(unify_response_format('[{"key1": "value1"}, [{"key2": "value2"}]]')) == 2
    response = 'Invalid JSON [{"key1": "value1"}, {"key2": "value2"}] Invalid JSON'
    assert len(unify_response_format(response)) == 2
    response = 'Invalid JSON'
    assert len(unify_response_format(response)) == 0
    response = '[{`key1`: "value1"}, [{"key2": "value2"}]]'
    assert len(unify_response_format(response)) == 0

def test_preprocess_json_string_single_quotes():
    json_string = "{'key': 'value'}"
    expected_string = '{"key": "value"}'
    assert preprocess_json_string(json_string) == expected_string

def test_preprocess_json_string_mixed_quotes():
    json_string = "{'key': \"value's\"}"
    expected_string = '{"key": "value\'s"}'
    assert preprocess_json_string(json_string) == expected_string


@pytest.mark.parametrize("input_data,expected", [
    ('[{"key": "value"}]', [{"key": "value"}]),
    ('[{"key": "value"}, {"key2": "value2"}]', [{"key": "value"}, {"key2": "value2"}]),
    ('[{"key": "value", "key2": "value2"}]', [{"key": "value", "key2": "value2"}]),
])
def test_parse_json_response_valid_input(input_data, expected):
    assert parse_json_response(input_data) == expected

def test_parse_json_response_invalid_input():
    input_data = '[{key: "value"}]'
    expected = []
    assert parse_json_response(input_data) == expected

