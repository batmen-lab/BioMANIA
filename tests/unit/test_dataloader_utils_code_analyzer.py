from src.dataloader.utils.code_analyzer import extract_io_variables, is_variable_in_parentheses
import pytest
import pandas as pd
import numpy as np

@pytest.mark.parametrize("code, namespace, expected_output", [
    (
        "df = pd.DataFrame(data)\nresult = df[df['A'] > 5].groupby('B').sum()",
        {"pd": pd, "data": {"A": [1, 2, 3, 10], "B": [5, 6, 7, 8]}},
        ({"data"}, {"df", "result"})
    ),
    (
        "arr = np.array(data); mean_val = np.mean(arr)",
        {"np": np, "data": [1, 2, 3]},
        ({"data"}, {"arr", "mean_val"})
    ),
    (
        "for i in range(len(data)):\n\tprocessed_data.append(data[i] + 5)",
        {"data": [1, 2, 3], "processed_data": []},
        ({"data", "processed_data"}, {"i", "processed_data"})
    ),
    (
        "import scipy.stats\nresult = scipy.stats.zscore(data)",
        {"data": [1, 2, 3, 4, 5]},
        ({"data"}, {"result"})
    )
])
def test_extract_io_variables(code, namespace, expected_output):
    assert extract_io_variables(code, namespace)[0] == expected_output[0]

def test_is_variable_in_parentheses():
    code = "if (var == 5):"
    assert is_variable_in_parentheses("var", code) == True
    code = "if var == 5:"
    assert is_variable_in_parentheses("var", code) == False
    code = "if ((var) == 5):"
    assert is_variable_in_parentheses("var", code) == True
    code = "if (var + 5) == 10:"
    assert is_variable_in_parentheses("var", code) == True
    code = "(variable == 5)"
    assert is_variable_in_parentheses("var", code) == True