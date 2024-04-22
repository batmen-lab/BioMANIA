import pandas as pd
import os
from src.deploy.utils import dataframe_to_markdown, convert_image_to_base64, change_format

def test_dataframe_to_markdown():
    """ Test the conversion of a DataFrame to a Markdown table format. """
    data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
    df = pd.DataFrame(data)
    expected_output = (
        "| Name | Age |\n"
        "| --- | --- |\n"
        "| Alice | 25 |\n"
        "| Bob | 30 |"
    )
    result = dataframe_to_markdown(df)
    assert result == expected_output

def test_convert_image_to_base64():
    """ Test the conversion of an image file to a Base64 encoded string. """
    test_image_path = 'test_image.png'
    with open(test_image_path, 'wb') as f:
        f.write(os.urandom(10))

    result = convert_image_to_base64(test_image_path)
    assert result is not None
    
    os.remove(test_image_path)
    
    result = convert_image_to_base64('non_existent_image.png')
    assert result is None

def test_change_format():
    """ Test filtering and restructuring of input parameters. """
    input_params = {
        'param1': {'type': 'int', 'description': 'an integer', 'default': 5},
        'param2': {'type': 'str', 'description': 'a string', 'default': 'default'}
    }
    param_name_list = ['param1']
    expected_output = [{
        'name': 'param1',
        'type': 'int',
        'description': 'an integer',
        'default_value': 5
    }]
    result = change_format(input_params, param_name_list)
    assert result == expected_output

    result = change_format(input_params, [])
    assert result == []
