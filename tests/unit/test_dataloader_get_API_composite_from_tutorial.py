from src.dataloader.get_API_composite_from_tutorial import (classify_code_blocks, extract_function_name, apply_unknown_code_blocks,collect_imports, merge_jsons, rearrange_code_blocks,separate_nodes, separate_code_blocks, save_list_code, get_html_description, get_final_API_from_docstring, wrap_function, insert_docstring, put_docstring_into_code, filter_line)

def test_classify_code_blocks():
    code_blocks = [
        "import os\nimport sys",
        "def function():\n    pass",
        "print('Hello, World!')"
    ]
    pre_code_list = ["import os"]
    expected = [
        {'code': "import os\nimport sys", 'type': 'import'},
        {'code': "def function():\n    pass", 'type': 'def'},
        {'code': "print('Hello, World!')", 'type': 'unknown'}
    ]
    assert classify_code_blocks(code_blocks, pre_code_list) == expected

def test_extract_function_name():
    code = "def my_function():\n    pass"
    assert extract_function_name(code) == "my_function"

    code_no_function = "print('Hello')"
    assert extract_function_name(code_no_function) is None

def test_apply_unknown_code_blocks():
    classified_blocks = [
        {'code': "def function():\n    pass", 'type': 'def'},
        {'code': "print(function())", 'type': 'unknown'}
    ]
    result = apply_unknown_code_blocks(classified_blocks)
    assert result[1]['type'] == 'apply'

def test_collect_imports():
    code = "import os\nimport sys\nprint('Hello')"
    imports, non_imports = collect_imports(code)
    assert imports == "import os\nimport sys"
    assert non_imports == "print('Hello')"

def test_merge_jsons():
    list_of_dicts = [{'a': 1}, {'b': 2}]
    assert merge_jsons(list_of_dicts) == {'a': 1, 'b': 2}

def test_rearrange_code_blocks():
    code_blocks = ["print('Hello')", "import os"]
    expected = ["import os", "print('Hello')"]
    assert rearrange_code_blocks(code_blocks) == expected

def test_separate_nodes():
    from ast import parse
    nodes = parse("def function():\n    pass\nprint('Hello')").body
    separated = separate_nodes(nodes)
    assert len(separated) == 2

def test_separate_code_blocks():
    code_blocks = ["def function():\n    pass\nprint('Hello')"]
    separated = separate_code_blocks(code_blocks)
    assert len(separated) == 2 

def test_save_list_code(tmp_path):
    file_path = tmp_path / "output.txt"
    save_list_code(["line1", "line2"], str(file_path))
    with open(file_path, 'r') as file:
        lines = file.readlines()
    assert lines == ["line1\n", "line2\n"]

def test_get_html_description():
    combined_blocks = [{'code': "print('Hello')", 'text': "Description"}]
    assert get_html_description(combined_blocks, "print('Hello')") == "Description"

def test_get_final_API_from_docstring():
    docstring = """Function does something.
    
    Parameters
    ----------
    param1 : int
        Description of param1.
        
    Returns
    -------
    int
        Description of return.
    """
    result = get_final_API_from_docstring(docstring)
    assert 'Parameters' in result
    assert 'param1' in result['Parameters']

def test_wrap_function():
    code = "print('Hello')"
    wrapped, call = wrap_function(code, ['data'], ['result'], 'wrapper')
    assert 'def wrapper(data):' in wrapped
    assert 'result = wrapper(data)' == call

def test_insert_docstring():
    source_code = "def function():\n    pass"
    docstring = "This function does nothing"
    expected = "def function():\n    \"\"\"This function does nothing\"\"\"\n    pass"
    assert insert_docstring(source_code, docstring) == expected

def test_put_docstring_into_code():
    code = "def function():\n    pass"
    new_docstring = "Updated docstring"
    new_function_name = "new_function"
    result = put_docstring_into_code(code, new_docstring, new_function_name)
    assert "new_function" in result
    assert "Updated docstring" in result

def test_filter_line():
    code_block = "print('Hello')\nprint('World')"
    assert filter_line(code_block) == ""
