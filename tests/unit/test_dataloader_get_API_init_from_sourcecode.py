import pytest
from unittest.mock import patch, MagicMock, mock_open
from src.dataloader.get_API_init_from_sourcecode import (get_dynamic_types, type_to_string, expand_types, resolve_forwardref, format_type)
import typing

def test_get_dynamic_types():
    result = get_dynamic_types()
    assert int in result
    assert str in result
    assert typing.Union in result

@pytest.mark.parametrize("test_input,expected", [
    (int, "int"),
    (typing.List[int], "List"),
    (typing.Tuple[int, str], "Tuple"),
    (typing.Dict[str, int], "Dict"),
])
def test_type_to_string(test_input, expected):
    assert type_to_string(test_input) == expected

def test_expand_types():
    assert expand_types("int | str | float") == ["int", "str", "float"]
    assert expand_types("int or str or float") == ["int", "str", "float"]
    assert expand_types("int") == ["int"]

def test_resolve_forwardref():
    assert resolve_forwardref("int") == int
    assert resolve_forwardref("Optional[int]") == typing.Optional[int]

def test_format_type():
    from typing import Optional, Union
    assert format_type(int) == "int"
    assert format_type(typing.Optional[typing.List[int]]) == "Optional[List[int]]"
    assert format_type(Optional[int]) == "Optional[int]"
    assert format_type(Union[int, float]) == "Union[int, float]"
    assert format_type(typing.ForwardRef("str")) == "str"
