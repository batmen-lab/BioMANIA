import pytest
import json
import os
import numpy as np
from src.inference.utils import json_to_docstring, compress_and_save_image, predict_by_similarity, find_similar_pairs, bert_embed

parameters = {
    "param1": {"type": "int", "default": 10, "optional": True, "description": "An example integer parameter"},
    "param2": {"type": "str", "default": "default", "optional": True, "description": "An example string parameter"}
}

def test_json_to_docstring():
    result = json_to_docstring("example_function", "This is a test function.", parameters)
    expected_signature = "def example_function(param1: int = 10, param2: str = default):"
    assert expected_signature in result
    assert "This is a test function." in result
    assert "param1" in result and "An example integer parameter" in result
    assert "param2" in result and "An example string parameter" in result

def test_predict_by_similarity():
    user_query_vector = np.array([[1, 2, 3]])
    centroids = np.array([[1, 2, 3], [4, 5, 6]])
    labels = ["class1", "class2"]
    predicted_label = predict_by_similarity(user_query_vector, centroids, labels)
    assert predicted_label == "class1", "Should correctly predict the closest class"

@pytest.fixture
def setup_ml_model():
    from transformers import BertTokenizer, BertModel
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

def test_bert_embed(setup_ml_model):
    model, tokenizer = setup_ml_model
    text = "example text"
    embedding = bert_embed(model, tokenizer, text)
    assert embedding.shape[0] == 768, "BERT embedding size should be 768"

