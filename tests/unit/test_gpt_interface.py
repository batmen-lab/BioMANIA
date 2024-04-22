from src.gpt.gpt_interface import query_openai

def test_query_openai():
    # Set up the necessary variables for the test
    prompt = "hi"
    mode = 'openai'
    model = 'gpt-3.5-turbo-16k'
    # Call the function and get the response
    response = query_openai(prompt, mode, model)
    # Assert that the response is not empty
    assert response.strip() == "Hello! How can I assist you today?"
