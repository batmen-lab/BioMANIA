# prompt
def prepare_parameters_prompt(user_query, api_docstring, parameters_name_list):
    # modified on 240519
    return f"""
USER QUERY: {user_query}
API DOCSTRING: {api_docstring}
Task Description: Extract and format parameter values from a USER QUERY based on the parameters information provided from API DOCSTRING. Focus on parameters listed in {parameters_name_list}, ensuring values are ready for API usage.
Instructions:
- Identify all parameters from {parameters_name_list}.
- Extract values for parameters explicitly mentioned or inferred from the USER QUERY. Use context clues to infer parameter values when not explicitly mentioned.
- Only return parameters that should be modified from their default values based on the USER QUERY.
- Format results as [{{"param_name": "name", "value": "extracted value"}}], ensuring the values are executable and match the type specified in the API DOCSTRING.
- Avoid using 'or' in parameter values and do not use descriptions as values. Ensure all extracted values are properly formatted, case-sensitive, and directly usable in API calls.
- For parameters of type 'int', 'str', etc., extract the possible value from the inquiry. For parameters of type 'Literal', extract the value from the provided candidate list.
- Pay extra attention to boolean types, and infer their values based on their descriptions and overall context of the USER QUERY.
- Ensure that every parameter mentioned in the query is matched with a corresponding parameter in {parameters_name_list}. Use the provided list, tuple, or dictionary values to predict parameters. Extract numerical values for prediction.
- Ensure the response matches the parameters' type specified in the API DOCSTRING. For example, do not return [] for a parameter expecting Tuple ().
- Separate different parameters. For example, if the inquiry is 'I want to hide the default legends and the colorbar legend in the dot plot', return two parameters: "show": "False" and "show_colorbar": "False".
- Pay attention to synonyms. For example, for the query "Could you downsample counts from count matrix, allowing counts to be sampled with replacement?", infer the appropriate parameters.
- For boolean parameters, carefully consider the overall query context and the parameter description to determine if the value should be True or False.
- For similar parameters like 'knn' and 'knn_max', assign values based on the specific details mentioned in the USER QUERY. Ensure the values are correctly assigned to the appropriate parameters.
- If the inquiry contains information like "Diffusion Map", infer and extract the correct parameter values to ensure correct execution, such as converting "Diffusion Map" to "diffmap", and converting "blank space delimiter" to "delimiter=' '".
- Ensure the final returned values match the expected parameter type, including complex structures like [("keyxx", 1)] for parameters such as "obsm_keys" with types like "Iterable[tuple[str, int]]".

In-context examples for inferring values:
  - If the inquiry mentions "with logarithmic axes", infer log=True.
  - If the inquiry states "without returning axis information", infer show=True if the parameter description implies this functionality.
  - If the inquiry mentions "chunk_size 6000", infer chunked=True in addition to setting chunk_size=6000.
  - For list, tuple, or dictionary values, ensure all mentioned elements are included. For example, if the inquiry includes "['A', 'B', 'C']", extract ['A', 'B', 'C'] and find the corresponding parameters to assign this value.
"""

def bak_bak_prepare_parameters_prompt(user_query, api_docstring, parameters_name_list):
    # before 240519
    return f"""
USER QUERY: {user_query}
API Docstring: {api_docstring}
You are tasked with extracting parameter values from a USER QUERY and API DOCSTRING. Your focus is on parameters that are explicitly mentioned or implied within the query. Parameters not mentioned should not be included, even if they have default values. Return the identified parameters with their assigned values in the format [{{"param_name": "name", "value": "assigned value"}}], only considering those listed in {parameters_name_list}. If no specific values can be assigned to these mentioned parameters, return an empty list. Prioritize parameters critical to the specific API call, aligning with user intent and API requirements."""

def bak_prepare_parameters_prompt(user_query, api_description, api_name, api_parameters_information, parameters_name_list):
    # before 240315
    return f"""
    USER QUERY: {user_query}
    API DESCRIPTION: {api_description}
    API NAME: {api_name}
    API PARAMETERS INFORMATION: 
    {api_parameters_information}
    You are skilled coder, you have been provided USER QUERY, API DESCRIPTION, API NAME, and API PARAMETERS INFORMATION. Your assignment involves two primary tasks. 

    First, you need to evaluate whether a parameter has been assigned its designated value in the user's query. Afterward, if the value is indeed assigned, you should proceed to assign that value to its respective parameters. The parameters should only from: {parameters_name_list}

    The order for assigning value is that, you should first use the value you find from query. If not find, then use the default value. You should only return the parameters with assigned value, no matter using default value or found value from query. If the parameters has no searched value from query and no default value, do not include it in your response. If you're unable to find any value or validate a match or default value, simply return an empty list.

    Restricted to the response format: [{{"param_name": parameters name, "value": the value you find}}].
    Your answer should be a list of valid json.
    Do not offer any additional information.
    The value should be found from USER QUERY instead of other information.
    Please carefully review the user query and API description to ensure that the extracted values match the intended user meaning. 
    Note that I will kindly mentioned the parameters name as keyword. Please only select the parameters I mentioned!!!
    Do not include the parameters that is not clearly mentioned in query!!!
    """
