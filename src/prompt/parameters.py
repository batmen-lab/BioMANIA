# prompt
def prepare_parameters_prompt(user_query, api_docstring, parameters_name_list):
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
