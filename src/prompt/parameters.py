# prompt
def prepare_parameters_prompt(user_query, api_description, api_name, api_parameters_information, parameters_name_list):
    return f"""
    USER QUERY: {user_query}
    API DESCRIPTION: {api_description}
    API NAME: {api_name}
    API PARAMETERS INFORMATION: 
    {api_parameters_information}
    You are skilled coder, you have been provided USER QUERY, API DESCRIPTION, API NAME, and API PARAMETERS INFORMATION. Your assignment involves two primary tasks. 

    First, you need to evaluate whether a parameter has been assigned its designated value in the user's query. Afterward, if the value is indeed assigned, you should proceed to assign that value to its respective parameters. The parameters should only from: {parameters_name_list}

    The order for assigning value is that, you should first use the value you find from query. If not find, then use the default value. You should only return the parameters with assigned value, no matter using default value or found value from query. If the parameters has no searched value from query and no default value, do not include it in your response. If you're unable to find any value or validate a match or default value, simply return an empty list.

    Restricted to the response format: [{{'param_name': parameters name, 'value': the value you find}}].
    Your answer should be a list of valid json.
    Do not offer any additional information.
    The value should be found from USER QUERY instead of other information.
    Please carefully review the user query and API description to ensure that the extracted values match the intended user meaning. 
    """
