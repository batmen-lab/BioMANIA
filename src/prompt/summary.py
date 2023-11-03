
def prepare_summary_prompt(user_query, api_function, api_description, parameters_description,return_information):
    return f"""
    Help to summarize the task with its solution in layman tone, it should be understandable by non professional programmers. Starts from description of the user query `{user_query}`
and includes the selected API function `{api_function}` which functionality is `{api_description}` with its parameters `{parameters_description}`. The returned variable is `{return_information}`
Please use the template as `The task is ..., we solved it by ...`. The response should be in three sentences
"""
