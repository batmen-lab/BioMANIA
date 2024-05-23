"""
Author: Zhengyuan Dong
Date Created: January 16, 2024
Last Modified: May 22, 2024
Description: aggregate prompts for all tasks
"""

from abc import ABC, abstractmethod

class PromptBuilder(ABC):
    @abstractmethod
    def build_prompt(self, *args):
        pass

class CompositeDocstringPromptBuilder(PromptBuilder):
    # from prompt/composite.py
    def build_prompt(self, API_description, func_inputs, func_outputs, description_text):
        return f"""
Write a concise docstring for an invisible function in Python, focusing solely on its core functionality derived from the sequential composition of sub APIs.
- API Description: {API_description}
- Parameters: {func_inputs}
- Returns: {func_outputs}
- Additional Description: {description_text}
Craft a 1-2 sentence docstring that extracts and polishes the core information. The response should be in reStructuredText format, excluding specific API names and unprofessional terms. Remember to use parameter details only to refine the core functionality explanation, not for plain input/output information.
"""

class CompositeNamePromptBuilder(PromptBuilder):
    # from prompt/composite.py
    def build_prompt(self, sub_API_names, llm_docstring):
        return f"""Your task is to suggest an appropriate name for the given invisible function:
- Here are the sub API used together with function's docstring, please consider the API name to generate function name. sub API names: {sub_API_names}, 
function docstring: ```{llm_docstring}```
- Your name should consist of 4 to 5 keywords that combined with `_`, name should be recognizable and contain as much information as you can in keywords, and should display API information in a sequential order.
Your Response format: {{'func_name': (your designed function name)}}
Please do not include other information except for response format.
"""

class InstructionGenerationPromptBuilder(PromptBuilder):
    # from prompt/instruction.py
    def build_prompt(self, API_name, docs):
        return f"""
Instruction: Generate 10 examples that uses the function {API_name}(...) to train an intelligent code assistant for computational biology. Each example should be a JSON dictionary with fields "instruction" and "code". The text field is a single line of language instruction the user would say to the code assistant. The code field is a single line of code the code assistant should generate given the user's instruction. Use one line per example. Be specific when writing the user instruction - write it as if a human user would say it to the assistant. Never include function name or parameters name. Vary tones among queries: polite, straightforward, casual. Emphasize intent and methods of the function and distinguish between Target and Contrasting descriptions. For example, given Target description "Scatter plot in UMAP basis" Contrasting description ["Scatter plot in PHATE basis"], generated examples `Please create a scatter plot in the UMAP basis`should include intent of "create a scatter plot" and details "in the UMAP basis", omitting "PHATE specifics".""",f"""Refer to the following code documentation on what each function does, and the distinctions between Target and Contrasting descriptions. \ndocstring of {docs}\n"""

class ParametersPromptBuilder(PromptBuilder):
    # from prompt/parameters.py
    def build_prompt(self, user_query, api_docstring, parameters_name_list):
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

class SummaryPromptBuilder(PromptBuilder):
    # from prompt/summary.py
    def build_prompt(self, user_query, api_function, api_description, parameters_description, return_information):
        return f"""
Help to summarize the task with its solution in layman tone, it should be understandable by non professional programmers. Starts from description of the user query `{user_query}` and includes the selected API function `{api_function}` which functionality is `{api_description}` with its parameters `{parameters_description}`. The returned variable is `{return_information}`
Please use the template as `The task is ..., we solved it by ...`. The response should be in three sentences
"""

class SummaryPromptFullBuilder(PromptBuilder):
    # from prompt/summary.py
    def build_prompt(self, user_query, api_function, api_description, parameters_description, return_information, execution_code):
        return f"""
Help to summarize the task with its solution in layman tone, it should be understandable by non professional programmers. Starts from description of the user query `{user_query}` and includes the selected API function `{api_function}` which functionality is `{api_description}` with its parameters `{parameters_description}`. The returned variable is `{return_information}`. The generated code is `{execution_code}`.
Please use the template as `The task is ..., we solved it by ...`. The response should be in four sentences. Additionally, the interpretation encompasses explanations for the parameters utilized in the generated code.
"""

class ExecutionCorrectionPromptBuilder(PromptBuilder):
    def build_prompt(self, user_input, history_record, error_code, error_message, variables, LIB):
        return f"""
Your task is to correct a Python code snippet based on the provided information. The user's inquiry is represented by '{user_input}'. The history of successful executions is provided in '{history_record}', and variables in the namespace are supplied in a dictionary '{variables}'. Execute the error code snippet '{error_code}' and capture the error message '{error_message}'. Analyze the error to determine its root cause. Then, using the entire API name instead of abbreviation in the format '{LIB}.xx.yy'. Ensure any new variables created are named with the prefix 'result_' followed by digits, without reusing existing variable names. If you feel necessary to perform attribute operations similar to 'result_1.var_names.intersection(result_2.var_names)' or subset an AnnData object by columns like 'adata_ref = adata_ref[:, var_names]', go ahead. If you feel importing some libraries are necessary, go ahead. Maintain consistency with the style of previously executed code. Ensure that the corrected code, given the variables in the namespace, can be executed directly without errors. Return the corrected code snippet in the format: '\"\"\"your_corrected_code\"\"\"'. Do not include additional descriptions.
"""# please return minimum line of codes that you think is necessary to execute for the task related inquiry

class LLMPromptBuilder(PromptBuilder):
    def build_prompt(self, user_input, history_record, variables, LIB):
        return f"""
Your task is to generate a Python code snippet based on the provided information. The user's inquiry is represented by '{user_input}'. The history of successful executions is provided in '{history_record}', and variables in the namespace are supplied in a dictionary '{variables}'. Analyze the user intent about the task to generate code. Then, using the entire API name instead of abbreviation in the format '{LIB}.xx.yy'. Ensure any new variables created are named with the prefix 'result_' followed by digits, without reusing existing variable names. If you feel necessary to perform attribute operations similar to 'result_1.var_names.intersection(result_2.var_names)' or subset an AnnData object by columns like 'adata_ref = adata_ref[:, var_names]', go ahead. If you feel importing some libraries are necessary, go ahead. Maintain consistency with the style of previously executed code. Ensure that the generated code, given the variables in the namespace, can be executed directly without errors. Return the corrected code snippet in the format: '\"\"\"your_corrected_code\"\"\"'. Do not include additional descriptions.
"""# please return minimum line of codes that you think is necessary to execute for the task related inquiry

class MultiTaskPromptBuilder(PromptBuilder):
    def build_prompt(self, goal_description, data_list=[]):
        prompt = f"""
Act as a biologist proficient in Python programming, the rules must be strictly followed! 
Rules are: 
- Do not deviate from the role of a biologist proficient in Python programming. 
- Strictly adhere to all rules. 
- Utilize input information to craft a comprehensive plan for achieving the goal. 
- Specify API and function names and avoid including non PyPI functions in your answered code. 
- Respond solely in JSON format according to the fixed format. 
- Limit response content strictly to JSON enclosed in double quotes. 
- Combine steps where possible; do not separate loading data as a distinct step. 
- Exclude any extraneous content from your response. 
- Provide detailed responses whenever possible. 
- Goal: {goal_description}. 
- The fixed format for JSON response is: {{\"plan\": [\"Your detailed step-by-step sub-tasks in a list to finish your goal, for example: ['step 1: content', 'step 2: content', 'step 3: content']\"]}}"
"""
        if data_list:
            prompt+="For data, you have the following information in a list with the format `file path: file description`. I provide those files to you, so you don't need to prepare the data."
        else:
            prompt+="For data, you don't have any local data provided. Please use API to load builtin dataset."
        return prompt

class PromptFactory:
    def create_prompt(self, prompt_type, *args):
        if prompt_type == 'composite_docstring':
            return CompositeDocstringPromptBuilder().build_prompt(*args)
        elif prompt_type == 'composite_name':
            return CompositeNamePromptBuilder().build_prompt(*args)
        elif prompt_type == 'instruction_generation':
            return InstructionGenerationPromptBuilder().build_prompt(*args)
        elif prompt_type == 'parameters':
            return ParametersPromptBuilder().build_prompt(*args)
        elif prompt_type == 'summary':
            return SummaryPromptBuilder().build_prompt(*args)
        elif prompt_type == 'summary_full':
            return SummaryPromptFullBuilder().build_prompt(*args)
        elif prompt_type == 'execution_correction':
            return ExecutionCorrectionPromptBuilder().build_prompt(*args)
        elif prompt_type == 'LLM_code_generation':
            return LLMPromptBuilder().build_prompt(*args)
        elif prompt_type == 'multi_task':
            return MultiTaskPromptBuilder().build_prompt(*args)
        else:
            raise ValueError("Unknown prompt type")

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    factory = PromptFactory()
    prompt = factory.create_prompt('composite_docstring', "API description", "func_inputs", "func_outputs", "description_text")
    print(prompt)
