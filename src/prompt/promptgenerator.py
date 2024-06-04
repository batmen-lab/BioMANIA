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
    def build_prompt(self, user_query, api_docstring, parameters_name_list, param_type):
        common_part = f"""
Task Description: Determine from the API DOCSTRING and USER QUERY whether each parameter needs a predicted value. Based on the parameter type and meaning, infer if it needs to fill in a value or modify the default value, and return all the parameters that need prediction or modification with their predicted values.
Instructions:
1. Extract values and assign to parameters: If there are values explicitly appearing in the USER QUERY, such as those quoted with '', values with underscores like value_value, or list/tuple/dict like ['A', 'B'], they are likely to be actual parameter values. When handling these values:
Boolean values: If we are predicting boolean values, you can ignore these explicitly mentioned values.
Integer/float/str/list/tuple values: If we are predicting these types of values, you can assign them to the most corresponding parameters based on the modifiers of the extracted values.
2. Extract and convert values if necessary: Some values might not be obvious and require conversion (e.g., converting "Diffusion Map" to "diffmap" if "Diffusion Map" appears in the user query, as this is the correct usage for some API. Another example is converting "blank space delimiter" to "delimiter=' '"). Ensure that the returned values are correctly formatted, case-sensitive, and directly usable in API calls.
3. Identify relevant parameters: From the USER QUERY and API DOCSTRING, identify which parameters require value extraction or default value modification based on their type and meaning.
4. Never generate fake values: Only include values explicitly extracted in the USER QUERY or that can be clearly inferred. Do not generate fake values for remaining parameters.
5. Response format: Format results as {{"param name": "extracted value"}}, ensuring they are executable and match the correct type from the API DOCSTRING. For example, return ['a', 'b'] for list type instead of Tuple ('a', 'b'). Only return parameters with non-empty values (i.e. not None).
6. Avoid complexity: Avoid using 'or' in parameter values and ensure no overly complex escape characters. The format should be directly loadable by json.loads.
7. Distinguish similar parameters: For similar parameters (e.g., 'knn' and 'knn_max'), distinguish them in meaning and do not mix them up. If there are two parameter candidates with the same meaning for one value, fillin them both with the extracted value.
8. Maintain Format and transfer format if necessary: Ensure the final returned values match the expected parameter type, including complex structures like [("keyxx", 1)] for parameters like "obsm_keys" with types like "Iterable[tuple[str, int]]". If USER QUERY contains list, tuple, or dictionary values, extract them as a whole and keep the format as a value.
"""
        if param_type=='boolean':
            return f"""
{common_part}
Boolean:
- Determine the state (True/False) based on the full USER QUERY and the parameter description, rather than just a part of it. If the 'True/False' value is not explicitly mentioned but the meaning is implied in the USER QUERY, still predict the parameter.

Examples:
USER QUERY: "... with logarithmic axes" => {{"log":"True"}}
USER QUERY: "... without logarithmic axes" => {{"log":"False"}}
USER QUERY: "... hide the default legends and the colorbar legend" => {{"show":"False", "show_colorbar":"False"}}
USER QUERY: "... and create a copy of the data structure" => {{"copy": "True"}}, do not predict "data"

Now start extracting the values with parameters from USER QUERY based on parameters descriptions and default value from API DOCSTRING
USER QUERY: {user_query}
API DOCSTRING: {api_docstring}
param_list: {parameters_name_list}
Return parameter with extracted value in format: {{"param1":"value1", "param2":"value2", ...}}. Only return json format answer in one line, do not return other descriptions.
"""
        elif param_type=='literal':
            return f"""
{param_type}
Literal:
- Find the candidate with the same level of informativeness. If there are multiple candidates containing the keyword, but the query provides only minimal information, select the candidate with the least information content. If a parameter name is mentioned without a value, skip it.

Examples:
USER QUERY: "... with basis 'Diffusion Map'" => {{"basis":'diffmap'}}
USER QUERY: "... blank space delimiter" => {{"delimiter":' '}}

Now start extracting the values with parameters from USER QUERY based on parameters descriptions and default value from API DOCSTRING
USER QUERY: {user_query}
API DOCSTRING: {api_docstring}
param_list: {parameters_name_list}
Return parameter with extracted value in format: {{"param1":"value1", "param2":"value2", ...}}. Only return json format answer in one line, do not return other descriptions.
"""
        else: # if param_type=='int'
            return f"""
{common_part}
int, float, str, list[str], tuple[str]:
- Choose only parameters with explicitly mentioned values in the USER QUERY. If a parameter name is mentioned without a value (e.g., "a specific group/key" where "group/key" is the parameter name but no value is given), ignore that parameter.

Examples:
USER QUERY: "... with chunk size 6000" => {{"chunked":"True", "chunk_size":"6000"}}
USER QUERY: "... with groups ['A', 'B', 'C']" => {{"groups":['A', 'B', 'C']}}
USER QUERY: "... with at least 100 counts." => {{"min_counts": 100}}
USER QUERY: "... for a specific layer 'X_new'?" => {{"layer": "X_new"}}, do not assign "X_new" to "X"

Now start extracting the values with parameters from USER QUERY based on parameters descriptions and default value from API DOCSTRING
USER QUERY: {user_query}
API DOCSTRING: {api_docstring}
param_list: {parameters_name_list}
Return parameter with extracted value in format: {{"param1":"value1", "param2":"value2", ...}}. Only return json format answer in one line, do not return other descriptions.
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
You should Act as a biologist proficient in Python programming, the rules must be strictly followed!
Rules:
- Import all necessary libraries before writing codes.
- Do not deviate from the role of a biologist proficient in Python programming.
- Strictly adhere to all rules.
- Utilize input information to craft a comprehensive plan for achieving the goal.
- Never mess up the input and output directory. You can never plan to load data from output filepath.
- Specify API and function names and avoid including non PyPI functions in your answered code.
- Respond solely in JSON format according to the fixed format.
- Limit response content strictly to JSON enclosed in double quotes.
- Combine steps where possible; do not separate loading data as a distinct step.
- Exclude any extraneous content from your response.
- Provide detailed responses whenever possible.
- You can import and use API from basic tools like pandas, numpy, and anndata. If necessary, you can incorporate some intermediate data preprocessing steps using these basic tools.
- The tone of sub-tasks descriptions should vary among queries: polite, straightforward, casual.\n
Goal: {goal_description}\n
Response Format:
{{"plan": ["Your detailed step-by-step sub-tasks in a list to finish your goal, for example: ['step 1: content', 'step 2: content', 'step 3: content']."]}}
"""
        if data_list:
            prompt+=f"Input: You have the following information in a list with the format `file path: file description`. I provide those files to you, so you don't need to prepare the data. {data_list}"
        else:
            prompt+="You don't have any local data provided. Please use API to load builtin dataset."
        return prompt

class ExecutorPromptBuilder(PromptBuilder):
    def build_prompt(self, executor_info, script, possible_solution="", api_examples="", api_calling=""):
        if possible_solution:
            possible_solution_info = f"\nPossible solution from similar issues from Github Issue Discussion:\n{possible_solution}"
        else:
            possible_solution_info = ""
        if api_examples and api_examples != "{}":
            api_examples_info = f"\nHere are some example usages for API included in script, you might want to refer them to correct the script:\n{api_examples}"
        else:
            api_examples_info = ""
        if api_calling:
            api_calling_info = f"\nHere are the API calling examples, if the script contains wrong API calling please correct it:\n{api_calling}"
        prompt = f"""
Task: Assess the provided Python script based on the log output. The script encountered an error and needs modification. Use the provided information to correct and update the script to avoid these errors.

Rules:
- Import all necessary libraries before writing codes.
- "No such file or directory" is an error.
- Respond in JSON format.
- Only include JSON in your response.
- Provide detailed responses when possible.
- If data preprocessing is needed beyond this subtask, please include it.

Common Python Errors and Solutions:
- FileNotFoundError: Provide the correct file path or use built-in datasets. Never use the output file path to load data!!!
- ModuleNotFoundError: Use the correct module or check for typos.
- AttributeError: Use the correct attribute from the module documentation.
- NameError: If lack a module, import it. If lack a variable, find a correct one, or add prefix code to obtain it.
- TypeError: Ensure variables and parameters match correctly.
- ValueError: Ensure the input to int() is a valid integer string.
- KeyError: Verify the key exists in the dictionary and check for typos.
- NotImplementedError: Implement the functionality using other APIs or libraries.

Log output and possible variables: {executor_info}{possible_solution_info}{api_examples_info}{api_calling_info}\n
Wrong Code Script:{script}\n
Tips: Notice that solutions and suggestions above not guarantee solving the issue. Sometimes the error message is misleading and might not lead to solution directly, and you need to think carefully and logically both from the perspective of data format, obtained variable information from namespace, code script and traceback error message. If API Usage Examples are provided, you can check the co-occurrence of other APIs which might be required as previous running code. Now generate new correct script in JSON format which is different from previous wrong script to avoid the error.
"""
        # Response format: {{"info": "Summary and suggestion."}}
        return prompt

class SubtaskCodePromptBuilder(PromptBuilder):
    def build_prompt(self, data_list, goal_description, history_summary, execute_success=True, execute_info=None):
        prompt = f"""You should Act as a biologist proficient in Python programming, the rules must be strictly followed!
Rules:
- Import all necessary libraries before writing codes.
- Do not deviate from the role of a biologist proficient in Python programming.
- Strictly adhere to all rules.
- Utilize input information to craft a comprehensive plan for achieving the goal.
- Never mess up the input and output directory. You can never plan to load data from output filepath.
- You are provided a python terminal with target PyPI library already installed and imported.
- The history of what you have done is provided, you should take the name changes of some files into account, or use some output from previous steps.
- You should use all information you have to write python codes to finish your current task.
- All code requirements must be followed strictly when you write codes.
- Respond solely in JSON format according to the fixed format.
- Limit response content strictly to JSON enclosed in double quotes.
- Exclude any extraneous content from your response.
- Provide detailed responses whenever possible.
- Do not repeat what you have done in history.

Code requirements:
- Import all necessary libraries before writing codes.
- You have a python terminal environment with target PyPI library already installed and imported.
- You can import and use API from basic tools like pandas, numpy, and anndata. If necessary, you can incorporate some intermediate data preprocessing steps using these basic tools.
- Never forget to import the library before you use any API from this library.
- Only use the PyPI functions and APIs to generate code.
- Pay attention to the number of input files and do not miss any.
- Process each file independently and can not use FOR loop.
- Use the path for all files according to input and history.
- Use the default values for all parameters that are not specified.
- Not write anything else except for your JSON response.
- Not repeat what you have done in history.

History: {history_summary}\n
Current Task: {goal_description}\n
Response Format:
{{"tool": "name of the tool you use", "code": "python code to import library and accomplish the current task"}}"""
        if data_list:
            prompt += f"""
You have the following information in a list with the format `file path: file description`. I provide those files to you, so you don't need to prepare the data: {data_list}"""
        else:
            prompt += f"""
You don't have any local data provided. Please use API to load builtin dataset."""
        final_prompt = prompt
        if execute_success:
            pass
        else:
            final_prompt+=f'''{execute_info}''' # You got this error when you write this code last time. You should solve this bug: {. Here are some suggestions: Based on the error message, could you provide suggestions and solutions for the code to resolve the issue? This might involve running certain prefix APIs, preprocessing data, or addressing any missing attributes, or using correct API instead of fake API due to hallucination.
        return final_prompt

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
        #elif prompt_type == 'execution_correction':
        #    return ExecutionCorrectionPromptBuilder().build_prompt(*args)
        elif prompt_type == 'LLM_code_generation':
            return LLMPromptBuilder().build_prompt(*args)
        elif prompt_type == 'multi_task':
            return MultiTaskPromptBuilder().build_prompt(*args)
        elif prompt_type == 'executor_correction':
            return ExecutorPromptBuilder().build_prompt(*args)
        elif prompt_type == 'subtask_code':
            return SubtaskCodePromptBuilder().build_prompt(*args)
        else:
            raise ValueError("Unknown prompt type")

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    factory = PromptFactory()
    prompt = factory.create_prompt('composite_docstring', "API description", "func_inputs", "func_outputs", "description_text")
    print(prompt)
