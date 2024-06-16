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
    def build_prompt(self, LIB, goal_description, data_list=[]):
        prompt = f"""
Based on the provided task description, create a task plan with subtasks. 
The tone of Each subtask should vary among queries: polite, straightforward, casual. 
Each subtask should correspond to the exact usage scope of one single API from library {LIB}.
Avoid creating steps that are too coarse or too detailed. If you find that more than one API needs to be used, split the step into two or more subtasks. For example, if for the preprocessing data step, filtering and normalization are both required, then use two subtasks `preprocessing data by filtering` and `preprocessing data by normalization` to describe them separately.
Only include keywords in the subtask, omit API name from all subtasks.
Each subtask should consists of 15-20 words, should be clear and concise for one single API usage.
The arrangement of tasks should take into account API order and dependencies (for example, some APIs need to calculate metrics before visualization) and the logical order of tasks (for example, an example flow is to load data first, then preprocess data, then apply methods, and finally visualize the results).
To improve task planning and make it more user-friendly, integrate visualization subtasks between specific key subtasks.
If a file path is provided, use it to load the data. If no file path is provided, use the built-in dataset API to load the default dataset. If several filepaths are provided, either use them in one subtask or in different subtasks regarding the task description.
Never include data description in other subtasks except for the data loading subtask. Ensure Goal-Oriented Task Structuring, place the goal description at the beginning of each subtask.
Only respond in JSON format strictly enclosed in double quotes, adhering to the Response Format.
Exclude any extraneous content from your response.
Goal: {goal_description}\n
Response Format:
{{"plan": ["List your detailed step-by-step tasks to achieve your goal, e.g., ['step 1: content', 'step 2: content', 'step 3: content']."]}}
""" # Specify API and function names and avoid including non PyPI functions in your answered code.
        if data_list:
            prompt+=f"Input: You have the following information in a list with the format `file path: file description`. I provide those files to you, so you don't need to prepare the data. {data_list}"
        else:
            prompt+="You don't have any local data provided. Please only use API to load builtin dataset. Please avoid any API that need to load local data."
        return prompt

class ExecutorPromptBuilder(PromptBuilder):
    def build_prompt(self, executor_info, parameters_info, namespace_variables, error_code, possible_solution="", api_examples="", api_calling="", success_history_code="", goal_description=""):
        if possible_solution:
            possible_solution_info = f"\nPossible solution from similar issues from Github Issue Discussion:\n{possible_solution}"
        else:
            possible_solution_info = ""
        if api_examples and api_examples != "{}":
            api_examples_info = f"\nAPI Usage examples: {api_examples}."
        else:
            api_examples_info = ""
        prompt = f"""
Task: Analyze and correct the Python script based on the traceback information. Here are some information to help you analyze the error in code:
Success execution History: {success_history_code}
Current goal for code generation: {goal_description}
We try below codes for this task several times, but all fails: {error_code}
traceback error information from the last trial: {executor_info}
Current Namespace variables: {namespace_variables}
{possible_solution_info}{api_examples_info}
API calling template: {api_calling}. 
Parameters information for this API calling: {parameters_info}
You only need to keep required parameters from previous trial codes, only keep minimum optional parameters necessary for task. Remove optional parameters from error code which cause the problem. Please ensure that required parameters are passed in their proper positional order, as keyword arguments should only be used for optional parameters. You only need to include the task related correct code in your response, do not repeat other API from the success execution history in your response. For parameters starting with 'result_', use only those that exist in the namespace. Do not generate inexist variables.

Below are some common reasons, debug based on error types:
Import Verification: Ensure all necessary libraries are imported.
API: No matter the API inexists or the external lib is uninstalled, replace with the correct or similar API if necessary; otherwise, continue with the same API.
Parameter Names: Remove unnecessary optional parameters and keep only those essentials for successful execution. Remove fake parameters that not belong to target API.
Attribute and Values: Correct any incorrect parameter values. For AnnData object attributes, only fillin existing attributes in namespace variables instead of using hallucinated attributes.
Previous steps needed: Some pre-API are required for API executions due to the API design. If so, ensure these steps are included in the corrected code before target API call. E.g., before visualization, you might need to calculate metrics to store it in anndata object first. E.g., some API require to input logarimize data, you need to logarimize the data by another API first.
For required parameters, pass the values directly without specifying parameter names, like api(required_param1). For optional parameters, specify them explicitly by name, like api(required_param1, optional_param1=value).
If the data needs intermediate processing, address these by setting appropriate parameters or another API if possible. If not, use tools like AnnData, pandas related API to preprocess the data before calling the corresponding API from {api_calling}.
Sometimes errors are indirect; deduce and locate the real cause then correct it. Present the logic steps in the "analysis" part, while present the code in the "code" part

Rules:
- Conduct minimum correction.
- Import all necessary libraries at the beginning of the script.
- Include any prerequisite steps required for the task if you feel it is necessary for API dependency, e.g. in order to use API2, API1 must be executed ahead.
- Respond only with the answer in JSON format.
- Only successful execution history is recorded. Each time, remember to import the targeted API again in correct code, remember to use the exists variable, do not use variable from error trials as they are not recognized as part of the execution history.

Response Format:
{{"analysis": "Locate error, explain how to correct the bug.", "code": "Task oriented corrected bug-free Python code based on analysis."}}
"""
        # Response format: {{"info": "Summary and suggestion."}}
        return prompt
    
class ModifySubtaskPromptBuilder(PromptBuilder):
    def build_prompt(self, current_subtask,  execution_history, namespace_variables, api_docs):
        query_prompt = f'''
Code Execution History: {execution_history}
Namespace Variables: {namespace_variables}
Current Subtask: {current_subtask}
API documentation: {api_docs}
Your Task: Based on the Total task planning, current subtask, execution history prior to the current subtasks, namespace variables, and relevant API documentation, please rewrite the subtask description. The rewritten description should correspond to the most specific API and include only the necessary parameters and their values to clearly describe the subtask. Maintain a tone that is polite, neutral, or formal, as if it were a user inquiry. 
**IMPORTANT**
Never include data description in other subtasks except for the data loading subtask. Ensure Goal-Oriented Task Structuring, place the goal description at the beginning of each subtask.
Ensure to check docstring requirements for API dependencies, required optional parameters, parameter conflicts, and deprecations.
Just response with the modified subtask description directly. DO NOT add additional explanations or introducement.
'''
        return query_prompt

class SubtaskCodePromptBuilder(PromptBuilder):
    def build_prompt(self, data_list, goal_description, history_summary, execute_success=False, execute_info=None):
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
        elif prompt_type == 'modify_subtask':
            return ModifySubtaskPromptBuilder().build_prompt(*args)
        else:
            raise ValueError("Unknown prompt type")

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    factory = PromptFactory()
    prompt = factory.create_prompt('composite_docstring', "API description", "func_inputs", "func_outputs", "description_text")
    print(prompt)
