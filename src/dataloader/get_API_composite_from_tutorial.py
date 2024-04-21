import ast, re, os, json, astor, ast
from itertools import chain
from datetime import datetime
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
from docstring_parser import parser

from configs.model_config import ANALYSIS_PATH, get_all_variable_from_cheatsheet #tut, html_dict, code
from dataloader.utils.tutorial_loader_strategy import main_convert_tutorial_to_py
#from dataloader.utils.code_analyzer import extract_io_variables
from models.model import LLM_model, LLM_response
from prompt.composite import build_prompt_for_composite_docstring, build_prompt_for_composite_name
from typing import Optional, Any, Tuple
from gpt.utils import load_json, save_json

def classify_code_blocks(code_blocks: list, pre_code_list: list) -> list:
    """
    Classifies code blocks into different types such as import, unknown, or definition based on their content.

    Parameters
    ----------
    code_blocks : list
        A list of code blocks (strings) to be classified.
    pre_code_list : list
        A list of pre-existing code elements to consider during classification.

    Returns
    -------
    list
        A list of dictionaries, each containing the classified code block and its type.
    """
    classified_blocks = []
    for block in code_blocks:
        lines = block.split('\n')
        is_import = False
        open_parenthesis = False
        for line in lines:
            line = line.strip()
            if line.startswith(('import', 'from')):
                is_import = True
                if '(' in line and not line.endswith(')'):
                    open_parenthesis = True
                elif open_parenthesis and line.endswith(')'):
                    open_parenthesis = False
            elif line and line in pre_code_list:
                is_import = True
            elif line and not line.startswith(('import', 'from')):
                is_import = False
                break
        if is_import and not open_parenthesis:
            block_type = 'import'
        elif not any(re.search(r'^\s*(?:def|class)\s+', line) for line in lines):
            block_type = 'unknown'
        else:
            block_type = 'def'
        classified_block = {'code': block, 'type': block_type}
        classified_blocks.append(classified_block)
    return classified_blocks

def extract_function_name(code: str) -> Optional[str]:
    """
    Extract the function name from a block of code.

    Parameters
    ----------
    code : str
        The block of code from which to extract the function name.

    Returns
    -------
    Optional[str]
        The name of the function if found; otherwise, None.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None

def apply_unknown_code_blocks(classified_blocks: list) -> list:
    """
    Process blocks of code that are classified as 'unknown' to determine if they can be applied to known function definitions.

    Parameters
    ----------
    classified_blocks : list
        A list of dictionaries containing classified code blocks.

    Returns
    -------
    list
        The list of code blocks after processing unknown types.
    """
    def_blocks = []
    # Find all function names and def code blocks
    for block in classified_blocks:
        if block['type'] == 'def':
            tree = ast.parse(block['code'])
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    def_blocks.append(block)
    # Modify 'unknown' type code blocks that satisfy the condition
    for i in range(len(classified_blocks)):
        block = classified_blocks[i]
        if block['type'] == 'unknown':
            for def_block in def_blocks:
                function_name = extract_function_name(def_block['code'])
                if function_name and function_name in block['code']:
                    block['type'] = 'apply'
                    break
    return classified_blocks

def collect_imports(code: str) -> Tuple[str, str]:
    """
    Separates import statements from other statements in a block of code.

    Parameters
    ----------
    code : str
        The block of code to analyze.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the import part and the non-import part of the code.
    """
    import_statements = []
    non_import_statements = []
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.Import):
            import_statements.append(ast.unparse(node))
        elif isinstance(node, ast.ImportFrom):
            import_statements.append(ast.unparse(node))
        else:
            non_import_statements.append(ast.unparse(node))
    import_part = '\n'.join(import_statements)
    non_import_part = '\n'.join(non_import_statements)
    return import_part, non_import_part

def merge_jsons(list_of_dicts: list) -> dict:
    """
    Merges multiple dictionaries into a single dictionary.

    Parameters
    ----------
    list_of_dicts : list
        A list of dictionaries to be merged.

    Returns
    -------
    dict
        The resulting dictionary after merging.
    """
    merged_json = {}
    merged_dict = dict(chain.from_iterable(d.items() for d in list_of_dicts))
    merged_json.update(merged_dict)
    return merged_json

def rearrange_code_blocks(code_blocks: list) -> list:
    """
    Rearranges code blocks by separating import statements from other code.

    Parameters
    ----------
    code_blocks : list
        A list of code blocks to rearrange.

    Returns
    -------
    list
        A new list of code blocks, starting with all imports followed by other statements.
    """
    import_block = []
    other_blocks = []
    for block in code_blocks:
        # remove comments first, otherwise it would affect from/import extraction
        import_part, non_import_part = collect_imports(block)
        if import_part:
            import_block.append(import_part)
        if non_import_part.strip():  # Check if non-import part is not empty
            other_blocks.append('\n'.join([non_import_part]))  # add comments back
    new_code_blocks = import_block + other_blocks
    return new_code_blocks

def separate_nodes(nodes: list) -> list:
    """
    Separates nodes into distinct code blocks based on control structures.

    Parameters
    ----------
    nodes : list
        A list of AST nodes.

    Returns
    -------
    list
        A list of separated code blocks.
    """
    code_blocks = []
    current_block = []
    for node in nodes:
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef) or isinstance(node, ast.For) or isinstance(node, ast.While):
            if current_block:
                code_blocks.append(current_block)
                current_block = []
            code_blocks.append([node])
        else:
            current_block.append(node)
    if current_block:
        code_blocks.append(current_block)
    return ["\n".join([ast.unparse(block) for block in code_block]) for code_block in code_blocks]

def separate_code_blocks(code_blocks: list) -> list:
    """
    Separates complex code blocks into simpler, independent blocks using AST parsing.

    Parameters
    ----------
    code_blocks : list
        A list of complex code blocks to separate.

    Returns
    -------
    list
        A list of simpler, independent code blocks.
    """
    separated_blocks = []
    for code in code_blocks:
        try:
            tree = ast.parse(code)
            separated_blocks.extend(separate_nodes(tree.body))
        except:
            pass
    return separated_blocks

def save_list_code(json_list: list, filename: str) -> None:
    """
    Saves a list of strings to a file, each string on a new line.

    Parameters
    ----------
    json_list : list
        A list of strings to be saved.
    filename : str
        The path to the file where the list will be saved.
    """
    with open(filename, 'w') as f:
        f.writelines([i+'\n' for i in json_list])

def get_html_description(combined_blocks: list, input_code: str) -> str:
    """
    Retrieves a description from HTML blocks that match the given input code.

    Parameters
    ----------
    combined_blocks : list
        A list of combined code and text blocks from HTML.
    input_code : str
        The input code to match against the combined blocks.

    Returns
    -------
    str
        The description associated with the input code, if found; otherwise, an empty string.
    """
    input_code_lines = input_code.split('\n')
    for block in combined_blocks:
        if 'code' in block and 'text' in block:
            if all(line.strip() in block['code'] for line in input_code_lines):
                return block['text']
    return ''

def get_final_API_from_docstring(modified_docstring: str) -> dict:
    """
    Extracts API information from a modified docstring.

    Parameters
    ----------
    modified_docstring : str
        The docstring from which to extract API information.

    Returns
    -------
    dict
        A dictionary containing extracted API information such as parameters and return values.
    """
    docs = parser.parse(modified_docstring)
    parameters = {}
    for param in docs.params:
        param_dict = {
            "type": str(param.type_name),
            "default": None,
            "optional": False,
            "description": param.description
        }
        parameters[param.arg_name] = param_dict
    need_saved_API = {
        "Parameters": parameters,
        "Returns": str(docs.returns),
        "Docstring": modified_docstring.strip(),
        "example": ""
    }
    return need_saved_API

def wrap_function(code: str, input_nodes: list, output_nodes: list, wrapper_name: str) -> Tuple[str, str]:
    """
    Wraps a given code block into a function with specified inputs and outputs.

    Parameters
    ----------
    code : str
        The code block to wrap.
    input_nodes : list
        A list of input variable names.
    output_nodes : list
        A list of output variable names.
    wrapper_name : str
        The name of the function wrapper.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the wrapped function code and the function call string.
    """
    function_signature = f"def {wrapper_name}({', '.join(input_nodes)}):\n"
    indented_code = '\n'.join(['    ' + line for line in code.split('\n') if line])
    return_statement = f"    return {', '.join(output_nodes)}\n" if output_nodes else ""
    wrapped_code = function_signature + indented_code + "\n" + return_statement
    function_call = f"{wrapper_name}({', '.join(input_nodes)})"
    if output_nodes:
        function_call = ', '.join(output_nodes) + ' = ' + function_call
    return wrapped_code, function_call.lstrip()

def insert_docstring(source_code: str, docstring: str) -> str:
    """
    Inserts a docstring into the source code just below the function or class definition.

    Parameters
    ----------
    source_code : str
        The source code into which the docstring will be inserted.
    docstring : str
        The docstring to insert.

    Returns
    -------
    str
        The source code with the docstring inserted.
    """
    lines = source_code.split("\n")
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if line.lstrip().startswith("def ") or line.lstrip().startswith("class "):
            new_lines.append('    """' + docstring + '"""')
    return "\n".join(new_lines)

def put_docstring_into_code(code: str, modified_docstring: str, new_function_name: str) -> str:
    """
    Inserts a modified docstring into a piece of code and optionally renames the function.

    Parameters
    ----------
    code : str
        The original code block.
    modified_docstring : str
        The modified docstring to insert.
    new_function_name : str
        The new name for the function, if renaming is desired.

    Returns
    -------
    str
        The code with the modified docstring inserted and function renamed.
    """
    # Parse the code to an AST
    module = ast.parse(code)
    first_stmt = module.body[0]
    if isinstance(first_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Modify function name
        first_stmt.name = new_function_name
        first_stmt.body.insert(0, ast.Expr(value=ast.Constant(value=modified_docstring)))
    modified_code = astor.to_source(module)
    return modified_code

def filter_line(code_block: str) -> str:
    """
    Filters out unwanted lines from a code block, such as print statements.

    Parameters
    ----------
    code_block : str
        The code block to filter.

    Returns
    -------
    str
        The filtered code block.
    """
    # remove all print lines
    # remove last line
    lines = code_block.split("\n")
    lines = [i for i in lines if 'print(' not in i]
    if len(lines)==0:
        return ""
    last_line = lines[-1]
    if ((".save" in last_line) or (".set" in last_line) or (".show" in last_line) or ("plt." in last_line)):
        lines = lines
    elif (not ("=" in last_line) and
        not ("(" in last_line) and
        not ("import" in last_line) and
        not last_line.endswith(")") and
        not last_line.startswith(" ")):
        lines = lines[:-1]
    elif (not ("=" in last_line) and
          (".head(" in last_line)):
        lines = lines[:-1]
    """elif (not ("=" in last_line) and
        not ("import" in last_line) and
        not last_line.startswith(" ")):
        lines = lines[:-1]+['print('+lines[-1]+')']"""
    # if exist '.show(', add `plt.savefig('./tmp/timestamp')`
    current_time = datetime.now()
    filename = current_time.strftime("%Y%m%d%H%M%S") + ".png"
    directory = "./tmp"
    filepath = os.path.join(directory, filename)
    if '.show(' in code_block:
        lines+=[f'plt.savefig({filepath})']
    return "\n".join(lines)

def individual_tut(tut: str, html_dict: dict, output_folder_pyjson: str, pre_code_list: list, lib_alias: str) -> list:
    """
    Processes individual tutorial files to extract and classify code blocks.

    Parameters
    ----------
    tut : str
        The tutorial identifier.
    html_dict : dict
        A dictionary containing HTML content.
    output_folder_pyjson : str
        The output folder path for storing processed JSON data.
    pre_code_list : list
        A list of pre-existing code to consider during processing.
    lib_alias : str
        The library alias used to identify relevant code blocks.

    Returns
    -------
    list
        A list of processed and classified code blocks.
    """
    local_namespace = {}
    api_new_tmp = []
    print('-'*10)
    print('start ', tut,'!')
    code_blocks = ['\n'.join(i['code']) for i in html_dict[tut]]
    #code_blocks = [filter_last_line(i) for i in code_blocks] # some last line prints a variable, not used in py scripts
    new_code_blocks = code_blocks
    #new_code_blocks = rearrange_code_blocks(new_code_blocks)
    new_code_blocks = separate_code_blocks(new_code_blocks)
    modified_blocks = classify_code_blocks(new_code_blocks,pre_code_list)
    #modified_blocks = apply_unknown_code_blocks(classified_blocks)

    # get imports, saved in tmp files
    import_statements = []
    for block in modified_blocks:
        if block['type'] == 'import':
            import_statements.extend(block['code'].split('\n'))
            try:
                exec(block['code'],local_namespace)
            except Exception as e:
                print('import error: ', e)
    modified_blocks = [i for i in modified_blocks if i['type']!='import']
    modified_blocks = [{'code':'\n'.join(import_statements),'type':'import'}] + modified_blocks
    # get APIs for each code block
    unique_apis_sets = []
    code_blocks_processed = []
    imports = {}
    for block in modified_blocks:
        if block['type'] == 'import':
            imports.update(extract_imports(block['code']))
        else:
            current_apis = extract_api_calls(block['code'], imports, lib_alias)
            # remove duplicate API inside code block
            current_apis_set = set(current_apis)
            # remove same code block API subsets
            if current_apis_set not in unique_apis_sets:
                unique_apis_sets.append(current_apis_set)
                html_search_text = get_html_description(html_dict[tut], block['code'])
                code_blocks_processed.append({'code': block['code'], 'APIs': list(current_apis_set), 'text': html_search_text})
    # filter code block which contains only one API. We only concern those multiple API co-operate
    code_blocks_filtered = []
    for block in code_blocks_processed:
        if len(block['APIs']) > 1:
            code_blocks_filtered.append(block)
    """ # deprecate the io extraction as it is too time-consuming and not reliable
    # start wrapped
    #pre_code = ''
    index = 0
    iii = 0
    while iii<len(modified_blocks):
        #iii+=1
        #block = modified_blocks[iii-1]
        #block['code'] = filter_line(block['code'])
        # if only one api line, skip
        #if len(block['code'].split('\n'))<=1:
        #    pass
        # get html description
        #wrapper_name = f"wrapper_{index}"
        #index+=1
        # extract API
        #input_nodes, output_nodes = extract_io_variables(block['code'], local_namespace) # extract io parameters
        #wrapped_func, function_call = wrap_function(block['code'],input_nodes, output_nodes,wrapper_name)"""
    return code_blocks_filtered
def extract_imports(code_block: str) -> dict:
    """
    Extracts import statements from a block of code.

    Parameters
    ----------
    code_block : str
        The block of code from which to extract imports.

    Returns
    -------
    dict
        A dictionary of imports with aliases as keys and full paths as values.
    """
    imports = {}
    try:
        root = ast.parse(code_block)
        for node in ast.walk(root):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports[name.asname if name.asname else name.name] = name.name
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    imports[name.asname if name.asname else name.name] = node.module + '.' + name.name if node.module else name.name
    except:
        pass
    return imports

def find_api_calls(node: Any, imports: dict, lib_alias: str) -> list:
    """
    Recursively finds API calls in the AST node that correspond to the specified library alias.

    Parameters
    ----------
    node : Any
        The AST node to search for API calls.
    imports : dict
        A dictionary of imports used to resolve API call paths.
    lib_alias : str
        The library alias used to identify relevant API calls.

    Returns
    -------
    list
        A list of fully qualified API call paths.
    """
    api_calls = []
    def get_full_path(node):
        if isinstance(node, ast.Attribute):
            return get_full_path(node.value) + '.' + node.attr
        elif isinstance(node, ast.Name):
            return node.id
        return ''
    if isinstance(node, ast.Call):
        if isinstance(node.func, (ast.Attribute, ast.Name)):
            api_call_path = get_full_path(node.func)
            path_elements = api_call_path.split('.')
            if path_elements[0] in imports and imports[path_elements[0]].startswith(lib_alias):
                api_call = '.'.join([imports[path_elements[0]]] + path_elements[1:])
                api_calls.append(api_call)
    for child in ast.iter_child_nodes(node):
        api_calls.extend(find_api_calls(child, imports, lib_alias))
    return api_calls

def extract_api_calls(code_block: str, imports: dict, lib_alias: str) -> list:
    """
    Extracts API calls from a block of code based on the specified library alias.

    Parameters
    ----------
    code_block : str
        The block of code from which to extract API calls.
    imports : dict
        A dictionary of imports used to resolve API call paths.
    lib_alias : str
        The library alias used to identify relevant API calls.

    Returns
    -------
    list
        A list of fully qualified API call paths.
    """
    try:
        root = ast.parse(code_block)
        return find_api_calls(root, imports, lib_alias)
    except SyntaxError:
        return []

def process_docstring_with_LLM(llm: Any, tokenizer: Any, API_description: str, func_inputs: list, func_outputs: list, description_text: str = "") -> str:
    """
    Processes a docstring using a language model to modify or enhance it based on the provided API description and inputs/outputs.

    Parameters
    ----------
    llm : Any
        The language model to use.
    tokenizer : Any
        The tokenizer for the language model.
    API_description : str
        The description of the API to include in the prompt.
    func_inputs : list
        A list of function input descriptions.
    func_outputs : list
        A list of function output descriptions.
    description_text : str, optional
        Additional description text to include in the prompt. Default is empty.

    Returns
    -------
    str
        The modified or generated docstring.
    """
    # LLM for modifying docstring
    prompt = build_prompt_for_composite_docstring(API_description, func_inputs, func_outputs, description_text)
    response, history = LLM_response(llm,tokenizer,prompt,history=[],kwargs={})
    print(f'==>GPT docstring response: {response}')
    if 'def' in response.split('\n')[0]:
        return '\n'.join(response.split('\n')[1:])
    else:
        return response

def process_name_with_LLM(llm: Any, tokenizer: Any, sub_API_names: str, llm_docstring: str) -> str:
    """
    Processes API names using a language model to generate a function name based on the docstring and API details.

    Parameters
    ----------
    llm : Any
        The language model to use.
    tokenizer : Any
        The tokenizer for the language model.
    sub_API_names : str
        A comma-separated string of sub-API names to include in the prompt.
    llm_docstring : str
        The docstring to use as context in the prompt.

    Returns
    -------
    str
        The generated function name.
    """
    prompt = build_prompt_for_composite_name(sub_API_names, llm_docstring)
    response, history = LLM_response(llm,tokenizer,prompt,history=[],kwargs={})
    print(f'==>GPT name response: {response}')
    MAX_trial = 5
    count=0
    while count<MAX_trial:
        try:
            ans = ast.literal_eval(response)['func_name']
            return ans
        except:
            try:
                ans = ast.literal_eval(response)
                return list(ans.keys())[0]
            except:
                response, history = LLM_response(llm,tokenizer,prompt,history=[],kwargs={})
                print(f'==>retry GPT {count}: {response}')
        count+=1
    return "function"

def main_get_API_composite(lib_analysis_path: str, output_folder_json: str, lib_alias: str) -> list:
    """
    Main function to process tutorial files and extract unique code blocks for API composition.

    Parameters
    ----------
    lib_analysis_path : str
        The path to the library analysis directory.
    output_folder_json : str
        The output folder for storing JSON data.
    lib_alias : str
        The library alias used during processing.

    Returns
    -------
    list
        A list of unique code blocks extracted from the tutorials.
    """
    # get text&code from tutorials
    context = main_convert_tutorial_to_py(lib_analysis_path, strategy_type=args.file_type, file_types=[args.file_type])
    html_dict = context.code_json
    # load json files
    pre_code_list = ["import warnings","warnings.filterwarnings('ignore')","import matplotlib","matplotlib.use('Agg')","import matplotlib.pyplot as plt","plt.interactive(False)"]# ,"import os" ,"os.environ['OMP_NUM_THREADS'] = '1'"]
    # parallel
    #with Pool() as pool:
    #    tasks = [(tut, html_dict, output_folder_json, pre_code_list) for tut in html_dict]
    #    pool.starmap(individual_tut, tasks)
    # combine the code blocks from different tutorial
    code_blocks_total = []
    for tut in html_dict:
        code_blocks_total.extend(individual_tut(tut, html_dict, output_folder_json, pre_code_list, lib_alias))
    # remove duplicate code block regarding their APIs set.
    seen_api_sets = set()
    unique_code_blocks = []
    for block in code_blocks_total:
        api_set = frozenset(block['APIs'])
        if api_set not in seen_api_sets:
            unique_code_blocks.append(block)
            seen_api_sets.add(api_set)
    print('get unique_code_blocks from tutorials:', unique_code_blocks)
    return unique_code_blocks
    
def main_get_LLM_docstring(unique_code_blocks: list, LIB: str) -> None:
    """
    Processes unique code blocks to generate enhanced docstrings using a language model.

    Parameters
    ----------
    unique_code_blocks : list
        A list of unique code blocks to process.
    LIB : str
        The library for which the docstrings are being generated.

    """
    # LLM model
    llm, tokenizer = LLM_model()
    # load API_init.json
    API_init = load_json(os.path.join('data','standard_process',LIB,"API_init.json"))
    API_composite = API_init.copy()
    idxxxxx = 1
    for code_blocks in unique_code_blocks:
        API_description = []
        func_inputs = []
        func_outputs = []
        sub_API_names = []
        count = 1
        if all(api in API_init for api in code_blocks['APIs']):
            for API_single in code_blocks['APIs']:
                one_desc = API_init[API_single]["Docstring"].split('\n')[0]
                API_description.append(f"API{count}: {one_desc}")
                sub_API_names.append(API_single)
                # only need required parameters currently
                func_inputs.extend([param+'("'+str(API_init[API_single]['Parameters'][param]['type'])+'")'+str(API_init[API_single]['Parameters'][param]['description']) for param in API_init[API_single]['Parameters'] if not API_init[API_single]['Parameters'][param]['optional']])
                func_outputs.extend(str(API_init[API_single]['Returns']['type'])+str(API_init[API_single]['Returns']['description']))
                count+=1
        else:
            continue
        # drop duplicate
        func_inputs = list(set(func_inputs))
        # prompt
        print('llm: ', llm)
        llm_docstring = process_docstring_with_LLM(llm, tokenizer, '\n'.join(API_description), json.dumps(func_inputs),json.dumps(func_outputs), description_text=code_blocks['text'])
        new_name = process_name_with_LLM(llm, tokenizer, ','.join(sub_API_names),llm_docstring)
        if new_name=='function':
            new_name = f'function_{idxxxxx}'
            idxxxxx+=1
        parameters = {}
        for API_single in sub_API_names:
            parameters.update(API_init[API_single]['Parameters'])
        API_composite[new_name] = {"Parameters":parameters,
                                   "Returns":{"type":API_init[sub_API_names[0]]['Returns']['type'],
                                              "description": API_init[sub_API_names[0]]['Returns']['description'] # TODO: io parameters
                                              },
                                    "Docstring":llm_docstring,
                                    "description": llm_docstring.split('. ')[0],
                                    "example": "",
                                    "api_type": "function",
                                    "api_calling": [
                                        "fake_placeholder()"
                                    ],
                                    "relevant APIs": code_blocks['APIs'],
                                    "type": "compositeAPI"
                                   }
    # generate_api_calling
    API_composite = generate_api_callings(API_composite, basic_types=['str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set', 'any', 'List', 'Dict'])
    # save API_composite.json
    #with open(os.path.join(LIB_ANALYSIS_PATH, 'API_composite.json'), 'w') as f: # test path
    save_json(os.path.join("data","standard_process",LIB, 'API_composite.json', API_composite))

def generate_api_callings(results: dict, basic_types: list = ['str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set', 'any', 'List', 'Dict']) -> dict:
    """
    Generates API calling strings for each API in the results based on the parameters.

    Parameters
    ----------
    results : dict
        The API results to process.
    basic_types : list, optional
        A list of basic type strings used for filtering. Default includes common data types.

    Returns
    -------
    dict
        The results with updated API calling strings.
    """
    updated_results = {}
    for api_name, api_info in results.items():
        if api_info["api_type"]: # in ['function', 'method', 'class', 'functools.partial']
            # Update the optional_value key for each parameter
            for param_name, param_details in api_info["Parameters"].items():
                param_type = param_details.get('type')
                if param_type is not None:  # Add this line to check for None
                    param_details['optional_value'] = not any(basic_type in param_type for basic_type in basic_types)
                else:
                    param_details['optional_value'] = True  # or False, depending on your default logic
            api_calling_keys = generate_api_calling_simple(api_name, api_info["Parameters"])
            api_info["api_calling"] = api_calling_keys
        else:
            api_info["api_calling"] = []
        updated_results[api_name] = api_info
    return updated_results

def generate_api_calling_simple(api_name: str, parameters: dict) -> list:
    """
    Generates a simple API calling structure based on the API name and its parameters.

    Parameters
    ----------
    api_name : str
        The name of the API to generate calling for.
    parameters : dict
        The parameters of the API.

    Returns
    -------
    list
        A list of strings representing simple API callings.
    """
    api_calling = []
    if parameters:
        param_strs = []
        for idx, (param_name, param_details) in enumerate(parameters.items(), 1):
            if param_details.get('optional_value', False):
                param_strs.append(f"{param_name}=$")
            else:
                param_strs.append(f"{param_name}=@")
        api_call = f"{api_name}({', '.join(param_strs)})"
        api_calling.append(api_call)
    else:
        api_call = f"{api_name}()"
        api_calling.append(api_call)
    return api_calling


import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--LIB', type=str, required=True, help='PyPI tool')
    parser.add_argument('--file_type', type=str, default='ipynb', help='tutorial files type')
    args = parser.parse_args()
    info_json = get_all_variable_from_cheatsheet(args.LIB)
    LIB_ALIAS, LIB_ANALYSIS_PATH = [info_json['LIB_ALIAS'], info_json['LIB_ANALYSIS_PATH']]
    
    output_folder = os.path.join(ANALYSIS_PATH,args.LIB,"Git_Tut_py")
    output_folder_json = os.path.join(ANALYSIS_PATH,args.LIB,"Git_Tut_py2json")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder_json):
        os.makedirs(output_folder_json)
    all_files = os.listdir(output_folder)
    all_files = [i for i in all_files if not i.endswith('_tmp.py') and not i.endswith('_remain.py')]
    print(all_files)
    
    unique_code_blocks = main_get_API_composite(os.path.join(ANALYSIS_PATH,args.LIB), output_folder_json, LIB_ALIAS)
    main_get_LLM_docstring(unique_code_blocks, args.LIB)