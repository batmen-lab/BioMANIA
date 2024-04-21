"""
Author: Zhengyuan Dong
Date Created: December 10, 2023
Last Modified: December 10, 2023
Description: prepare training data for tutorials text API prediction
"""
from ..configs.model_config import get_all_variable_from_cheatsheet, READTHEDOC_PATH, get_all_variable_from_cheatsheet #tut, html_dict, code
from ..dataloader.utils.tutorial_loader_strategy import main_convert_tutorial_to_py
import os, ast, io, tokenize
from collections import OrderedDict
from ..models.model import LLM_model, LLM_response
from tqdm import tqdm
from ..gpt.utils import save_json
#from dataloader.preprocess_retriever_data import preprocess_fake_test_data

# Step1: get tutorial pieces
def get_tutorial_pieces(LIB, TUTORIAL_HTML, LIB_ANALYSIS_PATH, use_source="readthedoc"):
    if use_source=='github':
        context = main_convert_tutorial_to_py(LIB_ANALYSIS_PATH, subpath='Git_Tut', strategy_type='ipynb', file_types=['ipynb'])
        html_dict = context.code_json
        save_json('context1.json', html_dict)
    elif use_source=='readthedoc':
        readthedoc_realpath = os.path.join(READTHEDOC_PATH, TUTORIAL_HTML)
        context = main_convert_tutorial_to_py(readthedoc_realpath, subpath='', strategy_type='html', file_types=['html'])
        html_dict = context.code_json
        save_json('context2.json', html_dict)
    # Step2: break into pieces
    window_size = 1  # Set your desired sliding window size
    combined_tutorials = {}
    for tut_key in html_dict.keys():
        tutorial = html_dict[tut_key]
        combined_pieces = []
        for i in range(0, len(tutorial), window_size):
            window = tutorial[i:i + window_size]
            combined_code = []
            for piece in window:
                combined_code.extend(piece['code'])
            combined_code = '\n'.join(combined_code)
            combined_text = '\n\n'.join(piece['text'] for piece in window if piece['text'])
            combined_pieces.append({'code': combined_code, 'text': combined_text})
        combined_tutorials[tut_key] = combined_pieces
    # Save combined_tutorials as JSON
    output_json_path = f'./data/autocoop/{LIB}/combined_tutorials.json'
    output_dir = os.path.dirname(output_json_path)
    os.makedirs(output_dir, exist_ok=True)
    save_json(output_json_path, combined_tutorials)
    print(f"Combined tutorials saved to {output_json_path}")
    return combined_tutorials

# Step3: llm summarize
def build_prompt_for_summarize(code, text):
    # TODO: modify it later
    return f"""Please summarize the original long text description in to 1-2 sentence. You need to ensure that the summarized description corresponds to the API funcitonality. Do not include any other text or code except for the targeted description.
        code: `{code}`
        text: `{text}`
    """

def get_full_path(node):
    if isinstance(node, ast.Attribute):
        return get_full_path(node.value) + '.' + node.attr
    elif isinstance(node, ast.Name):
        return node.id
    return ''

def find_api_calls(node, imports, lib_alias=None):
    api_calls = []
    def recursive_search(node, current_path=None):
        if isinstance(node, (ast.Call, ast.Attribute)):
            api_call_path = get_full_path(node.func if isinstance(node, ast.Call) else node)
            #print(f"Found path: {api_call_path}")
            path_elements = api_call_path.split('.')
            if path_elements[0] in imports:
                if (imports[path_elements[0]].startswith(lib_alias)) or (not lib_alias):
                    api_call = '.'.join([imports[path_elements[0]]] + path_elements[1:])
                    if not current_path or (current_path and len(api_call.split('.')) > len(current_path.split('.'))):
                        current_path = api_call
                        if current_path not in api_calls:
                            api_calls.append(current_path)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, (ast.Attribute, ast.Name)):
                    api_call_path = get_full_path(target)
                    path_elements = api_call_path.split('.')
                    if path_elements[0] in imports:
                        if (imports[path_elements[0]].startswith(lib_alias)) or (not lib_alias):
                            api_call = '.'.join([imports[path_elements[0]]] + path_elements[1:])
                            api_calls.append(api_call)
        if current_path and current_path not in api_calls:
            api_calls.append(current_path)
        for child in ast.iter_child_nodes(node):
            recursive_search(child, current_path)
    recursive_search(node)
    return api_calls

def remove_indentation(lines):
    return '\n'.join(line.lstrip() for line in lines.split('\n'))

def try_parse_extract(buffer, imports, lib_alias):
    try:
        root = ast.parse(buffer)
        return find_api_calls(root, imports, lib_alias)
    except SyntaxError as e:
        try:
            root = ast.parse(remove_indentation(buffer))
            return find_api_calls(root, imports, lib_alias)
        except SyntaxError as e:
            print(f"Failed to parse even after removing indentation: {buffer.strip()} | Error: {e}")
            return []

def extract_api_calls(code_block, imports, lib_alias):
    """
    Extract all library alias API calls from the provided code block, attempting to parse it and
    recover from syntax errors gracefully.
    """
    api_calls = []
    lines = code_block.split('\n')
    buffer = ''
    open_parens = open_braces = open_brackets = 0
    # Process each line to handle partial parsing
    for line in lines:
        buffer += line.strip().split('#')[0] # there exist comment in multiple lines!!!
        # Update counters for open parenthesis, braces, and brackets
        open_parens += line.count('(') - line.count(')')
        open_braces += line.count('{') - line.count('}')
        open_brackets += line.count('[') - line.count(']')
        # Check if all types of brackets are closed
        if open_parens == 0 and open_braces == 0 and open_brackets == 0:
            if 'sc.pl.umap(' in buffer:
                print('-------------------!!!!', buffer)
            api_calls += try_parse_extract(buffer, imports, lib_alias)
            buffer = ''  # Reset buffer if all brackets are closed
    # Attempt to parse any remaining code in the buffer
    if buffer.strip():
        api_calls += try_parse_extract(buffer, imports, lib_alias)
    return api_calls

def extract_imports(code_block):
    """
    Extract all import statements from a Python code block, handling multiline import statements.
    """
    imports = {}
    lines = code_block.split('\n')
    multiline_code = ''
    in_multiline_import = False
    # Process each line to handle multiline imports
    for line in lines:
        # Clean non-ASCII characters and strip extra whitespace
        cleaned_line = ''.join(c if ord(c) < 128 else ' ' for c in line).strip()
        # Check if we are currently gathering a multiline import statement
        if in_multiline_import or cleaned_line.startswith('from ') or cleaned_line.startswith('import '):
            multiline_code += cleaned_line + ' '
            # Check if the line ends with a backslash indicating a continuation or contains an open parenthesis
            if '\\' in cleaned_line or '(' in cleaned_line:
                in_multiline_import = True
                continue
            # Check if we are still inside parentheses
            if '(' in multiline_code and ')' not in multiline_code:
                in_multiline_import = True
                continue
            in_multiline_import = False  # End of a multiline statement
        else:
            continue  # Skip non-import lines
        # Try to parse the gathered import statement
        try:
            root = ast.parse(multiline_code)
            for node in ast.walk(root):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports[name.asname if name.asname else name.name] = name.name
                elif isinstance(node, ast.ImportFrom):
                    module = node.module if node.module else ''
                    for name in node.names:
                        full_name = module + '.' + name.name if module else name.name
                        imports[name.asname if name.asname else name.name] = full_name
        except SyntaxError as e:
            print('error for extracting imports:', e)
            # If a syntax error occurs, reset and start a new multiline statement
            multiline_code = ''
            in_multiline_import = False
        # Reset multiline_code after successful parsing
        multiline_code = ''
    return imports

def get_sub_API(code, imports, LIB_ALIAS):
    current_apis = extract_api_calls(code, imports, LIB_ALIAS)
    # remove duplicate API inside code block
    current_apis_set = list(OrderedDict.fromkeys(current_apis))
    return current_apis, list(current_apis_set)

def extract_comments_from_code(code):
    """
    Extract comments from a given piece of code.
    :param code: A string containing the code.
    :return: A string containing all extracted comments.
    """
    comments = []
    try:
        tokens = tokenize.tokenize(io.BytesIO(code.encode('utf-8')).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                comments.append(tok.string)
    except tokenize.TokenError as e:
        print('error parsing for:', e)
        pass  # Handle incomplete code snippets
    return comments

def get_relevant_API(combined_tutorials, LIB_ALIAS, ask_GPT=False):
    if ask_GPT:
        llm, tokenizer = LLM_model()
    summarized_responses = {}
    for tut_key in tqdm(combined_tutorials):
        combined_pieces = combined_tutorials[tut_key]
        tutorial_responses = []
        imports = {}
        for idx, piece in enumerate(combined_pieces):
            code = piece['code']
            text = piece['text']
            # extract comments
            #comments = extract_comments_from_code(code)
            #text+='Here are some comments for code.'+'\n'.join(comments)
            imports.update(extract_imports(code))
            ori_relevant_API, relevant_API = get_sub_API(code, imports, LIB_ALIAS)
            if ask_GPT:
                # ask gpt to polish and summarize the tutorial text
                prompt = build_prompt_for_summarize(code, text)
                response, history = LLM_response(llm, tokenizer, prompt)
            else:
                response = text
            if not relevant_API:
                relevant_API = []
                ori_relevant_API = []
            tutorial_responses.append({'text':response, "code":code, "ori_relevant_API":ori_relevant_API, "relevant_API":relevant_API})
        summarized_responses[tut_key] = tutorial_responses
    output_json_path = f'./data/autocoop/{args.LIB}/summarized_responses.json'
    output_dir = os.path.dirname(output_json_path)
    os.makedirs(output_dir, exist_ok=True)
    save_json(output_json_path, summarized_responses)
    print(f"summarized_responses saved to {output_json_path}")
    return summarized_responses

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--LIB', type=str, required=True, help='PyPI tool')
    args = parser.parse_args()
    info_json = get_all_variable_from_cheatsheet(args.LIB)
    TUTORIAL_HTML, LIB_ANALYSIS_PATH, LIB_ALIAS = [info_json[key] for key in ['TUTORIAL_HTML', 'LIB_ANALYSIS_PATH', 'LIB_ALIAS']]
    combined_tutorials = get_tutorial_pieces(args.LIB, TUTORIAL_HTML, LIB_ANALYSIS_PATH)
    summarized_responses = get_relevant_API(combined_tutorials, LIB_ALIAS)
