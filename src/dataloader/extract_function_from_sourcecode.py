"""
extract info from source code
"""
import os
import ast
import json
import glob
from docstring_parser import parse
import re
import sys
import importlib
import inspect
import astunparse
from configs.model_config import *

def process_function(node,tree,filename,pair_decorator={}):
    """
    Extract information of node
    """
    docstring = ast.get_docstring(node)
    doc = parse(docstring)
    if docstring:
        if ('{' in docstring) and ('}' in docstring):
            docstring = replace_docstring_placeholder(docstring, node, tree, filename, pair_decorator)
        params, returns, examples = get_returnparam_docstring(docstring)
    else:
        returns = {}
    func_info = {
        'func_name': node.name,
        'docstring': docstring,
        'param': [],
        'filepath':filename,
        'examples':[],
        'returns':{
            'returnObj':[],
            'returnParam':{},
        }
    }
    defaults = [None for _ in range(len(node.args.args) - len(node.args.defaults))] + list(node.args.defaults)
    for arg,default in zip(node.args.args,defaults):
        arg_info = {
            'param_name': arg.arg,
            'type': get_type_astObject(arg.annotation),
            'default': get_type_astObject(default), # ast.literal_eval(default) if default else 
            'description':get_description(params, arg.arg)
        }
        func_info['param'].append(arg_info)
    for arg,default_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
        arg_info = {
            'param_name': arg.arg,
            'type': get_type_astObject(arg.annotation),
            'default': ast.unparse(default_node).strip() if default_node else 'None',
            'description': get_description(params, arg.arg)
        }
        func_info['param'].append(arg_info)
    if node.args.vararg is not None:
        arg_info = {
            'param_name': '*' + node.args.vararg.arg,
            'type': get_type_astObject(node.args.vararg.annotation),
            'default': None,
            'description': get_description(params, node.args.vararg.arg)}
        func_info['param'].append(arg_info)
    func_info['dples'] = examples
    func_info['returns']['returnObj']=get_type_astObject(node.returns)
    func_info['returns']['returnParam'] = returns
    func_info['relativeimport'] = LIB_ALIAS+filename.split(LIB_ALIAS)[-1].replace(".py", "").replace("/", ".")
    return func_info

def extract_single_block(docstring, keyword_base):
    keyword_variations = [keyword_base, keyword_base.capitalize(),
        keyword_base + "s", keyword_base.capitalize() + "s"]
    keyword_variations = ["\n"+i+"\n" for i in keyword_variations]
    end_keywords_map = {
        "Example": ["Parameter", "Return"],
        "Parameter": ["Example", "Return"],
        "Return": ["Example", "Parameter"]
    }
    start_indices = [docstring.find(kw) for kw in keyword_variations if docstring.find(kw) != -1]
    if not start_indices:  # If no start keyword is found, return an empty string
        return ""
    start_index = min(start_indices)
    end_keyword_variations = sum([[
        "\n" + end_kw+"\n", "\n" + end_kw.capitalize()+"\n",
        "\n" + end_kw + "s"+"\n", "\n" + end_kw.capitalize() + "s"+"\n"
    ] for end_kw in end_keywords_map[keyword_base]], [])
    end_indices = [docstring.find(kw, start_index) for kw in end_keyword_variations if docstring.find(kw, start_index) != -1]
    end_index = min(end_indices) if end_indices else len(docstring)
    return docstring[start_index:end_index].strip()

def clean_return_block(return_block):
    return_keywords = ["returns", "return", "Returns", "Return"]
    lines = [line.strip() for line in return_block.split("\n") if line.strip() and not all(c == '-' for c in line)]
    if lines and (lines[0] in return_keywords or lines[0].rstrip(':') in return_keywords):
        lines = lines[1:]
    return "\n".join(lines)

def get_returnparam_docstring(docstring):
    """
    Parses a function docstring to extract Name, type and description for Parameters, Returns, Examples
    Following specific patterns for parameter, return and example sections. 
    """
    # get blocks which starts with keywords
    params_str = extract_single_block(docstring, "Parameter")
    returns_str = extract_single_block(docstring, "Return")
    examples_str = extract_single_block(docstring, "Example")

    # get paired value for [parameters, dtype, description]
    # try four methods, get longest answer
    #params1 = get_param_re_from_block_t1(params_str)
    #params2 = get_param_re_from_block_t2(params_str)
    #params3 = get_param_re_from_block_t3(params_str)
    params = get_param_from_block_t4(docstring)
    #params = find_longest_dict_list(params1, params2, params3, params4)
    #params = find_longest_dict_list(params4)

    """returns1 = get_param_re_from_block_t1(returns_str)
    returns2 = get_param_re_from_block_t2(returns_str)
    returns3 = get_param_re_from_block_t3(returns_str)
    returns = find_longest_dict_list(returns1, returns2, returns3)"""
    
    #example = extract_executable_content_from_block(examples_str)
    return params, clean_return_block(returns_str), examples_str

def extract_executable_content_from_block(block):
    lines = block.split("\n")
    executable_lines = [line.strip() for line in lines if not line.strip().startswith(":")]
    return executable_lines

def find_longest_dict_list(*dicts):
    # get longest dict from all input dicts
    max_length = 0
    longest_dict = {}
    for d in dicts:
        if d is not None:
            length = len(d)
            if length > max_length:
                max_length = length
                longest_dict = d
    return longest_dict

def get_param_from_block_t4(params_str):
    """
    Get parameter description from docstring using parse() function.
    """
    params = {}
    doc = parse(params_str)
    for desc in doc.params:
        if desc is not None:
            params[desc.arg_name] = {
                "type": desc.type_name.strip() if desc.type_name else None,
                "description": desc.description.strip() if desc.description else None,
                "default":desc.default.strip() if desc.default else None,
            }
    return params

def get_param_re_from_block_t1(params_str):
    args_pattern = r"^\s*(\w+)\s*:\s*(\w+)\s*\n\s{8}(.*)$"
    params = {}
    for match in re.finditer(args_pattern, params_str, re.MULTILINE):
        name, dtype, description = match.groups()
        params[name] = {
            "type": dtype.strip(),
            "description": description.strip(),
            "default":None
        }
    return params

def get_param_re_from_block_t2(params_str):
    args_pattern = r"^\s*\**(\w+)\**\s*:\s*([\s\S]*?)(\n[^\n]*)"
    params = {}
    matches = re.findall(args_pattern, params_str, re.MULTILINE)
    for match in matches:
        name = match[0]
        content = match[1]
        description = match[2]
        params.update({name:{"type":content,"description":description,"default":None}})
    return params

def get_param_re_from_block_t3(param_str):
    # work for not containing : between param and default value & type
    pattern = r"(?<=\n\s{4}).+?(?=\n\s{4}\w+|$)"
    matches = re.findall(pattern, param_str, re.DOTALL)
    params = {}
    for match in matches:
        lines = match.strip().split("\n")
        name = lines[0]
        description = " ".join(lines[1:])
        params[name] = {
            "type": "",
            "description": description.strip(),
            "default":None
        }
    return params

def handle_tuple(node):
    """
    Get type of ast.Tuple 
    """
    a = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant):
            a.append(type(elt.value).__name__)
        elif isinstance(elt, ast.Name):
            a.append(elt.id)
        elif isinstance(elt, ast.Subscript):
            a.append(f'{elt.value.id}[{elt.slice.value.id}]')
        elif isinstance(elt, ast.Tuple):
            a.append(handle_tuple(elt))
        elif isinstance(elt, ast.List):
            a.append(handle_tuple(elt))
    return a

def get_type_astObject(node_return):
    """
    Extracts type information from function return annotation
    """
    if node_return:
        if isinstance(node_return, ast.Subscript):
            value_str = astunparse.unparse(node_return.value).strip()
            slice_str = astunparse.unparse(node_return.slice).strip()
            slice_str = slice_str.replace('(','').replace(')','')
            type_name = f"{value_str}[{slice_str}]"
        elif isinstance(node_return, ast.Name):
            type_name = node_return.id
        elif isinstance(node_return, ast.Constant):
            type_name = str(node_return.value)
        elif isinstance(node_return, ast.Attribute):
            obj_str = astunparse.unparse(node_return.value).strip()
            attr_str = node_return.attr
            type_name = f"{obj_str}.{attr_str}"
        elif isinstance(node_return, ast.Call):
            func_str = astunparse.unparse(node_return.func).strip()
            args_str = ','.join([astunparse.unparse(arg).strip() for arg in node_return.args])
            keywords_str = ','.join([f"{kw.arg}={astunparse.unparse(kw.value).strip()}" for kw in node_return.keywords])
            type_name = f"{func_str}[{args_str}, {keywords_str}]"
        elif isinstance(node_return, ast.Tuple):
            type_name = handle_tuple(node_return)
        elif isinstance(node_return, ast.UnaryOp):
            type_name = get_type_astObject(node_return.operand)
        else:
            try:
                type_name = ast.literal_eval(node_return)
            except:
                type_name = None
    else:
        type_name=None
    return type_name

def extract_unknown_text(input_string):
    # get decorator part from a large string
    pattern = re.compile(r"@\w+\((.*?)\)\s*\ndef\s+(\w+)\(", re.DOTALL)
    matches = pattern.findall(input_string)
    # create a dictionary of matches
    result = {funcname: params for params, funcname in matches}
    return result

def get_decorator_mapping(source_code, func=None):
    """
    get mapping of new old variables from string
    return: {fun1: {param2:param1}, func2: {param4:param3,param6:param5}}
    """
    try:
        source_lines, _ = inspect.getsourcelines(func)
        source_code = "".join(source_lines)
    except:
        pass
    result = extract_unknown_text(source_code)
    pattern = re.compile(r"(\w+)\s*=\s*([\w.]+)")
    results = {}
    for key in result:
        variable_mapping = {}
        matches = pattern.findall(result[key])
        for match in matches:
            variable_mapping[match[1]] = match[0]
        results[key] = variable_mapping
    return results

def extract_decorators(node):
    # Extracts decorator variable assignments from a function definition
    devar_dict = {}
    for decorator in node.decorator_list:
        for pair in decorator.keywords:
            devar_dict.update({pair.value.id: pair.arg})
    return devar_dict

def replace_docstring_placeholder(docstring, node, tree, filepath,devar_dict={}):
    """
    replace_placeholder in docstring
    Two steps:
    1. get string from other files import
    2. get string from defined variables within file
    """
    success_count = 0
    fail_count = 0
    # path for import 
    filedir = os.path.dirname(filepath)
    sys.path.append(filedir)
    # step 1
    global placeholders
    placeholders = [word.strip('{}') for word in docstring.split() if word.startswith('{') and word.endswith('}')]
    # update var name in placeholders to original import name
    
    for key in devar_dict:
        docstring = docstring.replace('{'+devar_dict[key]+'}','{'+key+'}')
    # make up for the remaining key in placeholders
    for key in placeholders:
        if key not in list(devar_dict.values()):
            devar_dict.update({key:key})
    placeholders = list(devar_dict.keys())
    # step 2
    # get all "import" and "from import"
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, None))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                if node.level == 0:
                    full_name = node.module + '.' + '*' if not node.names or not node.names[0].name else node.module
                else:
                    full_name = '.' * node.level + node.module + '.' + '*' if not node.names or not node.names[0].name else '.' * node.level + node.module
            else:
                full_name = '*' if not node.names or not node.names[0].name else node.names[0].name
            for alias in node.names:
                if alias.name != '*':
                    asname = alias.asname or alias.name
                    imports.append((asname, full_name))
                else:
                    imports.append((None, full_name))
    # Gets imports that contain relevant placeholders
    relevant_imports = []
    for name, full_name in imports:
        if name in placeholders or full_name in placeholders:
            relevant_imports.append((name, full_name))
    # Creates import statements from relevant imports
    import_statements = {}
    for name, full_name in relevant_imports:
        if name:
            import_statements[name]=[f"from {full_name} import {name}",full_name]
        else:
            import_statements[name]=[f"import {full_name}",full_name]
    # execute import
    variables = {}
    for name in import_statements:
        import_statement = import_statements[name][0]
        full_name = import_statements[name][1]
        import_statement=  modify_import(LIB_ALIAS, import_statement, filepath)

        full_name = import_statement.split('from ')[-1].split(' import')[0]
        try:
            #exec(import_statement)
            module = importlib.import_module(full_name)
            #attr_value = getattr(module,name)
            get_value = get_variable_value(name,module)
            if isinstance(get_value, str):
                variables.update({name: get_value})
            success_count+=1
            #print('success:', module, full_name, name, filepath)
        except:
            print('fail:', module, full_name, name, filepath)
            fail_count+=1
            pass
    # part 2
    # get defined string inside document
    find_vi = FindXAssignVisitor()
    find_vi.visit(tree)
    tmp_var = find_vi.find_string()
    variables.update(tmp_var)

    docstring = replace_placeholders(docstring,placeholders,variables)
    #print('success: ', success_count, 'fail: ',fail_count, 'filename', filepath)
    return docstring

class FindXAssignVisitor(ast.NodeVisitor):
    def __init__(self):
        self.return_list = {}
    def visit_Assign(self, node):
        global placeholders
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in placeholders:
                    if isinstance(node.value, ast.Str):
                        self.return_list.update({target.id:node.value.s})
                    
    def find_string(self, ):
        return self.return_list

def modify_import(libname, import_str, file_path):
    """
    replace import as absolute path
    """
    # get absolute filepath
    ori_path = import_str.split('from ')[-1].split(' import ')[0]
    relative_path = ori_path[:]
    file_dir = file_path.rsplit('/', 1)[0]
    if relative_path.startswith('..'):
        file_dir = os.path.dirname(file_dir)
        relative_path = relative_path[2:]
    if relative_path.startswith('.'):
        relative_path = relative_path[1:]
    for path_part in relative_path.split('.'):
        file_dir = os.path.join(file_dir, path_part)
    # get import path
    scanpy_path = file_dir.split(libname)[1].strip(os.path.sep)
    scanpy_path = libname+'.' + scanpy_path.replace(os.path.sep, '.')
    if ori_path.startswith('.'):
        import_str = import_str.replace(ori_path, scanpy_path)
    return import_str

def get_variable_value(variable_name, full_name=None):
    """
    get value for variable_name
    """
    """if globals()[variable_name] is not None:
        return globals()[variable_name]
    if vars()[variable_name] is not None:
        return vars()[variable_name]"""
    return getattr(full_name, variable_name)
    
def replace_placeholders(docstring, placeholders,input_variables):
    """
    Replace the placeholders in a string with the values of variables 
    in the variables
    (imported value can not appear in current namespace that belong to the placeholders list.)
    """
    variables = {var: input_variables[var] for var in placeholders if var in input_variables}
    if len(variables)>0:
        for var in variables:
            docstring = docstring.replace('{'+var+'}',variables[var])
        return docstring
    else:
        return docstring

def process_file(filename):
    """
    Process a single Python file and return paired func:information.
    """
    with open(filename, 'r') as file:
        content = file.read()
    tree = ast.parse(content)
    # get all decorators as {funcname:{param2:param1}}
    pair_decorators = get_decorator_mapping(content)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            try:
                pair_decorator = pair_decorators[node.name]
            except:
                pair_decorator = {}
            functions.append(process_function(node,tree,filename,pair_decorator))
    return functions

def to_tree_json(data):
    """
    Organizes data into nested dictionaries based on relative import paths
    """
    categorized_data = {}
    for item in data:
        relative_import = item.get("relativeimport")
        if relative_import:
            # Split the relative import string into a list of keywords
            import_parts = relative_import.split(".")

            # Create nested dictionaries based on the import hierarchy
            current_dict = categorized_data
            for part in import_parts:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            # Add the item to the innermost dictionary
            current_dict[item["func_name"]] = item
    return categorized_data

def processdir_to_function(dir_path):
    """
    Process all Python files in a directory and its subdirectories, and return a list of defined functions.
    """
    all_functions = []
    py_files = glob.glob(dir_path+'/**/*.py', recursive=True)
    for filepath in py_files:
        functions = process_file(filepath)
        all_functions.extend(functions)
    return all_functions

def main():
    command = 'import '+LIB_ALIAS
    exec(command)
    dir_path = eval(f'os.path.dirname({LIB_ALIAS}.__file__)')
    if not os.path.exists(LIB_ANALYSIS_PATH):
        os.makedirs(LIB_ANALYSIS_PATH)
    output_path = os.path.join(LIB_ANALYSIS_PATH,'API_func.json')
    #tree_output_path = os.path.join(resource_dir,'demo_analysis',LIB,'tree_API.json')
    all_functions = processdir_to_function(dir_path)
    #tree_functions = to_tree_json(all_functions)
    with open(output_path, 'w') as file:
        json.dump(all_functions, file, indent=4)

if __name__=='__main__':
    main()