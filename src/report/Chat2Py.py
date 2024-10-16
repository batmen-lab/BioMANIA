"""
Author: Zhengyuan Dong
Email: zydong122@gmail.com
Open Date: Sep 01, 2023
Last Edited: Oct 16, 2024
Description: This script contains functions to extract tasks from JSON chat files and generate Python code.
"""

import json
def load_json(filename: str) -> dict:
    """
    Load JSON data from a specified file.

    Parameters
    ----------
    filename : str
        The path to the JSON file to be loaded.

    Returns
    -------
    dict
        The data loaded from the JSON file.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def extract_tasks(file_path):
    data = load_json(file_path)
    result_data = []
    code_status = {}
    last_executed_code = None
    for item in data['history']:
        for message in item['messages']:
            task_title_list = []
            for task in message.get('tools', []):
                if "method_name" in task and task['method_name'] == "on_agent_action" and "code" in task["block_id"]:
                    last_executed_code = task['task']
                    #print(f"Found Executed Code: {last_executed_code}")
                if "Executed results [Success]" in task.get('task_title', ''):
                    #print(f"Success found for code: {last_executed_code}")
                    code_status[last_executed_code] = "Success"
                    last_executed_code = None 
                elif "Executed results [Fail]" in task.get('task_title', ''):
                    #print(f"Failure found for code: {last_executed_code}")
                    code_status[last_executed_code] = "Fail"
                    last_executed_code = None 
            #if last_executed_code:
            #    print(f"No result for code yet: {last_executed_code}")
    result_data.append(code_status)
    return result_data

import os
def generate_python_code(result_data, file_path):
    code_snippets = []
    for result in result_data:
        for code, status in result.items():
            if code:
                pass
            else:
                continue
            if code.startswith("from"):
                if '\n' in code:
                    code_snippets.append(code.split('\n')[0])
                else:
                    code_snippets.append(code)
            if status == "Success" and code is not None:
                code_snippets.append(code)
    python_code = "\n".join([code for code in code_snippets if code is not None])
    python_code = deduplicate_python_code(python_code)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = f"{file_name}.py"
    output_path = os.path.join('report', output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        file.write(python_code)
    print(f"Python code saved to: {output_path}")

def deduplicate_python_code(python_code):
    seen_code = set()
    deduplicated_code = []
    code_lines = python_code.split("\n")
    for line in code_lines:
        if line not in seen_code:
            deduplicated_code.append(line)
            seen_code.add(line)
    return "\n".join(deduplicated_code)

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract tasks from JSON file")
    parser.add_argument("file_path", help="Path to the JSON file")
    args = parser.parse_args()
    tasks = extract_tasks(args.file_path)
    import json
    print(json.dumps(tasks, indent=2))
    python_code = generate_python_code(tasks, args.file_path)