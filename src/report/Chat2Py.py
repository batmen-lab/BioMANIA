"""
Author: Zhengyuan Dong
Email: zydong122@gmail.com
Description: This script contains functions to extract tasks from JSON chat files and generate Python code.
"""
from ..gpt.utils import load_json

def extract_tasks(file_path):
    data = load_json(file_path)
    result_data = []
    for item in data['history']:
        code_status = {}
        for message in item['messages']:
            record_status = False
            success_flag = False
            fail_flag = False
            task_title_list = []
            for task in message.get('tools', []):
                if task['method_name'] == "on_agent_action" and "block_id" in task and "code" in task["block_id"]:
                    code = task['task']
                    record_status = True
                task_title_list.append(task.get('task_title', ''))
            if record_status:
                if "Executed results [Success]" in task_title_list:
                    success_flag = True
                elif "Executed results [Fail]" in task_title_list:
                    fail_flag = True
                # one True, another False
                assert not(success_flag and fail_flag)
                assert (success_flag | fail_flag)
                code_status[code] = "Success" if success_flag else "Fail" if fail_flag else "No result"
                assert code_status[code]!='No result'
        result_data.append(code_status)
    for result in result_data:
        if "Success" in result:
            assert not "Fail" in result, "Both success and fail task results found in the same message!"
    return result_data

import os
def generate_python_code(result_data, file_path):
    code_snippets = []
    for result in result_data:
        for code, status in result.items():
            if status == "Success":
                code_snippets.append(code)
    python_code = "\n".join(code_snippets)
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract filename without extension
    output_file = f"{file_name}.py"
    with open(os.path.join('report',output_file), "w") as file:
        file.write(python_code)

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