import json
import argparse
import os

# Function to extract tasks from a JSON file
def extract_tasks(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    result_data = []
    for item in data['history']:
        code_status = {}
        for message in item['messages']:
            record_status = False
            success_flag = False
            fail_flag = False
            task_title_list = []
            for task in message.get('tools', []):
                # Check if the task is related to code execution
                if task['method_name'] == "on_agent_action" and "block_id" in task and "code" in task["block_id"]:
                    code = task['task']
                    record_status = True
                task_title_list.append(task.get('task_title', ''))
            if record_status:
                # Determine if the code execution was successful or failed
                if "Executed results [Success]" in task_title_list:
                    success_flag = True
                elif "Executed results [Fail]" in task_title_list:
                    fail_flag = True
                # Ensure that only one of success or fail is True, not both
                assert not(success_flag and fail_flag)
                # Ensure that either success or fail flag is True
                assert (success_flag or fail_flag)
                # Ensure that code status is not "No result"
                assert code_status[code] != 'No result'
                code_status[code] = "Success" if success_flag else "Fail" if fail_flag else "No result"
        result_data.append(code_status)
    for result in result_data:
        if "Success" in result:
            assert not "Fail" in result, "Both success and fail task results found in the same message!"
    return result_data

# Function to generate Python code snippets from successful task results
def generate_python_code(result_data, file_path):
    code_snippets = []
    for result in result_data:
        for code, status in result.items():
            if status == "Success":
                code_snippets.append(code)
    python_code = "\n".join(code_snippets)
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract filename without extension
    output_file = f"{file_name}.py"
    with open(os.path.join('report', output_file), "w") as file:
        file.write(python_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tasks from JSON file")
    parser.add_argument("file_path", help="Path to the JSON file")
    args = parser.parse_args()
    # Extract tasks from the JSON file and print the results
    tasks = extract_tasks(args.file_path)
    print(json.dumps(tasks, indent=2))
    # Generate Python code from successful task results and save it to a .py file
    generate_python_code(tasks, args.file_path)
