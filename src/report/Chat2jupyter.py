import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
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


# Define a function to extract the relevant elements from the JSON content
def extract_elements(json_content):
    extracted_elements = []
    for chat in json_content['history']:
        for message in chat['messages']:
            if message['role'] == 'assistant':
                for tool in message['tools']:
                    if 'block_id' in tool and tool['block_id'].startswith('code'):
                        extracted_elements.append({
                            'type': 'code',
                            'content': tool['task']
                        })
                    elif tool['block_id'].startswith('log'):
                        if (tool['task_title'].startswith('Predicted API') or tool['task_title'].startswith('Executed results')):
                            extracted_elements.append({
                                'type': 'markdown',
                                'content': f"**{tool['task_title']}**\n\n{tool['task']}"
                            })
                        if 'imageData' in tool and tool['imageData'].strip():
                            # Adding image data
                            extracted_elements.append({
                                'type': 'image',
                                'content': f"![Image](data:image/png;base64,{tool['imageData']})"
                            })
                        if 'tableData' in tool and tool['tableData'].strip('\"'):
                            # Convert table data to markdown table format
                            table_md = string_to_markdown_table(tool['tableData'].strip('\"'))
                            extracted_elements.append({
                                'type': 'table',
                                'content': table_md
                            })
    return extracted_elements

def string_to_markdown_table(table_string):
    # Split the string into rows by newline
    rows = table_string.strip().split("\n")
    # Convert each row into columns by splitting on commas
    markdown_rows = ["| " + " | ".join(row.split(",")) + " |" for row in rows]
    # Generate a separator row for markdown (assuming all rows have the same number of columns)
    num_columns = len(markdown_rows[0].split("|")) - 2  # Because of leading/trailing "|"
    separator_row = "|-" + "-|-".join(["" for _ in range(num_columns)]) + "-|"
    # Insert the separator after the first row (which is assumed to be the header)
    markdown_rows.insert(1, separator_row)
    return "\n".join(markdown_rows)

# Helper function to generate a Markdown table from data
def generate_markdown_table(data):
    if not data or not isinstance(data, list) or not all(isinstance(row, list) for row in data):
        return ""
    headers = data[0]
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "|-" + "-|-".join(["" for _ in headers]) + "-|\n"
    for row in data[1:]:
        md_table += "| " + " | ".join(row) + " |\n"
    return md_table

# Function to create a Jupyter notebook with the extracted elements
def create_notebook(extracted_elements):
    nb = new_notebook()

    # Iterate through the extracted elements and add them to the notebook in the correct order
    for element in extracted_elements:
        if element['type'] == 'code':
            nb.cells.append(new_code_cell(element['content']))
        elif element['type'] == 'image':
            nb.cells.append(new_markdown_cell(element['content']))
        elif element['type'] == 'table':
            nb.cells.append(new_markdown_cell(element['content']))
        elif element['type'] == 'markdown':
            nb.cells.append(new_markdown_cell(element['content']))

    return nb

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == "__main__":
    import argparse
    # Parsing arguments for the JSON file path
    parser = argparse.ArgumentParser(description="Extract tasks from JSON file")
    parser.add_argument("file_path", help="Path to the JSON file")
    args = parser.parse_args()
    json_content = load_json(args.file_path)
    extracted_elements = extract_elements(json_content)
    notebook = create_notebook(extracted_elements)
    notebook_file_path = args.file_path.replace('.json','.ipynb')
    with open(notebook_file_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
