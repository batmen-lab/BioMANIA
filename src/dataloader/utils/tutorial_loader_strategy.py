from abc import ABC, abstractmethod
import os, re, ast, json, requests, argparse, nbformat, subprocess
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime

from configs.model_config import ANALYSIS_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--LIB', type=str, required=True, help='PyPI tool')
parser.add_argument('--file_type', type=str, default='ipynb', help='tutorial files type')
args = parser.parse_args()
LIB_ANALYSIS_PATH = os.path.join(ANALYSIS_PATH, args.LIB)

# base
class CodeLoader(ABC):
    @abstractmethod
    def load_json(self, source):
        pass

    def save_as_py(self, source, code):
        """Save the code to a .py file."""
        filename = os.path.basename(source)
        base = os.path.splitext(filename)[0]
        py_filepath = os.path.join(LIB_ANALYSIS_PATH, "Git_Tut_py", f"{base}.py")
        with open(py_filepath, 'w') as f:
            f.write(code)

# html
class HtmlCodeLoader(CodeLoader):
    def load_json(self, filepath):
        html_dict = self._generate_html_dict(filepath)
        return html_dict

    def _generate_html_dict(self, filepath):
        directory = os.path.dirname(filepath)
        html_dict = {}
        print('-------', directory, os.listdir(directory))
        for filename in os.listdir(directory):
            if filename.endswith(".html"):
                key = filename.split('.')[0]
                html_dict[key] = self._extract_code_and_output_from_html(os.path.join(directory, filename))
        # Save the dictionary into a json file
        with open(os.path.join(directory, 'html_code_dict.json'), 'w') as f:
            json.dump(html_dict, f, indent=4)
        return html_dict

    def _extract_code_and_output_from_html(self, filepath):
        with open(filepath, 'r') as file:
            contents = file.read()
        soup = BeautifulSoup(contents, 'html.parser')
        # Prepare an empty list to store the ordered blocks
        ordered_blocks = []
        for block in soup.body.descendants:
            if block.name == 'p':
                text = block.get_text().strip()
                ordered_blocks.append(('text', text))
            elif block.name == 'div' and block.get('class') in [['input_area'], ['highlight']]:
                code = block.get_text().strip()
                code = self.clean_code_blocks(code) if self.check_ifcodeblock(code) else ''
                ordered_blocks.append(('code', code))
        # Combine consecutive 'text' or 'code' blocks into one, and store in a list of dicts
        combined_blocks = []
        current_dict = {}
        current_type = ordered_blocks[0][0]
        current_content = []
        for block in ordered_blocks:
            if block[0] == current_type:
                current_content.append(block[1])
            else:
                current_dict[current_type] = ' '.join(current_content)
                if len(current_dict) == 2:  # both 'text' and 'code' are found
                    combined_blocks.append(current_dict)
                    current_dict = {}
                current_type = block[0]
                current_content = [block[1]]
        if current_content:  # append the last block
            current_dict[current_type] = ' '.join(current_content)
        if current_dict:  # append the last dict if it's not empty
            combined_blocks.append(current_dict)
        return combined_blocks
    def clean_code_blocks(self, code_block):
        # remove leading digit 
        code_block = self.remove_line_numbers(code_block)
        # for tutorials coded in jupyter format, sometimes exist leading digit
        code_block = code_block.strip().split('\n')
        cleaned_lines = []
        for line in code_block:
            # Skip empty lines
            if len(line.strip())==0:
                continue
            cleaned_lines.append(line)
        code_block = cleaned_lines
        return '\n'.join(code_block)

    def remove_line_numbers(self, code_string):
        lines = code_string.split('\n')
        last_number = 0
        for i, line in enumerate(lines):
            match = re.match(r'^\s*(\d+)', line)
            if match:
                current_number = int(match.group(1))
                if last_number + 1 == current_number:
                    lines[i] = re.sub(r'^\s*\d+', '', line)
                    last_number = current_number
        return '\n'.join(lines)
        
    def check_ifcodeblock(self,block):
        # Filter code that is code, rather than output, from a jupyter notebook code block
        lines = block.split('\n')
        # If exist any function call
        try:
            tree = ast.parse(block)
            has_function_call = any(isinstance(node, ast.Call) for node in ast.walk(tree))
            if has_function_call:
                return has_function_call
        except:
            pass
        # maybe check function call is enough
        for line in lines:
            try:
                float(line)
                continue
            except ValueError:
                if line.strip().isdigit() or line.strip().replace('.', '').isdigit():
                    continue
                elif set(line.strip()) <= set('0123456789 '):
                    continue
                elif line.strip().startswith('{') and line.strip().endswith('}'):
                    continue
                else:
                    return True
        return False
    def save_as_json(self, directory, html_dict):
        """Save the html_dict to a .json file."""
        with open(os.path.join(directory, 'html_code_dict.json'), 'w') as f:
            json.dump(html_dict, f, indent=4)
    def save_as_code(self,json_input, directory):
        for key, value in json_input.items():
            file_name = key.replace(' ', '_').replace('\\u2014', '-') + '.py'
            code_snippets = [item['code'] for item in value if 'code' in item]
            with open(os.path.join(directory, file_name), 'w') as f:
                for snippet in code_snippets:
                    for code_line in snippet.split('\n'):
                        # check if code_line is valid
                        if code_line.startswith('['):
                            continue
                        try:
                            ast.parse(code_line)
                            f.write(code_line + '\n\n')
                        except:
                            continue

# url
class URLCodeLoader(CodeLoader):
    def load_json(self, url):
        raise NotImplementedError
        response = requests.get(url)
        return response.text
# ipynb
class IpynbCodeLoader(CodeLoader):
    def load_json(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)
        markdown_text = None
        result = []
        for cell in notebook_content.cells:
            if cell.cell_type == "markdown":
                markdown_text = cell.source
            elif cell.cell_type == "code":
                code_with_text = {}
                code_with_text['code'] = ['import subprocess']
                code_with_text['code'] = [self.process_line(i) for i in cell.source.split('\n') if i]
                if markdown_text is not None:
                    code_with_text['text'] = markdown_text
                    markdown_text = None
                else:
                    code_with_text['text'] = ''
                result.append(code_with_text)
        return result
    
    def process_line(self, line):
        if line.startswith('#'):
            new_line = line
        elif 'sns.plt.show()' in line:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            image_name = f"'figure_{timestamp}.jpg'"
            new_line = line.replace('sns.plt.show()', f'plt.savefig("{image_name}")')
        elif 'plt.show()' in line:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            image_name = f"'figure_{timestamp}.jpg'"
            new_line = line.replace('plt.show()', f'plt.savefig("{image_name}")')
        elif line.startswith('%'):
            new_line = '# ' + line
        elif line.startswith('!'):
            command = line[1:]
            new_line = f"subprocess.run('{command}', shell=True)"
        elif 'display' in line:
            # Replacing Jupyter's display function with print for simplicity.
            # This may not work as expected for non-text data.
            new_line = line.replace('display(', 'print(')
        else:
            new_line = line
        return new_line

class CodeLoaderContext:
    def __init__(self, input_folder, output_folder, strategy_type='ipynb',file_types=['py','ipynb','html']):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.loader = self.get_loader(strategy_type)
        self.strategy_type = strategy_type
        self.file_types = file_types
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def get_loader(self, strategy_type):
        if strategy_type == 'html':
            return HtmlCodeLoader()
        elif strategy_type == 'ipynb':
            return IpynbCodeLoader()
        else:
            return CodeLoader()

    def save_json_to_code(self, result, output_filename):
        with open(output_filename, 'w', encoding='utf-8') as f:
            for item in result:
                if 'text' in item:
                    f.write("# " + item['text'].replace('\n', '\n# ') + "\n")
                if 'code' in item:
                    f.write('\n'.join(item['code']) + '\n')

    def load_and_save(self):
        self.code_json = {}
        count=0
        for root, dirs, files in os.walk(self.input_folder):
            for file in files:
                if file.split('.')[-1] in self.file_types: # ,'rst'
                    count+=1
                    filepath = os.path.join(root, file)
                    code = self.loader.load_json(filepath)
                    self.code_json[file.split('.')[0]]=code
                    if file.split('.')[-1]=='ipynb':
                        self.save_json_to_code(code,os.path.join(self.output_folder, file.replace('.ipynb', '.py')))
                    elif file.split('.')[-1]=='html':
                        self.loader.save_as_code(code, self.output_folder)
        if count==0:
            print(f'Empty input folder, no files found in type {self.file_types}')
        else:
            print(f'Have successfully turned files in type {self.file_types} to python code!')
    
    def execute(self):
        import os
        os.environ['MPLBACKEND'] = 'Agg'
        success_files = []
        error_files = []
        for root, dirs, files in os.walk(self.output_folder):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        result = subprocess.run(['python', filepath], capture_output=True, text=True, cwd=root)
                        if result.returncode != 0:
                            print(f"Error executing {filepath}: {result.stderr}")
                            error_files.append(filepath)
                        else:
                            print(f"Execution of {filepath} completed successfully.")
                            success_files.append(filepath)
                    except Exception as e:
                        print(f"Error occurred while executing the code in {filepath}. Error: {str(e)}")
                        error_count += 1
                        error_files.append(filepath)
        print(f"\nExecution finished: {len(success_files)} files successfully executed, {len(error_files)} files had errors.")
        print(f"Successful files: {success_files}")
        print(f"Files with errors: {error_files}")

def main_convert_tutorial_to_py(LIB_ANALYSIS_PATH, strategy_type='ipynb', file_types=['ipynb'], execute = False):
    input_folder = os.path.join(LIB_ANALYSIS_PATH,"Git_Tut")
    output_folder = os.path.join(LIB_ANALYSIS_PATH,"Git_Tut_py")

    context = CodeLoaderContext(input_folder, output_folder, strategy_type,file_types)
    context.load_and_save()
    if execute: # if check each ipynb can run, this will cost a lot of time!
        context.execute()
    return context

if __name__=='__main__':
    main_convert_tutorial_to_py(LIB_ANALYSIS_PATH, strategy_type=args.file_type, file_types=[args.file_type])