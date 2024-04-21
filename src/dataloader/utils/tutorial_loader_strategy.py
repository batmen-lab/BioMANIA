from abc import ABC, abstractmethod
import os, re, ast, requests, nbformat, subprocess
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Any, Dict, List
from ...configs.model_config import READTHEDOC_PATH, get_all_variable_from_cheatsheet
from ...gpt.utils import save_json

# base
class CodeLoader(ABC):
    @abstractmethod
    def load_json(self, source: str) -> Any:
        """
        Abstract method to load data from a given source.
        
        Parameters
        ----------
        source : str
            The data source from which to load the data.
        
        Returns
        -------
        Any
            The data loaded from the source.
        """
        pass

    def save_as_py(self, source: str, code: str, LIB_ANALYSIS_PATH: str) -> None:
        """
        Save the code to a .py file in a specified directory.
        
        Parameters
        ----------
        source : str
            The source path of the original data.
        code : str
            The Python code to save.
        LIB_ANALYSIS_PATH : str
            The directory path where the .py file will be saved.
        """
        filename = os.path.basename(source)
        base = os.path.splitext(filename)[0]
        py_filepath = os.path.join(LIB_ANALYSIS_PATH, "Git_Tut_py", f"{base}.py")
        with open(py_filepath, 'w') as f:
            f.write(code)

# html
class HtmlCodeLoader(CodeLoader):
    def load_json(self, filepath: str) -> Dict[str, Any]:
        """
        Load JSON data from an HTML file.
        
        Parameters
        ----------
        filepath : str
            The file path of the HTML file from which to load data.
        
        Returns
        -------
        Dict[str, Any]
            A dictionary containing the loaded data.
        """
        html_dict = self._generate_html_dict(filepath)
        return html_dict

    def _generate_html_dict(self, filepath: str) -> Dict[str, Any]:
        """
        Generate a dictionary from HTML file paths.
        
        Parameters
        ----------
        filepath : str
            The base directory path to search for HTML files.
        
        Returns
        -------
        Dict[str, Any]
            A dictionary mapping unique keys to extracted code and output from HTML files.
        """
        base_directory = os.path.dirname(filepath)
        html_dict = {}
        for root, dirs, files in os.walk(base_directory):
            for filename in files:
                if filename.endswith(".html"):
                    full_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(full_path, start=base_directory)
                    key = relative_path.replace(os.path.sep, "_dot_").replace('.html','')
                    html_dict[key] = self._extract_code_and_output_from_html(full_path)
        json_output_path = os.path.join(base_directory, 'html_code_dict.json')
        #print(f'save to {json_output_path}')
        save_json(json_output_path, html_dict)
        return html_dict

    def _extract_code_and_output_from_html(self, filepath: str) -> List[Dict[str, str]]:
        """
        Extract code blocks and associated text outputs from an HTML file.
        
        Parameters
        ----------
        filepath : str
            The path to the HTML file from which to extract code.
        
        Returns
        -------
        List[Dict[str, str]]
            A list of dictionaries each containing 'text' and 'code' keys.
        """
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
        #combined_blocks = ordered_blocks.copy()
        # Combine consecutive 'text' or 'code' blocks into one, and store in a list of dicts
        combined_blocks = []
        current_dict = {}
        current_type = ordered_blocks[0][0]
        current_content = []
        for block in ordered_blocks:
            if block[0] == current_type:
                current_content.append(block[1])
            else:
                separator = '\n' if current_type == 'code' else ' '
                current_dict[current_type] = separator.join(current_content)
                if len(current_dict) == 2:  # both 'text' and 'code' are found
                    combined_blocks.append(current_dict)
                    current_dict = {}
                current_type = block[0]
                current_content = [block[1]]
        if current_content:  # append the last block
            separator = '\n' if current_type == 'code' else ' '
            current_dict[current_type] = separator.join(current_content)
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
    def save_as_json(self, directory: str, html_dict: Dict[str, Any]) -> None:
        """
        Save the html_dict data to a .json file.
        
        Parameters
        ----------
        directory : str
            The directory where the .json file will be saved.
        html_dict : Dict[str, Any]
            The dictionary containing the data to be saved.
        """
        save_json(os.path.join(directory, 'html_code_dict.json'), html_dict)
    def save_as_code(self, json_input: Dict[str, List[Dict[str, str]]], directory: str) -> None:
        """
        Save extracted code blocks from JSON input to .py files.
        
        Parameters
        ----------
        json_input : Dict[str, List[Dict[str, str]]]
            The dictionary containing code blocks to be saved.
        directory : str
            The directory where the .py files will be saved.
        """
        for key, value in json_input.items():
            file_name = key.replace(' ', '_').replace('\\u2014', '-') + '.py'
            code_snippets = [item['code'] for item in value if 'code' in item]
            py_filepath = os.path.join(directory, file_name)
            # 240407: add support for multiple line code
            with open(py_filepath, 'w') as f:
                accumulate_code = ''
                accumulate_flag = False
                brackets = {'curly': 0, 'square': 0, 'round': 0}
                for snippet in code_snippets:
                    for code_line in snippet.split('\n'):
                        if code_line.startswith('['):
                            continue
                        brackets['curly'] += code_line.count('{') - code_line.count('}')
                        brackets['square'] += code_line.count('[') - code_line.count(']')
                        brackets['round'] += code_line.count('(') - code_line.count(')')
                        all_closed = all(b == 0 for b in brackets.values())
                        if not all_closed:
                            accumulate_code += code_line + '\n'
                            continue
                        else:
                            accumulate_code += code_line + '\n'
                            try:
                                ast.parse(accumulate_code)
                                f.write(accumulate_code + '\n\n')
                            except:
                                pass
                            accumulate_code = ''
                            brackets = {'curly': 0, 'square': 0, 'round': 0}

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
                markdown_text = cell.source + '\n'
            elif cell.cell_type == "code":
                code_with_text = {}
                code_with_text['code'] = ['import subprocess']
                code_with_text['code'] = [self.process_line(i) + '\n' for i in cell.source.split('\n') if i]
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
    def __init__(self, input_folder: str, output_folder: str, strategy_type: str = 'ipynb', file_types: List[str] = ['py', 'ipynb', 'html']) -> None:
        """
        Initialize a context for converting code between formats.

        Parameters
        ----------
        input_folder : str
            The folder from which files are loaded.
        output_folder : str
            The folder where converted files will be saved.
        strategy_type : str, optional
            The type of loader to use. Default is 'ipynb'.
        file_types : List[str], optional
            The file types to be processed. Default is ['py', 'ipynb', 'html'].
        """
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

    def load_and_save(self) -> None:
        """
        Load files from the input folder, convert them, and save them in the output folder.
        """
        self.code_json = {}
        count=0
        for root, dirs, files in os.walk(self.input_folder):
            for file in files:
                if file.split('.')[-1] in self.file_types: # ,'rst'
                    count+=1
                    filepath = os.path.join(root, file)
                    #file_key_base = os.path.splitext(file)[0]
                    #file_path_key = os.path.relpath(filepath, start=self.input_folder).replace(os.path.sep, "_dot_")
                    if file.split('.')[-1]=='ipynb':
                        code = self.loader.load_json(filepath)
                        self.code_json[file.split('.')[0]]=code
                        self.save_json_to_code(code,os.path.join(self.output_folder, file.replace('.ipynb', '.py')))
                    elif file.split('.')[-1]=='html':
                        code = self.loader.load_json(filepath)
                        for k, v in code.items():
                            for item in v:
                                if 'code' not in item:
                                    item['code'] = ""
                                #else:
                                #    item['code'] = item['code'].split('\n')
                        updated_code = {k: v for k, v in code.items()} # f"{file_path_key}_dot_{k}"
                        self.loader.save_as_code(updated_code, self.output_folder)
                        for k, v in updated_code.items():
                            for item in v:
                                if not item['code']:
                                    item['code'] = []
                                else:
                                    item['code'] = item['code'].split('\n')
                        self.code_json.update(updated_code)
        if count==0:
            print(f'Empty input folder, no files found in type {self.file_types}')
        else:
            print(f'Have successfully turned files in type {self.file_types} to python code to path {self.output_folder}!')
    
    def execute(self) -> None:
        """
        Execute the saved Python files in the output folder and report the results.
        """
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

def main_convert_tutorial_to_py(LIB_ANALYSIS_PATH: str, subpath: str = 'Git_Tut', strategy_type: str = 'ipynb', file_types: List[str] = ['ipynb'], execute: bool = False) -> CodeLoaderContext:
    """
    Main function to convert tutorial files to Python script and optionally execute them.

    Parameters
    ----------
    LIB_ANALYSIS_PATH : str
        The path to the library analysis directory.
    subpath : str, optional
        The subpath within the library analysis directory. Default is 'Git_Tut'.
    strategy_type : str, optional
        The type of files to convert. Default is 'ipynb'.
    file_types : List[str], optional
        The types of files to convert. Default is ['ipynb'].
    execute : bool, optional
        Whether to execute the converted Python scripts. Default is False.

    Returns
    -------
    CodeLoaderContext
        An instance of CodeLoaderContext configured with the specified parameters.
    """
    input_folder = os.path.join(LIB_ANALYSIS_PATH, subpath)
    output_folder = os.path.join(LIB_ANALYSIS_PATH, subpath+'_py')
    context = CodeLoaderContext(input_folder, output_folder, strategy_type,file_types)
    context.load_and_save()
    if execute: # if check each ipynb is runable, this will cost a lot of time!
        context.execute()
    return context

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--LIB', type=str, required=True, help='PyPI tool')
    parser.add_argument('--file_type', type=str, default='ipynb', help='tutorial files type')
    parser.add_argument('--source', type=str, default='Git', help='transfer file from Git or Readthedoc')
    args = parser.parse_args()
    info_json = get_all_variable_from_cheatsheet(args.LIB)
    API_HTML, TUTORIAL_GITHUB, TUTORIAL_HTML, LIB_ANALYSIS_PATH = [info_json[key] for key in ['API_HTML', 'TUTORIAL_GITHUB', 'TUTORIAL_HTML', 'LIB_ANALYSIS_PATH']]
    if args.source=="Readthedoc":
        readthedoc_realpath = os.path.join(READTHEDOC_PATH, TUTORIAL_HTML)
        main_convert_tutorial_to_py(readthedoc_realpath, subpath='', strategy_type=args.file_type, file_types=[args.file_type])
    elif args.source=="Git":
        main_convert_tutorial_to_py(LIB_ANALYSIS_PATH, subpath='Git_Tut', strategy_type=args.file_type, file_types=[args.file_type])
    else:
        raise NotImplementedError
