# Flask
from flask import Flask, Response, stream_with_context, request, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin
from deploy.ServerEventCallback import ServerEventCallback
from queue import Queue
app = Flask(__name__)
cors = CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
# standard lib
import argparse, json, signal, time, copy, base64, requests, importlib, inspect, ast, os, random, io, sys, pickle, shutil, subprocess, re
from datetime import datetime
from urllib.parse import urlparse
# Computational
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
# device
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 
import concurrent.futures
from dotenv import load_dotenv
from string import punctuation

# inference pipeline
from models.model import LLM_response, LLM_model
from configs.model_config import *
from inference.execution_UI import CodeExecutor
from inference.utils import find_similar_two_pairs
from inference.retriever_finetune_inference import ToolRetriever
from deploy.utils import dataframe_to_markdown, convert_image_to_base64
from prompt.parameters import prepare_parameters_prompt
from prompt.summary import prepare_summary_prompt

basic_types = ['str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set', 'any', 'List', 'Dict']

def change_format(input_params, param_name_list):
    """
    Transforms the format of input parameters based on a provided parameter name list.
    """
    output_params = []
    for param_name, param_info in input_params.items():
        if param_name in param_name_list:
            output_params.append({
                "name": param_name,
                "type": param_info["type"],
                "description": param_info["description"],
                "default_value": param_info["default"]
            })
    return output_params

def generate_api_calling(api_name, api_details, returned_content_str):
    """
    Generates an API call and formats output based on provided API details and returned content string.
    """
    returned_content_str_new = returned_content_str.replace('null', 'None').replace('None', '"None"')
    returned_content = ast.literal_eval(returned_content_str_new)
    returned_content_dict = {item['param_name']: item['value'] for item in returned_content if (item['value'] not in ['None']) and item['value']} # remove null parameters
    api_description = api_details["description"]
    parameters = api_details['Parameters']
    return_type = api_details['Returns']['type']
    parameters_dict = {}
    parameters_info_list = []
    for param_name, param_details in parameters.items():
        # only include required parameters and optional parameters found from response
        if param_name in returned_content_dict or not param_details['optional']:
            print(param_name, param_name in returned_content_dict, not param_details['optional'])
            param_type = param_details['type']
            param_description = param_details['description']
            param_value = param_details['default']
            param_optional = param_details['optional']
            if param_name in returned_content_dict:
                param_value = returned_content_dict[param_name]
                if 'str' or 'PathLike' in param_type and ('"' not in param_type and "'" not in param_type) and (param_value not in ['None', None]):
                    param_value = "'"+str(param_value)+"'"
            # added condition to differentiate between basic and non-basic types
            if any(item in param_type for item in basic_types):
                param_value = param_value if ((param_value not in [ 'None']) and param_value) else "@"
            else:
                param_value = param_value if ((param_value not in [ 'None']) and param_value) else "$"
            parameters_dict[param_name] = param_value
            parameters_info_list.append({
                'name': param_name,
                'type': param_type,
                'value': param_value,
                'description': param_description,
                'optional': param_optional
            })
    parameters_str = ", ".join(f"{k}={v}" for k, v in parameters_dict.items())
    api_calling = f"{api_name}({parameters_str})"
    output = {
        "api_name": api_name,
        "parameters": {
            param['name']: {
                "type": param['type'],
                "description": param['description'],
                "value": param['value'],
                "optional": param['optional']
            } for param in parameters_info_list
        },
        "return_type": return_type
    }
    return api_name, api_calling, output

def predict_by_similarity(user_query_vector, centroids, labels):
    similarities = [cosine_similarity(user_query_vector, centroid.reshape(1, -1)) for centroid in centroids]
    return labels[np.argmax(similarities)]

def infer(query, vectorizer, centroids, labels):
    user_query_vector = vectorizer.transform([query])
    predicted_label = predict_by_similarity(user_query_vector, centroids, labels)
    return predicted_label

if not os.path.exists('tmp'):
    os.mkdir('tmp')

def download_data(url, save_dir="tmp"):
    response = requests.head(url)
    if response.status_code == 200:
        content_length = response.headers.get('Content-Length')
        if content_length:
            size = int(content_length)
            print(f"Data size: {size} bytes!")
        else:
            print("Can not estimate data size!")
        response = requests.get(url)
        if response.status_code == 200:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            parsed_url = urlparse(url)
            file_name = os.path.basename(parsed_url.path)
            save_path = f"{save_dir}/data_{timestamp}_{file_name}"
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print("Data downloaded successfully!")
            return save_path
        else:
            print("Data downloaded failed!")
            return None
    else:
        print("Data request failed!")
        return None

def save_decoded_file(raw_file):
    filename = raw_file['filename']
    source_type = raw_file['type']
    if source_type=='file':
        data_type, decoded_data = raw_file['data'].split(",")[0].split(";")[0], base64.b64decode(raw_file['data'].split(",")[1])
        filename = os.path.join('tmp', filename)
        with open(filename, 'wb') as f:
            f.write(decoded_data)
    elif source_type=='url':
        decoded_data = raw_file['data']
        filename = download_data(decoded_data)
    return filename

class Model:
    def __init__(self):
        print("Initializing...")
        self.indexxxx = 1
        self.inuse = False
        self.get_args()
        self.query_id = 0
        self.queue = Queue()
        self.callback = ServerEventCallback(self.queue)
        self.callbacks = [self.callback]
        self.occupied = False
        self.LIB = "scanpy"
        #load_dotenv()
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-test')
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        self.initialize_executor()
        reset_result = self.reset_lib(self.LIB)
        if reset_result=='Fail':
            return
        self.user_states = "initial"
        self.parameters_info_list = None
        self.image_folder = "./tmp/images/"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder, exist_ok=True)
        self.image_file_list = []
        self.image_file_list = self.update_image_file_list()
        self.buf = io.StringIO()
        with open(f'./data/standard_process/{self.LIB}/vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        print('==>chitchat vectorizer loaded!')
        with open(f'./data/standard_process/{self.LIB}/centroids.pkl', 'rb') as f:
            self.centroids = pickle.load(f)
        print('==>chitchat vectorizer loaded!')
        self.retrieve_query_mode = "similar"
        print("Server ready")
    def reset_lib(self, lib_name):
        #lib_name = lib_name.strip()
        print('================')
        print(f'==>Start reset the Lib {lib_name}!')
        # reset and reload all the LIB-related data/models
        # suppose that all data&model are prepared already in their path
        try:
            self.ambiguous_pair = find_similar_two_pairs(lib_name)
            self.ambiguous_api = list(set(api for api_pair in self.ambiguous_pair for api in api_pair))
            self.load_data(f"./data/standard_process/{lib_name}/API_composite.json")
            print('==>loaded API json done')
            #self.load_composite_code(lib_name)
            #print('==>loaded API composite done')
            t1 = time.time()
            print('==>Start loading model!')
            self.load_llm_model()
            print('loaded llm model!')
            retrieval_model_path = self.args.retrieval_model_path
            parts = retrieval_model_path.split('/')
            if len(parts)>=3: # only work for path containing LIB, otherwise, please reenter the path in script
                if not parts[-1]:
                    parts = parts[:-1]
                new_path = '/'.join(parts)
            retrieval_model_path = new_path
            self.retriever = ToolRetriever(LIB=lib_name,corpus_tsv_path=f"./data/standard_process/{lib_name}/retriever_train_data/corpus.tsv", model_path=retrieval_model_path)
            print('loaded retriever!')
            #self.executor.execute_api_call(f"from data.standard_process.{self.LIB}.Composite_API import *", "import")
            self.executor.execute_api_call(f"import {lib_name}", "import")
            # pyteomics tutorial needs these import libs
            self.executor.execute_api_call(f"import os, gzip, numpy as np, matplotlib.pyplot as plt", "import")
            self.executor.execute_api_call(f"from urllib.request import urlretrieve", "import")
            self.executor.execute_api_call(f"from pyteomics import fasta, parser, mass, achrom, electrochem, auxiliary", "import")
            end_of_docstring_summary = re.compile(r'[{}\n]+'.format(re.escape(punctuation)))
            all_apis = {x: end_of_docstring_summary.split(self.API_composite[x]['Docstring'])[0].strip() for x in self.API_composite}
            all_apis = list(all_apis.items())
            self.description_json = {i[0]:i[1] for i in all_apis}
            print('==>Successfully loading model!')
            print('loading model cost: ', time.time()-t1, 's')
            reset_result = "Success"
            self.LIB = lib_name
        except:
            print('at least one data or model is not ready, please install lib first!')
            reset_result = "Fail"
            [callback.on_tool_start() for callback in self.callbacks]
            [callback.on_tool_end() for callback in self.callbacks]
            [callback.on_agent_action(block_id="log-" + str(self.indexxxx), task="At least one data or model is not ready, please install lib first!",task_title="Setting error") for callback in self.callbacks]
            self.indexxxx+=1
        return reset_result

    def install_lib(self,github_url, doc_url, api_html, lib_name, lib_alias):
        '''github_url = "https://github.com/biocore/scikit-bio"
        doc_url = "scikit-bio.org/docs/latest/"
        api_html = "scikit-bio.org/docs/latest/index.html"
        api_html = None 
        lib_name = "scikit-bio"
        lib_alias = "skbio"'''
        self.LIB = lib_name
        from configs.model_config import GITHUB_PATH, ANALYSIS_PATH, READTHEDOC_PATH
        from configs.model_config import LIB, LIB_ALIAS, GITHUB_LINK, API_HTML
        from dataloader.utils.code_download_strategy import download_lib
        from dataloader.utils.other_download import download_readthedoc
        from dataloader.get_API_init_from_sourcecode import main_get_API_init
        from dataloader.get_API_full_from_unittest import merge_unittest_examples_into_API_init
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="start processing new lib...",task_title="0") for callback in self.callbacks]
        self.indexxxx+=1
        os.makedirs(f"./data/standard_process/{self.LIB}/", exist_ok=True)

        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="downloading materials...",task_title="13") for callback in self.callbacks]
        self.indexxxx+=1
        """if github_url:
            download_lib('git', self.LIB, github_url, lib_alias, GITHUB_PATH)"""
        self.buf = io.StringIO()
        sys.stdout = self.buf
        sys.stderr = self.buf
        subprocess.run(['pip', 'install', f'{lib_alias}'])
        if doc_url and api_html:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            content = self.buf.getvalue()
            download_readthedoc(doc_url, api_html)
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="preparing API_init.json ...",task_title="26") for callback in self.callbacks]
        self.indexxxx+=1
        if api_html:
            api_path = os.path.normpath(os.path.join(READTHEDOC_PATH, api_html))
        else:
            api_path = None
        main_get_API_init(self.LIB,lib_alias,ANALYSIS_PATH,api_path)
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="preparing API_composite.json ...",task_title="39") for callback in self.callbacks]
        self.indexxxx+=1
        # TODO: add API_composite
        #merge_unittest_examples_into_API_init(self.LIB, ANALYSIS_PATH, GITHUB_PATH)
        #from dataloader.get_API_composite_from_tutorial import main_get_API_composite
        #main_get_API_composite(ANALYSIS_PATH, self.LIB)
        shutil.copy(f'../../resources/json_analysis/{self.LIB}/API_init.json', f'./data/standard_process/{self.LIB}/API_composite.json')

        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="training api/non-api classification model ...",task_title="52") for callback in self.callbacks]
        self.indexxxx+=1
        command = [
            "python",
            "models/chitchat_classification.py",
            "--LIB", self.LIB,
        ]
        subprocess.run(command)
        base64_image = convert_image_to_base64(f"./plot/{self.LIB}/chitchat_test_tsne_modified.png")
        [callback.on_agent_action(block_id="transfer_" + str(self.indexxxx),task=base64_image,task_title="chitchat_train_tsne_modified.png",) for callback in self.callbacks]
        self.indexxxx+=1
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="preparing retriever data API_inquiry.json ...",task_title="65") for callback in self.callbacks]
        self.indexxxx+=1
        command = [
            "python", "dataloader/preprocess_retriever_data.py",
            "--LIB", self.LIB
        ]
        subprocess.run(command)
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="training retriever...",task_title="78") for callback in self.callbacks]
        self.indexxxx+=1
        subprocess.run(["mkdir", f"./hugging_models/retriever_model_finetuned/{self.LIB}"])
        command = [
            "python",
            "models/train_retriever.py",
            "--data_path", f"./data/standard_process/{self.LIB}/retriever_train_data/",
            "--model_name", "bert-base-uncased",
            "--output_path", f"./hugging_models/retriever_model_finetuned/{self.LIB}",
            "--num_epochs", "25",
            "--train_batch_size", "32",
            "--learning_rate", "1e-5",
            "--warmup_steps", "500",
            "--max_seq_length", "256",
            "--optimize_top_k", "3",
            "--plot_dir", f"./plot/{self.LIB}/retriever/"
        ]
        subprocess.run(command)
        base64_image = convert_image_to_base64(f"./plot/{self.LIB}/retriever/ndcg_plot.png")
        [callback.on_agent_action(block_id="transfer_" + str(self.indexxxx),task=base64_image,task_title="ndcg_plot.png",) for callback in self.callbacks]
        self.indexxxx+=1
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Process done! Please restart the program for new usage",task_title="100") for callback in self.callbacks]
        self.indexxxx+=1
        # TODO: need to add the new materials url into cheatsheet, avoid repeated entering

    def update_image_file_list(self):
        image_file_list = [f for f in os.listdir(self.image_folder) if f.endswith(".png")]
        return image_file_list
    def get_args(self):
        # Define the arguments here...
        parser = argparse.ArgumentParser(description="Inference Pipeline")
        parser.add_argument("--retrieval_model_path", type=str, default='./hugging_models/retriever_model_finetuned/scanpy/assigned', help="Path to the retrieval model")
        parser.add_argument("--top_k", type=int, default=3, help="Top K value for the retrieval")
        self.args = parser.parse_args()
    def load_composite_code(self, lib_name):
        # deprecated
        module_name = f"data.standard_process.{lib_name}.Composite_API"
        module = importlib.import_module(module_name)
        source_code = inspect.getsource(module)
        tree = ast.parse(source_code)
        self.functions_json = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                function_body = ast.unparse(node)
                self.functions_json[function_name] = function_body
    def retrieve_names(self,query):
        retrieved_names = self.retriever.retrieving(query, top_k=self.args.top_k)
        return retrieved_names
    def initialize_executor(self):
        self.executor = CodeExecutor()
        self.executor.callbacks = self.callbacks
    def load_llm_model(self):
        self.llm, self.tokenizer = LLM_model()
    def load_data(self, API_file):
        with open(API_file, 'r') as json_file:
            data = json.load(json_file)
        self.API_composite = data
    def generate_file_loading_code(self, file_path, file_type):
        # Define the loading code for each file type
        file_loading_templates = {
            '.txt': 'with open("{path}", "r") as f:\n\tdata_{idx} = f.read()',
            '.csv': 'import pandas as pd\ndata_{idx} = pd.read_csv("{path}")',
            '.xlsx': 'import pandas as pd\ndata_{idx} = pd.read_excel("{path}")',
            '.pdf': 'import PyPDF2\nwith open("{path}", "rb") as f:\n\treader = PyPDF2.PdfFileReader(f)\n\tdata_{idx} = [reader.getPage(i).extractText() for i in range(reader.numPages)]',
            '.py': 'with open("{path}", "r") as f:\n\tdata_{idx} = f.read()\nexec(data_{idx})',  
        }
        # Find the next available index for variable naming in self.executor.variables
        idx = 1
        while f"data_{idx}" in self.executor.variables:
            idx += 1
        # Get the loading code for the given file type
        loading_code_template = file_loading_templates.get(file_type, '')
        loading_code = loading_code_template.format(path=file_path, idx=idx)
        return loading_code
    def loading_data(self, files, verbose=False):
        for ids, file_path in enumerate(files):
            file_extension = os.path.splitext(file_path)[1]
            loading_code = self.generate_file_loading_code(file_path, file_extension)
            self.executor.execute_api_call(loading_code, "code")
            if verbose:
                [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="uploading files..."+str(ids+1)+'/'+str(len(files)),task_title=str(int((ids+1)/len(files)*100))) for callback in self.callbacks]
        if verbose:
            self.indexxxx+=1
            [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="uploading files finished!",task_title=str(int(100))) for callback in self.callbacks]
            self.indexxxx+=1
    def run_pipeline(self, user_input, lib, top_k=3, files=[],conversation_started=True):
        self.indexxxx = 1
        # only reset lib when changing lib
        if lib!=self.LIB:
            reset_result = self.reset_lib(lib)
            if reset_result=='Fail':
                print('Reset lib fail! Exit the dialog!')
                return 
            self.LIB = lib
        self.conversation_started=conversation_started
        # only clear namespace when starting new conversations
        if self.conversation_started in ["True", True]:
            print('==>new conversation_started!')
            print('reset status!')
            self.user_states="initial"
            self.initialize_executor()
            self.executor.variables={}
            self.executor.execute_code=[]
            for var_name in list(globals()):
                if var_name.startswith('result_'):
                    del globals()[var_name]
            for var_name in list(locals()):
                if var_name.startswith('result_'):
                    del locals()[var_name]
        else:
            print('==>old conversation_continued!')
        if self.user_states == "initial":
            print('start initial!')
            while not self.queue.empty():
                self.queue.get()
            self.loading_data(files)
            print('loading data finished!')
            self.query_id += 1
            temp_args = copy.deepcopy(self.args)
            temp_args.top_k = top_k
            self.user_query = user_input
            predicted_source = infer(self.user_query, self.vectorizer, self.centroids, ['chitchat-data', 'topical-chat', 'api-query'])
            print(f'----query inferred as {predicted_source}----')
            if predicted_source!='api-query':
                print('--classified as chitchat!--')
                [callback.on_tool_start() for callback in self.callbacks]
                [callback.on_tool_end() for callback in self.callbacks]
                response, _ = LLM_response(self.llm, self.tokenizer, user_input, history=[], kwargs={})  # llm
                [callback.on_agent_action(block_id="log-" + str(self.indexxxx), task=response,task_title="Non API chitchat") for callback in self.callbacks]
                self.indexxxx+=1
                return
            else:
                pass
            retrieved_names = self.retrieve_names(user_input)
            # produce prompt
            description_jsons = {}
            for i in retrieved_names:
                description_jsons[i] = self.description_json[i]
            if self.retrieve_query_mode=='similar':
                instruction_shot_example = self.retriever.retrieve_similar_queries(user_input, shot_k=5)
            else:
                sampled_shuffled = random.sample(self.retriever.shuffled_data, 5)
                instruction_shot_example = "".join(["\nInstruction: " + ex['query'] + "\nFunction: " + ex['gold'] for ex in sampled_shuffled])
            api_predict_prompt = f"""
            Task: choose one of the following APIs to use for the instruction. 
            {json.dumps(description_jsons)}
            {instruction_shot_example}
            Instruction: {user_input}
            API:
            """
            success = False
            for attempt in range(3):
                try:
                    print(f'==>Ask GPT: {api_predict_prompt}')
                    response, _ = LLM_response(self.llm, self.tokenizer, api_predict_prompt, history=[], kwargs={})  # llm
                    print('==>GPT response:', response)
                    # hack for if GPT answers this or that
                    response = response.split(',')[0].split("(")[0].split(' or ')[0]
                    response = response.split(':')[0]# for robustness, sometimes gpt will return api:description
                    self.description_json[response]
                    self.predicted_api_name = response 
                    success = True
                    break
                except Exception as e:
                    [callback.on_tool_start() for callback in self.callbacks]
                    [callback.on_tool_end() for callback in self.callbacks]
                    [callback.on_agent_action(block_id="log-" + str(self.indexxxx), task=f"GPT predicted API error: {response}. Please re-design the query and re-enter.",task_title="GPT predict Error",) for callback in self.callbacks]
                    self.indexxxx += 1
                    return
            if not success:
                [callback.on_agent_action(block_id="log-" + str(self.indexxxx),task="GPT can not return valid API name prediction, please try other query.",task_title="GPT predict Error",) for callback in self.callbacks]
                self.indexxxx += 1
                return
            print(f'length of ambiguous api list: {len(self.ambiguous_api)}')
            # if the predicted API is in ambiguous API list, then print those API and select one from them
            if self.predicted_api_name in self.ambiguous_api:
                filtered_pairs = [api_pair for api_pair in self.ambiguous_pair if self.predicted_api_name in api_pair]
                self.filtered_api = list(set(api for api_pair in filtered_pairs for api in api_pair))
                print(f'the filtered api list is {self.filtered_api}')
                [callback.on_tool_start() for callback in self.callbacks]
                [callback.on_tool_end() for callback in self.callbacks]
                next_str = ""
                idx_api = 1
                for api in self.filtered_api:
                    if idx_api>1:
                        next_str+='\n'
                    next_str+=f"Candidate [{idx_api}]: {api}"
                    description_1 = self.API_composite[api]['Docstring'].split("\n")[0]
                    next_str+='\n'+description_1
                    self.user_states = "ambiguous_mode"
                    idx_api+=1
                [callback.on_agent_action(block_id="log-" + str(self.indexxxx), task=next_str,task_title=f"Can you confirm which of the following {len(self.filtered_api)} candidates") for callback in self.callbacks]
                self.indexxxx += 1
            else:
                self.user_states = "after_API_selection"
                self.run_pipeline_after_fixing_API_selection(user_input)
        elif self.user_states == "ambiguous_mode":
            ans = self.run_pipeline_after_ambiguous(user_input)
            if ans in ['break']:
                return
            self.run_pipeline_after_fixing_API_selection(user_input)
        elif self.user_states == "select_params":
            self.run_select_params(user_input)
        elif self.user_states == "enter_params":
            self.run_pipeline_after_entering_params(user_input)
    def run_pipeline_after_ambiguous(self,user_input):
        user_input = user_input.strip()
        [callback.on_tool_start() for callback in self.callbacks]
        [callback.on_tool_end() for callback in self.callbacks]
        try:
            int(user_input)
        except:
            [callback.on_agent_action(block_id="log-" + str(self.indexxxx), task="Error: the input is not a number.\nPlease re-enter the index", task_title="Index Error") for callback in self.callbacks]
            self.indexxxx += 1
            self.user_states = "ambiguous_mode"
            return 'break'
        try:
            self.filtered_api[int(user_input)-1]
        except:
            [callback.on_agent_action(block_id="log-" + str(self.indexxxx), task="Error: the input index exceed the maximum length of ambiguous API list\nPlease re-enter the index",task_title="Index Error") for callback in self.callbacks]
            self.indexxxx += 1
            self.user_states = "ambiguous_mode"
            return 'break'
        self.user_states = "after_API_selection"
        self.predicted_api_name = self.filtered_api[int(user_input)-1]
    def process_api_info(self, api_info, single_api_name):
        relevant_apis = api_info.get(single_api_name, {}).get("relevant APIs")
        if not relevant_apis:
            return [{single_api_name: {'type': api_info[single_api_name]['api_type']}}]
        else:
            relevant_api_list = []
            for relevant_api_name in relevant_apis:
                relevant_api_type = api_info.get(relevant_api_name, {}).get("api_type")
                relevant_api_list.append({relevant_api_name: {'type': relevant_api_type}})
            return relevant_api_list
    def check_and_insert_class_apis(self, api_info, result):
        prefixes = set()
        class_apis = []
        for api_data in result:
            api_name = list(api_data.keys())[0]
            parts = api_name.split(".")
            for i in range(1, len(parts)):
                prefix = ".".join(parts[:i])
                if prefix not in prefixes and api_info.get(prefix) and api_info[prefix]["api_type"] == "class":
                    prefixes.add(prefix)
                    class_apis.append({prefix: {'type': api_info[prefix]['api_type']}})
        updated_result = result.copy()
        for class_api in class_apis:
            class_api_name = list(class_api.keys())[0]
            index_to_insert = None
            for i, api_data in enumerate(updated_result):
                if list(api_data.keys())[0] == ".".join(class_api_name.split(".")[:-1]):
                    index_to_insert = i
                    break
            if index_to_insert is not None:
                updated_result.insert(index_to_insert, class_api)
            else:
                updated_result.append(class_api)
        return {api_name: content for item in updated_result for api_name, content in item.items()}

    def run_pipeline_after_fixing_API_selection(self,user_input):
        # check if composite API/class method API, return the relevant APIs
        relevant_api_list = self.process_api_info(self.API_composite, self.predicted_api_name)
        self.api_name_json = self.check_and_insert_class_apis(self.API_composite, relevant_api_list)
        self.user_states = "initial"
        api_description = self.API_composite[self.predicted_api_name]['description']
        # summary task, TODO: check
        summary_prompt = prepare_summary_prompt(user_input, self.predicted_api_name, api_description, self.API_composite[self.predicted_api_name]['Parameters'],self.API_composite[self.predicted_api_name]['Returns'])
        response, _ = LLM_response(self.llm, self.tokenizer, summary_prompt, history=[], kwargs={})  
        
        [callback.on_tool_start() for callback in self.callbacks]
        [callback.on_tool_end() for callback in self.callbacks]
        [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task=response,task_title=f"Predicted API: {self.predicted_api_name}",) for callback in self.callbacks]
        self.indexxxx+=1

        combined_params = {}
        for api in relevant_api_list:
            for api_name, api_details in api.items():
                combined_params.update(self.API_composite[api_name]['Parameters'])
        parameters_name_list = [key for key, value in combined_params.items() if (not value['optional']) and (key not in ['path', "Path"])]
        api_parameters_information = change_format(combined_params, parameters_name_list)
        # filter out special type parameters, do not infer them using gpt
        api_parameters_information = [param for param in api_parameters_information if param['type'] in basic_types]
        parameters_name_list = [param_info['name'] for param_info in api_parameters_information]
        print('api_parameters_information:', api_parameters_information)
        apis_description = ""
        apis_name = ""
        for idx,api_name_tmp_list in enumerate(relevant_api_list):
            if len(relevant_api_list)>1:
                api_name_tmp = list(api_name_tmp_list.keys())[0]
                apis_name+=f"{idx}:{api_name_tmp}"
                apis_description+=f"{idx}:{self.API_composite[api_name_tmp]['description']}."
            else:
                api_name_tmp = list(api_name_tmp_list.keys())[0]
                apis_name+=f"{api_name_tmp}"
                apis_description+=f"{self.API_composite[api_name_tmp]['description']}."
        parameters_prompt = prepare_parameters_prompt(self.user_query, apis_description, apis_name, 
        json.dumps(api_parameters_information), json.dumps(parameters_name_list))  # prompt
        print('parameters_prompt: ', parameters_prompt)

        if len(parameters_name_list)==0:
            # if there is no required parameters, skip using gpt
            response = "[]"
        else:
            success = False
            for attempt in range(3):
                try:
                    response, _ = LLM_response(self.llm, self.tokenizer, parameters_prompt, history=[], kwargs={})  
                    print('GPT response:', response)
                    returned_content_str_new = response.replace('null', 'None').replace('None', '"None"')
                    returned_content = ast.literal_eval(returned_content_str_new)
                    success = True
                    break
                except Exception as e:
                    [callback.on_agent_action(block_id="log-" + str(self.indexxxx), task="API key error: " + str(e),task_title="GPT predict Error",) for callback in self.callbacks]
                    self.indexxxx += 1
                    return
            if not success:
                [callback.on_agent_action(block_id="log-" + str(self.indexxxx),task="GPT can not return valid parameters prediction, please redesign prompt in backend.",task_title="GPT predict Error",) for callback in self.callbacks]
                self.indexxxx += 1
                return
            print('The prompt is: ', parameters_prompt)
        # generate api_calling
        self.predicted_api_name, api_calling, self.parameters_info_list = generate_api_calling(self.predicted_api_name, self.API_composite[self.predicted_api_name], response)
        
        print('After GPT predicting parameters, now the produced API calling is :', api_calling)
        ####### infer parameters
        # $ param
        print("Automatically selected params for $, original parameters are: ", self.parameters_info_list['parameters'])
        self.selected_params = self.executor.select_parameters(self.parameters_info_list['parameters'])
        print("Automatically selected params for $, after selection the parameters are: ", self.selected_params)
        # $ param if not fulfilled
        none_dollar_value_params = [param_name for param_name, param_info in self.selected_params.items() if param_info["value"] in ['$']]
        if none_dollar_value_params:
            print('none_dollar_value_params:', none_dollar_value_params)
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task="However, there are still some parameters with special type undefined. Please start from uploading data, or input your query from preprocessing dataset.",task_title="Missing Parameters: special type") for callback in self.callbacks]
            self.indexxxx+=1
            return
        # @ param
        self.none_at_value_params = [param_name for param_name, param_info in self.selected_params.items() if (param_info["value"] in ['@']) and (param_name not in ['path','Path'])]
        self.filtered_params = {key: value for key, value in self.parameters_info_list['parameters'].items() if (value["value"] in ['@']) and (key not in ['path','Path'])}
        self.filtered_pathlike_params = {}
        self.filtered_pathlike_params = {key: value for key, value in self.parameters_info_list['parameters'].items() if (value["value"] in ['@']) and (key in ['path','Path'])}
        # TODO: add condition later: if uploading data files, 
        # avoid asking Path params, assign it as './tmp'
        if self.none_at_value_params: # TODO: add type PathLike
            print('if exist non path, basic type parameters, start selecting parameters')
            print(self.none_at_value_params)
            tmp_input_para = ""
            for idx, api in enumerate(self.filtered_params):
                if idx!=0:
                    tmp_input_para+=" and "
                tmp_input_para+=self.filtered_params[api]['description']
                tmp_input_para+=f"('{api}': {self.filtered_params[api]['type']}), "
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task=f"The predicted API takes {tmp_input_para} as input. However, there are still some parameters undefined in the query.", task_title="Enter Parameters: basic type",color="red") for callback in self.callbacks]
            self.indexxxx+=1
            self.run_select_params(user_input)
            return
        self.run_pipeline_after_entering_params(user_input)
        
    def run_select_params(self, user_input):
        if self.user_states == "select_params":
            self.selected_params = self.executor.makeup_for_missing_single_parameter(params = self.selected_params, param_name_to_update=self.last_param_name, user_input = user_input)
        [callback.on_tool_start() for callback in self.callbacks]
        [callback.on_tool_end() for callback in self.callbacks]
        if len(self.filtered_params)>1:
            self.last_param_name = list(self.filtered_params.keys())[0]
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task="Which value do you think is appropriate for the parameters '"+self.last_param_name+"'?", task_title="Enter Parameters: basic type",color="red") for callback in self.callbacks]
            self.indexxxx+=1
            self.user_states = "select_params"
            del self.filtered_params[self.last_param_name]
            return
        elif len(self.filtered_params)==1:
            self.last_param_name = list(self.filtered_params.keys())[0]
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task="Which value do you think is appropriate for the parameters '"+self.last_param_name+"'?", task_title="Enter Parameters: basic type",color="red") for callback in self.callbacks]
            self.indexxxx+=1
            self.user_states = "enter_params"
            del self.filtered_params[self.last_param_name]
        else:
            raise ValueError
    def split_params(self, selected_params, parameters_list):
        extracted_params = []
        for params in parameters_list:
            extracted = {}
            for param_name, param_info in params.items():
                if param_name in selected_params: #  and selected_params[param_name]["type"] == param_info["type"]
                    extracted[param_name] = selected_params[param_name]
                else: # because sometimes the sub API has different name but stands for same parameters, like adata/data
                    # Find a match based on type when the parameter is not in selected_params
                    for sel_name, sel_info in selected_params.items():
                        if (
                            sel_info["type"] == param_info["type"]
                            and sel_name not in extracted.values()
                        ):
                            extracted[param_name] = sel_info
                            break
            extracted_params.append(extracted)
        return extracted_params
    def extract_parameters(self, api_name_json, api_info):
        parameters_combined = []
        for api_name in api_name_json:
            details = api_info[api_name]
            parameters = details["Parameters"]
            api_params = {param_name: {"type": param_details["type"]} for param_name, param_details in parameters.items() if not param_details['optional']}
            combined_params = {}
            for param_name, param_info in api_params.items():
                if param_name not in combined_params:
                    combined_params[param_name] = param_info
            parameters_combined.append(combined_params)
        return parameters_combined

    def run_pipeline_after_entering_params(self, user_input):
        print('Now run pipeline after entering parameters')
        self.user_states = "initial"
        self.image_file_list = self.update_image_file_list()
        if self.none_at_value_params:
            self.selected_params = self.executor.makeup_for_missing_single_parameter(params = self.selected_params, param_name_to_update=self.last_param_name, user_input = user_input)
        if self.filtered_pathlike_params:
            # add 'tmp' 
            for key in self.filtered_pathlike_params:
                param_info = self.filtered_pathlike_params[key]
                self.selected_params[key] = {
                    "type": param_info["type"],
                    "value": "./tmp",
                    "valuefrom": 'userinput',
                    "optional": param_info["optional"],
                }
        print('==>selected_params:',self.selected_params)
        # split parameters according to multiple API, or class/method API
        parameters_list = self.extract_parameters(self.api_name_json, self.API_composite)
        extracted_params = self.split_params(self.selected_params, parameters_list)
        extracted_params_dict = {api_name: extracted_param for api_name, extracted_param in zip(self.api_name_json, extracted_params)}
        print('==>extracted_params_dict:',extracted_params_dict)
        api_params_list = []
        for idx, api_name in enumerate(self.api_name_json):
            if self.api_name_json[api_name]['type']!='class':
                class_selected_params = {}
                fake_class_api = '.'.join(api_name.split('.')[:-1])
                if fake_class_api in self.api_name_json and self.api_name_json[fake_class_api]['type']=='class':
                    class_selected_params = extracted_params_dict[fake_class_api]
                if 'inplace' in self.API_composite[api_name]['Parameters']:
                    extracted_params[idx]['inplace'] = {
                        "type": self.API_composite[api_name]['Parameters']['inplace']['type'],
                        "value": True,
                        "valuefrom": 'value',
                        "optional": True,
                    }
                api_params_list.append({"api_name":api_name, 
                                        "parameters":extracted_params[idx], 
                                        "return_type":self.API_composite[api_name]['Returns']['type'],
                                        "class_selected_params":class_selected_params})
        print('==>api_params_list:',api_params_list)
        execution_code = self.executor.generate_execution_code(api_params_list)
        print('==>execution_code:',execution_code)
        [callback.on_tool_start() for callback in self.callbacks]
        [callback.on_tool_end() for callback in self.callbacks]
        [callback.on_agent_action(block_id="code-"+str(self.indexxxx),task=execution_code,task_title="Executed code",) for callback in self.callbacks]
        self.indexxxx+=1
        execution_code_list = execution_code.split('\n')
        error_list = []
        self.plt_status = plt.get_fignums()
        self.buf = io.StringIO()
        sys.stdout = self.buf
        sys.stderr = self.buf
        for code in execution_code_list:
            error_list.append(self.executor.execute_api_call(code, "code"))
            if plt.get_fignums()!=self.plt_status:
                error_list.append(self.executor.execute_api_call("from inference.utils import save_plot_with_timestamp", "import"))
                error_list.append(self.executor.execute_api_call("save_plot_with_timestamp()", "code"))
                self.plt_status = plt.get_fignums()
            else:
                pass
        if self.executor.execute_code[-1]['success']=='True':
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            content = self.buf.getvalue()
            # if execute, visualize value
            vari = [i.strip() for i in code.split('(')[0].split('=')]
            print(f'-----code: {code}')
            print(f'-----vari: {vari}')
            tips_for_execution_success = True
            if len(vari)>1:
                if self.executor.variables[vari[0]]['value']:
                    print('if vari value is not None, return it')
                    [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task="We obtain a new " + str(self.executor.variables[vari[0]]['value']),task_title="Executed results [Success]",) for callback in self.callbacks]
                    self.indexxxx+=1
                else:
                    print('if vari value is not None, just return the success message')
                    [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task="Execute success. We didn't obtain new variable",task_title="Executed results [Success]",) for callback in self.callbacks]
                    self.indexxxx+=1
                if self.executor.variables[vari[0]]['type']=='AnnData':
                    print('if the new variable is of type AnnData, ')
                    if 'obs' in dir(self.executor.variables[vari[0]]['value']):
                        print('if obs in adata, visualize anndata.obs')
                        output_table = self.executor.variables[vari[0]]['value'].obs.head(5).to_csv(index=True, header=True, sep=',', lineterminator='\n')
                        # if exist \n in the last index, remove it
                        last_newline_index = output_table.rfind('\n')
                        if last_newline_index != -1:
                            output_table = output_table[:last_newline_index] + '' + output_table[last_newline_index + 1:]
                        else:
                            output_table = output_table
                        [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task="We visualize the first 5 rows of the table data",task_title="Executed results [Success]",tableData=output_table) for callback in self.callbacks]
                        self.indexxxx+=1
                    else:
                        pass
                try:
                    print('if exist table, visualize it')
                    output_table = self.executor.variables[vari[0]]['value'].head(5).to_csv(index=True, header=True, sep=',', lineterminator='\n')
                    last_newline_index = output_table.rfind('\n')
                    if last_newline_index != -1:
                        output_table = output_table[:last_newline_index] + '' + output_table[last_newline_index + 1:]
                    else:
                        output_table = output_table
                    [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task="We visualize the first 5 rows of the table data",task_title="Executed results [Success]",tableData=output_table) for callback in self.callbacks]
                    self.indexxxx+=1
                except:
                    pass
                tips_for_execution_success = False
            else:
                # 
                pass
            print('if generate image, visualize it')
            new_img_list = self.update_image_file_list()
            new_file_list = set(new_img_list)-set(self.image_file_list)
            if new_file_list:
                for new_img in new_file_list:
                    print('send image to frontend')
                    base64_image = convert_image_to_base64(os.path.join(self.image_folder,new_img))
                    if base64_image:
                        [callback.on_agent_action(block_id="log-" + str(self.indexxxx),task="We visualize the obtained figure",task_title="Executed results [Success]",imageData=base64_image) for callback in self.callbacks]
                        self.indexxxx += 1
                        tips_for_execution_success = False
            self.image_file_list = new_img_list
            if tips_for_execution_success:
                [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task=str(content),task_title="Executed results [Success]",) for callback in self.callbacks]
                self.indexxxx+=1
        else:
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task="Execution failed! "+"".join(error_list),task_title="Executed results [Fail]",) for callback in self.callbacks]
            self.indexxxx+=1
        print("Show current variables in namespace:" + str(self.executor.variables))
        new_str = []
        for i in self.executor.execute_code:
            new_str.append({"code":i['code'],"execution_results":i['success']})
        print("Currently all executed code:", new_str)
        self.executor.save_variables_to_json()
    def get_queue(self):
        while not self.queue.empty():
            yield self.queue.get()

model = Model()
@app.route('/stream', methods=['GET', 'POST'])
@cross_origin()
def stream():
    data = json.loads(request.data)
    print('='*10)
    print('get data:')
    for key, value in data.items():
        if key not in ['files']:
            print(f'{key}: {value}')
    print('='*10)
    user_input = data["text"]
    top_k = data["top_k"]
    Lib = data["Lib"]
    conversation_started = data["conversation_started"]
    raw_files = data["files"]
    try:
        new_lib_doc_url = data["new_lib_doc_url"]
        new_lib_github_url = data["new_lib_github_url"]
        api_html = data['api_html']
        lib_alias = data['lib_alias']
    except:
        # TODO: this is a bug, the default value will also be ""
        new_lib_doc_url = ""
        new_lib_github_url = ""
        api_html = ""
        lib_alias = ""

    # process uploaded files
    if len(raw_files)>0:
        print('length of files:',len(raw_files))
        for i in range(len(raw_files)):
            try:
                print(raw_files[i]['data'].split(",")[0])
                print(raw_files[i]['filename'])
            except:
                pass
        files = [save_decoded_file(raw_file) for raw_file in raw_files]
        files = [i for i in files if i] # remove None
    else:
        files = []
    
    global model
    def generate(model):
        print("Called generate")
        if model.inuse:
            return Response(json.dumps({
                "method_name": "error",
                "error": "Model in use"
            }), status=409, mimetype='application/json')
            return
        model.inuse = True
        if new_lib_doc_url:
            print('new_lib_doc_url is not none, start installing lib!')
            model.install_lib(new_lib_github_url, new_lib_doc_url, api_html, Lib, lib_alias)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print('start running pipeline!')
            future = executor.submit(model.run_pipeline, user_input, Lib, top_k, files, conversation_started)
            # keep waiting for the queue to be empty
            while True:
                if model.queue.empty():
                    if future.done():
                        print("Finished with future")
                        break
                    time.sleep(0.01)
                    continue
                else:
                    obj = model.queue.get()
                if obj["method_name"] == "unknown": continue
                if obj["method_name"] == "on_request_end":
                    yield json.dumps(obj)
                    break
                try:
                    yield json.dumps(obj) + "\n"
                except Exception as e:
                    model.inuse = False
                    print(obj)
                    print(e)
            try:
                future.result()
            except Exception as e:
                model.inuse = False
                print(e)
        model.inuse = False
        return
    return Response(stream_with_context(generate(model)))

@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    data = json.loads(request.data)
    api_key = data.get('apiKey')
    os.environ["OPENAI_API_KEY"] = api_key
    model.load_llm_model()
    return Response(json.dumps({"status": "success"}), status=200, mimetype='application/json')

def handle_keyboard_interrupt(signal, frame):
    global model
    exit(0)

signal.signal(signal.SIGINT, handle_keyboard_interrupt)

if __name__ == '__main__':
    app.run(use_reloader=False, host="0.0.0.0", debug=True, port=5000)