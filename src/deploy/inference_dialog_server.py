# Flask
from flask import Flask, Response, stream_with_context, request, send_file, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin
from deploy.ServerEventCallback import ServerEventCallback
from queue import Queue
app = Flask(__name__)
cors = CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
# standard lib
import argparse, json, signal, time, copy, base64, requests, importlib, inspect, ast, os, random, io, sys, pickle, shutil, subprocess, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime
from urllib.parse import urlparse
# Computational
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any
import multiprocessing
from sentence_transformers import SentenceTransformer, models
from inference.utils import predict_by_similarity, json_to_docstring
from tqdm import tqdm
from deploy.utils import change_format
from gpt.utils import get_all_api_json, correct_pred

import logging
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
os.makedirs(f"../logs", exist_ok=True)
log_filename = f"../logs/BioMANIA_log_{timestamp}.log"
logging.basicConfig(filename=log_filename, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
print("Logging setup complete.")

# device
import torch
if torch.cuda.is_available():
    gpu_index = 2
    torch.cuda.set_device(gpu_index)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("Current GPU Index: %s", torch.cuda.current_device())

# 
import concurrent.futures
from dotenv import load_dotenv
from string import punctuation
import warnings
warnings.filterwarnings("ignore")

# inference pipeline
from models.model import LLM_response, LLM_model
from configs.model_config import *
from inference.execution_UI import CodeExecutor
from inference.utils import find_similar_two_pairs, sentence_transformer_embed
from inference.retriever_finetune_inference import ToolRetriever
from deploy.utils import dataframe_to_markdown, convert_image_to_base64
from prompt.parameters import prepare_parameters_prompt
from prompt.summary import prepare_summary_prompt, prepare_summary_prompt_full

basic_types = ['str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set', 'List', 'Dict', 'Any', 'any', 'Path', 'path', 'Pathlike']
basic_types.extend(['_AvailShapes']) # extend for squidpy `shape` type

def generate_api_calling(api_name, api_details, returned_content_str):
    """
    Generates an API call and formats output based on provided API details and returned content string.
    """
    try:
        returned_content_str_new = returned_content_str.replace('null', 'None').replace('None', '"None"')
        returned_content = ast.literal_eval(returned_content_str_new)
        returned_content_dict = {item['param_name']: item['value'] for item in returned_content if (item['value'] not in ['None', None, 'NoneType']) and item['value']} # remove null parameters from prompt
    except Exception as e:
        returned_content_dict = {}
        print(f"Error parsing returned content: {e}")
    api_description = api_details["description"]
    parameters = api_details['Parameters']
    return_type = api_details['Returns']['type']
    parameters_dict = {}
    parameters_info_list = []
    for param_name, param_details in parameters.items():
        # only include required parameters and optional parameters found from response, and a patch for color in scanpy/squidpy pl APIs
        if (param_name in returned_content_dict) or (not param_details['optional']) or (param_name=='color' and (api_name.startswith('scanpy.pl') or api_name.startswith('squidpy.pl'))) or (param_name=='encodings' and (api_name.startswith('ehrapy.pp') or api_name.startswith('ehrapy.preprocessing'))) or (param_name=='encoded' and (api_name.startswith('ehrapy.'))):
            #print(param_name, param_name in returned_content_dict, not param_details['optional'])
            param_type = param_details['type']
            if param_type in [None, 'None', 'NoneType']:
                param_type = "Any"
            param_description = param_details['description']
            param_value = param_details['default']
            param_optional = param_details['optional']
            if returned_content_dict:
                if param_name in returned_content_dict:
                    param_value = returned_content_dict[param_name]
                    #if param_type is not None and ('str' in param_type or 'PathLike' in param_type):
                    #    if ('"' not in param_type and "'" not in param_type) and (param_value not in ['None', None]):
                    #        param_value = "'"+str(param_value)+"'"
            # added condition to differentiate between basic and non-basic types
            if any(item in param_type for item in basic_types):
                param_value = param_value if ((param_value not in [ 'None']) and param_value) else "@"
            # add some general rules for basic types.
            elif ('openable' in param_type) or ('filepath' in param_type) or ('URL' in param_type):
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

def infer(query, model, centroids, labels):
    # 240125 modified chitchat model
    user_query_vector = np.array([sentence_transformer_embed(model, query)])
    try:
        predicted_label = predict_by_similarity(user_query_vector, centroids, labels)
    except Exception as e:
        print(e)
    return predicted_label

if not os.path.exists('tmp'):
    os.mkdir('tmp')

def download_file_from_google_drive(file_id, save_dir="./tmp", output_path="output.zip"):
    import gdown
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, os.path.join(save_dir, output_path), quiet=False)
    subprocess.run(["unzip", os.path.join(save_dir, output_path), "-d", save_dir], check=True)

def download_data(url, save_dir="tmp"):
    # try uploading drive first
    try:
        save_path = download_file_from_google_drive(url)
        return save_path
    except:
        pass
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
        try:
            filename = download_data(decoded_data)
        except:
            print('==>Input URL Error!')
            pass
    return filename

def correct_bool_values(optional_param):
    """
    Convert boolean values from lowercase (true, false) to uppercase (True, False).

    :param optional_param: The dictionary containing the optional parameters.
    :return: The modified dictionary with corrected boolean values.
    """
    for key, value in optional_param.items():
        if 'optional' in value and isinstance(value['optional'], bool):
            value['optional'] = str(value['optional'])
        if 'optional_value' in value and isinstance(value['optional_value'], bool):
            value['optional_value'] = str(value['optional_value'])
    return optional_param

def convert_bool_values(optional_param):
    """
    Convert 'true' and 'false' in the 'optional' and 'optional_value' fields 
    to 'True' and 'False' respectively.

    :param optional_param: The dictionary containing the optional parameters.
    :return: The modified dictionary with converted boolean values.
    """
    for key, value in optional_param.items():
        if 'optional' in value and isinstance(value['optional'], str):
            value['optional'] = value['optional'].capitalize()
        if 'optional_value' in value and isinstance(value['optional_value'], str):
            value['optional_value'] = value['optional_value'].capitalize()
    return optional_param

class Model:
    def __init__(self):
        print("Initializing...")
        self.indexxxx = 1
        self.inuse = False
        # Define the arguments here...
        self.query_id = 0
        self.queue = Queue()
        self.callback = ServerEventCallback(self.queue)
        self.callbacks = [self.callback]
        self.occupied = False
        self.LIB = "scanpy"
        self.args_retrieval_model_path = f'./hugging_models/retriever_model_finetuned/{self.LIB}/assigned'
        self.args_top_k = 3
        self.session_id = ""
        #load_dotenv()
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-test')
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        self.initialize_executor()
        reset_result = self.reset_lib(self.LIB)
        if reset_result=='Fail':
            return
        self.last_user_states = ""
        self.user_states = "initial"
        self.parameters_info_list = None
        self.image_folder = "./tmp/images/"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder, exist_ok=True)
        if not os.path.exists("./tmp/states/"):
            os.makedirs("./tmp/states/", exist_ok=True)
        if not os.path.exists("./tmp/sessions/"):
            os.makedirs("./tmp/sessions/", exist_ok=True)
        self.image_file_list = []
        self.image_file_list = self.update_image_file_list()
        #with open(f'./data/standard_process/{self.LIB}/vectorizer.pkl', 'rb') as f:
        #    self.vectorizer = pickle.load(f)
        print('==>chitchat vectorizer loaded!')
        with open(f'./data/standard_process/{self.LIB}/centroids.pkl', 'rb') as f:
            self.centroids = pickle.load(f)
        print('==>chitchat vectorizer loaded!')
        self.retrieve_query_mode = "similar"
        self.all_apis, self.all_apis_json = get_all_api_json(f"./data/standard_process/{self.LIB}/API_init.json")
        print("Server ready")
    def load_bert_model(self, load_mode='unfinetuned_bert'):
        if load_mode=='unfinetuned_bert':
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        else:
            # load pretrained model
            self.bert_model = SentenceTransformer(f"./hugging_models/retriever_model_finetuned/{self.LIB}/assigned", device=device)

    def reset_lib(self, lib_name):
        #lib_name = lib_name.strip()
        print('================')
        print('==>Start reset the Lib %s!', lib_name)
        # reset and reload all the LIB-related data/models
        # suppose that all data&model are prepared already in their path
        try:
            # load the previous variables, execute_code, globals()
            self.args_retrieval_model_path = f'./hugging_models/retriever_model_finetuned/{lib_name}/assigned'
            self.ambiguous_pair = find_similar_two_pairs(lib_name)
            self.ambiguous_api = list(set(api for api_pair in self.ambiguous_pair for api in api_pair))
            self.load_data(f"./data/standard_process/{lib_name}/API_composite.json")
            print('==>loaded API json done')
            self.load_bert_model()
            print('==>loaded finetuned bert for chitchat')
            #self.load_composite_code(lib_name)
            #print('==>loaded API composite done')
            t1 = time.time()
            print('==>Start loading model!')
            self.load_llm_model()
            print('loaded llm model!')
            retrieval_model_path = self.args_retrieval_model_path
            parts = retrieval_model_path.split('/')
            if len(parts)>=3: # only work for path containing LIB, otherwise, please reenter the path in script
                if not parts[-1]:
                    parts = parts[:-1]
            parts[-2]= lib_name
            new_path = '/'.join(parts)
            retrieval_model_path = new_path
            print('load retrieval_model_path in: %s', retrieval_model_path)
            self.retriever = ToolRetriever(LIB=lib_name,corpus_tsv_path=f"./data/standard_process/{lib_name}/retriever_train_data/corpus.tsv", model_path=retrieval_model_path, add_base=False)
            print('loaded retriever!')
            #self.executor.execute_api_call(f"from data.standard_process.{self.LIB}.Composite_API import *", "import")
            self.executor.execute_api_call(f"import {lib_name}", "import")
            # pyteomics tutorial needs these import libs
            self.executor.execute_api_call(f"import os, gzip, numpy as np, matplotlib.pyplot as plt", "import")
            #self.executor.execute_api_call(f"from urllib.request import urlretrieve", "import")
            #self.executor.execute_api_call(f"from pyteomics import fasta, parser, mass, achrom, electrochem, auxiliary", "import")
            self.executor.execute_api_call(f"import numpy as np", "import")
            self.executor.execute_api_call(f"np.seterr(under='ignore')", "import")
            self.executor.execute_api_call(f"import warnings", "import")
            self.executor.execute_api_call(f"warnings.filterwarnings('ignore')", "import")
            self.all_apis, self.all_apis_json = get_all_api_json(f"./data/standard_process/{lib_name}/API_init.json")
            print('==>Successfully loading model!')
            print('loading model cost: %s s', str(time.time()-t1))
            reset_result = "Success"
            self.LIB = lib_name
        except Exception as e:
            print('at least one data or model is not ready, please install lib first!')
            print('Error: %s', e)
            reset_result = "Fail"
            [callback.on_tool_start() for callback in self.callbacks]
            [callback.on_tool_end() for callback in self.callbacks]
            [callback.on_agent_action(block_id="log-" + str(self.indexxxx), task=f"Something wrong with loading data and model! \n{e}",task_title="Setting error") for callback in self.callbacks]
            self.indexxxx+=1
        return reset_result
    def install_lib(self,lib_name, lib_alias, api_html=None, github_url=None, doc_url=None):
        self.install_lib_simple(lib_name, lib_alias, github_url, doc_url, api_html)
        #self.install_lib_full(lib_name, lib_alias, github_url, doc_url, api_html)

    def install_lib_simple(self,lib_name, lib_alias, api_html=None, github_url=None, doc_url=None):
        #from configs.model_config import get_all_variable_from_cheatsheet
        #info_json = get_all_variable_from_cheatsheet(lib_name)
        #API_HTML, TUTORIAL_GITHUB = [info_json[key] for key in ['API_HTML', 'TUTORIAL_GITHUB']]
        self.LIB = lib_name
        self.args_retrieval_model_path = f'./hugging_models/retriever_model_finetuned/{lib_name}/assigned'
        from configs.model_config import GITHUB_PATH, ANALYSIS_PATH, READTHEDOC_PATH
        #from configs.model_config import LIB, LIB_ALIAS, GITHUB_LINK, API_HTML
        from dataloader.utils.code_download_strategy import download_lib
        from dataloader.utils.other_download import download_readthedoc
        from dataloader.get_API_init_from_sourcecode import main_get_API_init
        from dataloader.get_API_full_from_unittest import merge_unittest_examples_into_API_init
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Downloading lib...",task_title="0") for callback in self.callbacks]
        os.makedirs(f"./data/standard_process/{self.LIB}/", exist_ok=True)
        #[callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="downloading materials...",task_title="13") for callback in self.callbacks]
        #self.indexxxx+=1
        if github_url: # use git install
            download_lib('git', self.LIB, github_url, lib_alias, GITHUB_PATH)
        else: # use pip install
            subprocess.run(['pip', 'install', f'{lib_alias}'])
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Lib downloaded...",task_title="0") for callback in self.callbacks]
        self.indexxxx+=1
        if doc_url and api_html:
            download_readthedoc(doc_url, api_html)
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Preparing API_init.json ...",task_title="26") for callback in self.callbacks]
        if api_html:
            api_path = os.path.normpath(os.path.join(READTHEDOC_PATH, api_html))
        else:
            api_path = None
        main_get_API_init(self.LIB,lib_alias,ANALYSIS_PATH,api_path)
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Finished API_init.json ...",task_title="26") for callback in self.callbacks]
        self.indexxxx+=1
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Preparing API_composite.json ...",task_title="39") for callback in self.callbacks]
        shutil.copy(f'./data/standard_process/{self.LIB}/API_init.json', f'./data/standard_process/{self.LIB}/API_composite.json')
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Finished API_composite.json ...",task_title="39") for callback in self.callbacks]
        self.indexxxx+=1
        ###########
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Preparing instruction generation API_inquiry.json ...",task_title="52") for callback in self.callbacks]
        self.indexxxx+=1
        command = [
            "python", "dataloader/preprocess_retriever_data.py",
            "--LIB", self.LIB
        ]
        print("Running command:", command)
        subprocess.Popen(command)
        ###########
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Copying chitchat model from multicorpus pretrained chitchat model ...",task_title="65") for callback in self.callbacks]
        shutil.copy(f'./data/standard_process/multicorpus/centroids.pkl', f'./data/standard_process/{self.LIB}/centroids.pkl')
        shutil.copy(f'./data/standard_process/multicorpus/vectorizer.pkl', f'./data/standard_process/{self.LIB}/vectorizer.pkl')
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Done preparing chitchat model ...",task_title="65") for callback in self.callbacks]
        self.indexxxx+=1
        ###########
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Copying retriever from multicorpus pretrained retriever model...",task_title="78") for callback in self.callbacks]
        self.indexxxx+=1
        subprocess.run(["mkdir", f"./hugging_models/retriever_model_finetuned/{self.LIB}"])
        shutil.copytree(f'./hugging_models/retriever_model_finetuned/multicorpus/assigned', f'./hugging_models/retriever_model_finetuned/{self.LIB}/assigned')
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Process done! Please restart the program for usage",task_title="100") for callback in self.callbacks]
        self.indexxxx+=1
        # TODO: need to add tutorial_github and tutorial_html_path
        cheatsheet_path = './configs/Lib_cheatsheet.json'
        with open(cheatsheet_path, 'r') as file:
            cheatsheet_data = json.load(file)
        new_lib_details = {self.LIB: 
            {
                "LIB": self.LIB, 
                "LIB_ALIAS":lib_alias,
                "API_HTML_PATH": api_html,
                "GITHUB_LINK": github_url,
                "READTHEDOC_LINK": doc_url,
                "TUTORIAL_HTML_PATH":None,
                "TUTORIAL_GITHUB":None
            }
        }
        cheatsheet_data.update(new_lib_details)
        with open(cheatsheet_path, 'w') as file:
            json.dump(cheatsheet_data, file, indent=4)

    def install_lib_full(self,lib_name, lib_alias, api_html=None, github_url=None, doc_url=None):
        #from configs.model_config import get_all_variable_from_cheatsheet
        #info_json = get_all_variable_from_cheatsheet(lib_name)
        #API_HTML, TUTORIAL_GITHUB = [info_json[key] for key in ['API_HTML', 'TUTORIAL_GITHUB']]
        self.LIB = lib_name
        self.args_retrieval_model_path = f'./hugging_models/retriever_model_finetuned/{lib_name}/assigned'
        from configs.model_config import GITHUB_PATH, ANALYSIS_PATH, READTHEDOC_PATH
        #from configs.model_config import LIB, LIB_ALIAS, GITHUB_LINK, API_HTML
        from dataloader.utils.code_download_strategy import download_lib
        from dataloader.utils.other_download import download_readthedoc
        from dataloader.get_API_init_from_sourcecode import main_get_API_init
        from dataloader.get_API_full_from_unittest import merge_unittest_examples_into_API_init
        
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Downloading lib...",task_title="0") for callback in self.callbacks]
        os.makedirs(f"./data/standard_process/{self.LIB}/", exist_ok=True)
        #[callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="downloading materials...",task_title="13") for callback in self.callbacks]
        #self.indexxxx+=1
        if github_url: # use git install
            download_lib('git', self.LIB, github_url, lib_alias, GITHUB_PATH)
        else: # use pip install
            subprocess.run(['pip', 'install', f'{lib_alias}'])
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Lib downloaded...",task_title="0") for callback in self.callbacks]
        self.indexxxx+=1
        
        if doc_url and api_html:
            download_readthedoc(doc_url, api_html)
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Preparing API_init.json ...",task_title="26") for callback in self.callbacks]
        if api_html:
            api_path = os.path.normpath(os.path.join(READTHEDOC_PATH, api_html))
        else:
            api_path = None
        main_get_API_init(self.LIB,lib_alias,ANALYSIS_PATH,api_path)
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Finished API_init.json ...",task_title="26") for callback in self.callbacks]
        self.indexxxx+=1
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Preparing API_composite.json ...",task_title="39") for callback in self.callbacks]
        # TODO: add API_composite
        #merge_unittest_examples_into_API_init(self.LIB, ANALYSIS_PATH, GITHUB_PATH)
        #from dataloader.get_API_composite_from_tutorial import main_get_API_composite
        #main_get_API_composite(ANALYSIS_PATH, self.LIB)
        shutil.copy(f'./data/standard_process/{self.LIB}/API_init.json', f'./data/standard_process/{self.LIB}/API_composite.json')
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Finished API_composite.json ...",task_title="39") for callback in self.callbacks]
        self.indexxxx+=1
        
        ###########
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Preparing instruction generation API_inquiry.json ...",task_title="52") for callback in self.callbacks]
        self.indexxxx+=1
        command = [
            "python", "dataloader/preprocess_retriever_data.py",
            "--LIB", self.LIB
        ]
        subprocess.run(command)
        ###########
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Preparing chitchat model ...",task_title="65") for callback in self.callbacks]
        command = [
            "python",
            "models/chitchat_classification.py",
            "--LIB", self.LIB,
        ]
        subprocess.run(command)
        base64_image = convert_image_to_base64(f"./plot/{self.LIB}/chitchat_test_tsne_modified.png")
        [callback.on_agent_action(block_id="transfer_" + str(self.indexxxx),task=base64_image,task_title="chitchat_train_tsne_modified.png",) for callback in self.callbacks]
        self.indexxxx+=1
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Done chitchat model ...",task_title="65") for callback in self.callbacks]
        self.indexxxx+=1
        ###########
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Preparing retriever...",task_title="78") for callback in self.callbacks]
        self.indexxxx+=1
        subprocess.run(["mkdir", f"./hugging_models/retriever_model_finetuned/{self.LIB}"])
        command = [
            "python",
            "models/train_retriever.py",
            "--data_path", f"./data/standard_process/{self.LIB}/retriever_train_data/",
            "--model_name", "all-MiniLM-L6-v2",
            "--output_path", f"./hugging_models/retriever_model_finetuned/{self.LIB}",
            "--num_epochs", "20",
            "--train_batch_size", "32",
            "--learning_rate", "1e-5",
            "--warmup_steps", "500",
            "--max_seq_length", "256",
            "--optimize_top_k", "3",
            "--plot_dir", f"./plot/{self.LIB}/retriever/"
            "--gpu '0'"
        ]
        subprocess.run(command)
        base64_image = convert_image_to_base64(f"./plot/{self.LIB}/retriever/ndcg_plot.png")
        [callback.on_agent_action(block_id="transfer_" + str(self.indexxxx),task=base64_image,task_title="ndcg_plot.png",) for callback in self.callbacks]
        self.indexxxx+=1
        [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="Process done! Please restart the program for usage",task_title="100") for callback in self.callbacks]
        self.indexxxx+=1
        # TODO: need to add tutorial_github and tutorial_html_path
        cheatsheet_path = './configs/Lib_cheatsheet.json'
        with open(cheatsheet_path, 'r') as file:
            cheatsheet_data = json.load(file)
        new_lib_details = {self.LIB: 
            {
                "LIB": self.LIB, 
                "LIB_ALIAS":lib_alias,
                "API_HTML_PATH": api_html,
                "GITHUB_LINK": github_url,
                "READTHEDOC_LINK": doc_url,
                "TUTORIAL_HTML_PATH":None,
                "TUTORIAL_GITHUB":None
            }
        }
        cheatsheet_data.update(new_lib_details)
        with open(cheatsheet_path, 'w') as file:
            json.dump(cheatsheet_data, file, indent=4)

    def update_image_file_list(self):
        image_file_list = [f for f in os.listdir(self.image_folder) if f.endswith(".webp")]
        return image_file_list
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
        retrieved_names = self.retriever.retrieving(query, top_k=self.args_top_k)
        return retrieved_names
    def initialize_executor(self):
        self.executor = CodeExecutor()
        self.executor.callbacks = self.callbacks
        self.executor.variables={}
        self.executor.execute_code=[]
        self.clear_globals_with_prefix('result_')
    def clear_globals_with_prefix(self, prefix):
        global_vars = list(globals().keys())
        for var in global_vars:
            if var.startswith(prefix):
                del globals()[var]
    def load_llm_model(self):
        self.llm, self.tokenizer = LLM_model()
    def load_data(self, API_file):
        # fix 231227, add API_base.json
        with open(API_file, 'r') as json_file:
            data = json.load(json_file)
        with open("./data/standard_process/base/API_composite.json", 'r') as json_file:
            base_data = json.load(json_file)
        self.API_composite = data
        self.API_composite.update(base_data)
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
        print('uploading files finished!')
        if verbose:
            self.indexxxx+=1
            [callback.on_agent_action(block_id="installation-" + str(self.indexxxx), task="uploading files finished!",task_title=str(int(100))) for callback in self.callbacks]
            self.indexxxx+=1
    def save_state(self):
        a = str(self.session_id)
        file_name = f"./tmp/states/{a}_state.pkl"
        state = {k: v for k, v in self.__dict__.copy().items() if self.executor.is_picklable(v) and k != 'executor'}
        with open(file_name, 'wb') as file:
            pickle.dump(state, file)
        print("State saved to %s", file_name)
    def load_state(self, session_id):
        a = str(session_id)
        file_name = f"./tmp/states/{a}_state.pkl"
        with open(file_name, 'rb') as file:
            state = pickle.load(file)
        self.__dict__.update(state)
        print("State loaded from %s", file_name)
    def run_pipeline(self, user_input, lib, top_k=3, files=[],conversation_started=True,session_id=""):
        self.indexxxx = 2
        #if session_id != self.session_id:
        if True:
            self.session_id = session_id
            try:
                self.load_state(session_id)
                a = str(self.session_id)
                file_name=f"./tmp/sessions/{a}_environment.pkl"
                self.executor.load_environment(file_name)
            except:
                print('no local session_id environment exist! start from scratch')
                self.initialize_executor()
                pass
        # only reset lib when changing lib
        if lib!=self.LIB:
            reset_result = self.reset_lib(lib)
            if reset_result=='Fail':
                print('Reset lib fail! Exit the dialog!')
                return 
            self.args_retrieval_model_path = f'./hugging_models/retriever_model_finetuned/{lib}/assigned'
            self.LIB = lib
        # only clear namespace when starting new conversations
        if conversation_started in ["True", True]:
            print('==>new conversation_started!')
            self.user_states="initial"
            self.initialize_executor()
            self.executor.variables={}
            self.executor.execute_code=[]
            for var_name in list(globals()):
                if var_name.startswith('result_') or (var_name.endswith('_instance')):
                    del globals()[var_name]
            for var_name in list(locals()):
                if var_name.startswith('result_') or (var_name.endswith('_instance')):
                    del locals()[var_name]
        else:
            print('==>old conversation_continued!')
        if self.user_states == "initial":
            print('start initial!')
            while not self.queue.empty():
                self.queue.get()
            self.loading_data(files)
            print('loading data finished')
            self.query_id += 1
            self.user_query = user_input
            predicted_source = infer(self.user_query, self.bert_model, self.centroids, ['chitchat-data', 'topical-chat', 'api-query'])
            print(f'----query inferred as %s----', predicted_source)
            if predicted_source!='api-query':
                [callback.on_tool_start() for callback in self.callbacks]
                [callback.on_tool_end() for callback in self.callbacks]
                response, _ = LLM_response(self.llm, self.tokenizer, user_input, history=[], kwargs={})  # llm
                [callback.on_agent_action(block_id="log-" + str(self.indexxxx), task=response,task_title="Non API chitchat        ") for callback in self.callbacks]
                self.indexxxx+=1
                return
            else:
                pass
            retrieved_names = self.retrieve_names(user_input)
            print("retrieved_names: %s", retrieved_names)
            # produce prompt
            if self.retrieve_query_mode=='similar':
                instruction_shot_example = self.retriever.retrieve_similar_queries(user_input, shot_k=5)
            else:
                sampled_shuffled = random.sample(self.retriever.shuffled_data, 5)
                instruction_shot_example = "".join(["\nInstruction: " + ex['query'] + "\nFunction: " + ex['gold'] for ex in sampled_shuffled])
                similar_queries = ""
                shot_k=5 # 5 examples
                idx = 0
                for iii in sampled_shuffled:
                    instruction = iii['query']
                    tmp_retrieved_api_list = self.retriever.retrieving(instruction, top_k=top_k)
                    # ensure the order won't affect performance
                    tmp_retrieved_api_list = random.sample(tmp_retrieved_api_list, len(tmp_retrieved_api_list))
                    # ensure the example is correct
                    if iii['gold'] in tmp_retrieved_api_list:
                        if idx<shot_k:
                            idx+=1
                            # only retain shot_k number of sampled_shuffled
                            tmp_str = "Instruction: " + instruction + "\nFunction: [" + iii['gold'] + "]"
                            new_function_candidates = [f"{i}:{api}, description: "+self.all_apis_json[api].replace('\n',' ') for i, api in enumerate(tmp_retrieved_api_list)]
                            similar_queries += "function candidates:\n" + "\n".join(new_function_candidates) + '\n' + tmp_str + "\n---\n"
                instruction_shot_example = similar_queries
            # 240315: substitute prompt
            from gpt.utils import get_retrieved_prompt, get_nonretrieved_prompt
            api_predict_init_prompt = get_retrieved_prompt()
            retrieved_apis_prepare = ""
            for idx, api in enumerate(retrieved_names):
                retrieved_apis_prepare+=f"{idx}:" + api+", description: "+self.all_apis_json[api].replace('\n',' ')+"\n"
            api_predict_prompt = api_predict_init_prompt.format(query=user_input, retrieved_apis=retrieved_apis_prepare, similar_queries=instruction_shot_example)
            success = False
            for attempt in range(3):
                try:
                    response, _ = LLM_response(self.llm, self.tokenizer, api_predict_prompt, history=[], kwargs={})  # llm
                    print(f'==>Ask GPT: %s\n==>GPT response: %s', api_predict_prompt, response)
                    # hack for if GPT answers this or that
                    """response = response.split(',')[0].split("(")[0].split(' or ')[0]
                    response = response.replace('{','').replace('}','').replace('"','').replace("'",'')
                    response = response.split(':')[0]# for robustness, sometimes gpt will return api:description"""
                    response = correct_pred(response, self.LIB)
                    response = response.strip()
                    print('self.all_apis_json keys: ', self.all_apis_json.keys())
                    print('response in self.all_apis_json: ', response in self.all_apis_json)
                    self.all_apis_json[response]
                    self.predicted_api_name = response 
                    success = True
                    break
                except Exception as e:
                    print('error during api prediction: ', e)
                    pass
                    #return 
            if not success:
                [callback.on_tool_start() for callback in self.callbacks]
                [callback.on_tool_end() for callback in self.callbacks]
                [callback.on_agent_action(block_id="log-" + str(self.indexxxx),task=f"GPT can not return valid API name prediction, please redesign your prompt.",task_title="GPT predict Error",) for callback in self.callbacks]
                self.indexxxx += 1
                return
            print(f'length of ambiguous api list: {len(self.ambiguous_api)}')
            # if the predicted API is in ambiguous API list, then show those API and select one from them
            if self.predicted_api_name in self.ambiguous_api:
                filtered_pairs = [api_pair for api_pair in self.ambiguous_pair if self.predicted_api_name in api_pair]
                self.filtered_api = list(set(api for api_pair in filtered_pairs for api in api_pair))
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
                    self.last_user_states = self.user_states
                    self.user_states = "ambiguous_mode"
                    idx_api+=1
                [callback.on_agent_action(block_id="log-" + str(self.indexxxx), task=next_str,task_title=f"Can you confirm which of the following {len(self.filtered_api)} candidates        ") for callback in self.callbacks]
                self.indexxxx += 1
                self.save_state()
            else:
                self.last_user_states = self.user_states
                self.user_states = "after_API_selection"
                self.run_pipeline_after_fixing_API_selection(user_input)
        elif self.user_states =="run_pipeline_after_doublechecking_API_selection":
            self.run_pipeline_after_doublechecking_API_selection(user_input)
        elif self.user_states == "ambiguous_mode":
            ans = self.run_pipeline_after_ambiguous(user_input)
            if ans in ['break']:
                return
            self.run_pipeline_after_fixing_API_selection(user_input)
        elif self.user_states == "run_select_special_params":
            self.run_select_special_params(user_input)
        elif self.user_states == "run_pipeline_after_select_special_params":
            self.run_pipeline_after_select_special_params(user_input)
        elif self.user_states == "run_select_basic_params":
            self.run_select_basic_params(user_input)
        elif self.user_states == "run_pipeline_after_entering_params":
            self.run_pipeline_after_entering_params(user_input)
        elif self.user_states == "run_pipeline_after_doublechecking_execution_code":
            self.run_pipeline_after_doublechecking_execution_code(user_input)
    def run_pipeline_after_ambiguous(self,user_input):
        print('==>run_pipeline_after_ambiguous')
        user_input = user_input.strip()
        [callback.on_tool_start() for callback in self.callbacks]
        [callback.on_tool_end() for callback in self.callbacks]
        try:
            int(user_input)
        except:
            [callback.on_agent_action(block_id="log-" + str(self.indexxxx), task="Error: the input is not a number.\nPlease re-enter the index", task_title="Index Error        ") for callback in self.callbacks]
            self.indexxxx += 1
            self.last_user_states = self.user_states
            self.user_states = "ambiguous_mode"
            return 'break'
        try:
            self.filtered_api[int(user_input)-1]
        except:
            [callback.on_agent_action(block_id="log-" + str(self.indexxxx), task="Error: the input index exceed the maximum length of ambiguous API list\nPlease re-enter the index",task_title="Index Error        ") for callback in self.callbacks]
            self.indexxxx += 1
            self.last_user_states = self.user_states
            self.user_states = "ambiguous_mode"
            return 'break'
        self.last_user_states = self.user_states
        self.user_states = "after_API_selection"
        self.predicted_api_name = self.filtered_api[int(user_input)-1]
        self.save_state()
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
        print('==>run_pipeline_after_fixing_API_selection')
        # check if composite API/class method API, return the relevant APIs
        self.relevant_api_list = self.process_api_info(self.API_composite, self.predicted_api_name) # only contains predicted API
        print('self.relevant_api_list', self.relevant_api_list)
        self.api_name_json = self.check_and_insert_class_apis(self.API_composite, self.relevant_api_list)# also contains class API
        print('self.api_name_json', self.api_name_json)
        self.last_user_states = self.user_states
        self.user_states = "initial"
        api_description = self.API_composite[self.predicted_api_name]['description']
        # summary task
        summary_prompt = prepare_summary_prompt(user_input, self.predicted_api_name, api_description, self.API_composite[self.predicted_api_name]['Parameters'],self.API_composite[self.predicted_api_name]['Returns'])
        response, _ = LLM_response(self.llm, self.tokenizer, summary_prompt, history=[], kwargs={})  
        
        [callback.on_tool_start() for callback in self.callbacks]
        [callback.on_tool_end() for callback in self.callbacks]
        [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task=response,task_title=f"Predicted API: {self.predicted_api_name}        ",) for callback in self.callbacks]
        self.indexxxx+=1
        [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task="Could you confirm whether this API should be called? Please enter y/n.",task_title=f"Double Check        ",) for callback in self.callbacks]
        self.indexxxx+=1
        self.last_user_states = self.user_states
        self.user_states = "run_pipeline_after_doublechecking_API_selection"
        self.save_state()
    
    def run_pipeline_after_doublechecking_API_selection(self, user_input):
        print('==>run_pipeline_after_doublechecking_API_selection')
        user_input = str(user_input)
        if user_input in ['y', 'n']:
            if user_input == 'n':
                print('input n')
                self.last_user_states = self.user_states
                self.user_states = "initial"
                [callback.on_tool_start() for callback in self.callbacks]
                [callback.on_tool_end() for callback in self.callbacks]
                [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task="We will start another round. Could you re-enter your inquiry?",task_title=f"Start another round        ",) for callback in self.callbacks]
                self.indexxxx+=1
                self.save_state()
                return
            else:
                print('input y')
                pass
        else:
            print('input not y or n')
            [callback.on_tool_start() for callback in self.callbacks]
            [callback.on_tool_end() for callback in self.callbacks]
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task="The input was not y or n, please enter the correct value.",task_title=f"Index Error",) for callback in self.callbacks]
            self.indexxxx+=1
            self.save_state()
            # user_states didn't change
            return
        # print("==>Need to collect all parameters for a composite API")
        combined_params = {}
        # if the class API has already been initialized, then skip it
        for api in self.api_name_json:
            api_parts = api.split('.')
            maybe_class_name = api_parts[-1]
            maybe_instance_name = maybe_class_name.lower() + "_instance"
            if (maybe_instance_name in self.executor.variables) and (self.API_composite[api]['api_type']=='class'):
                # print(f'skip parameters for {maybe_instance_name}')
                continue
            else:
                pass
            combined_params.update(self.API_composite[api]['Parameters'])
        parameters_name_list = [key for key, value in combined_params.items() if (key not in ['path', "Path"])] 
        #parameters_name_list = [key for key, value in combined_params.items() if (not value['optional']) and (key not in ['path', "Path"])] # 
        api_parameters_information = change_format(combined_params, parameters_name_list)
        # turn None to All
        api_parameters_information = [
            {
                'name': param['name'],
                'type': 'All' if param['type'] in [None, 'null', 'None', 'NoneType'] else param['type'],
                'description': param['description'],
                'default_value': param['default_value']
            }
            for param in api_parameters_information
        ]
        #filter out special type parameters, do not infer them using gpt
        api_parameters_information = [param for param in api_parameters_information if any(basic_type in param['type'] for basic_type in basic_types)]
        parameters_name_list = [param_info['name'] for param_info in api_parameters_information]
        apis_description = ""
        apis_name = ""
        for idx,api_name_tmp_list in enumerate(self.relevant_api_list):
            if len(self.relevant_api_list)>1:
                api_name_tmp = list(api_name_tmp_list.keys())[0]
                apis_name+=f"{idx}:{api_name_tmp}"
                apis_description+=f"{idx}:{self.API_composite[api_name_tmp]['description']}."
            else:
                api_name_tmp = list(api_name_tmp_list.keys())[0]
                apis_name+=f"{api_name_tmp}"
                apis_description+=f"{self.API_composite[api_name_tmp]['description']}."
        api_docstring = json_to_docstring(apis_name, apis_description, api_parameters_information)
        parameters_prompt = prepare_parameters_prompt(self.user_query, api_docstring, parameters_name_list)
        if len(parameters_name_list)==0:
            # if there is no required parameters, skip using gpt
            response = "[]"
        else:
            success = False
            for attempt in range(3):
                try:
                    response, _ = LLM_response(self.llm, self.tokenizer, parameters_prompt, history=[], kwargs={})  
                    print(f'==>Asking GPT: %s, ==>GPT response: %s', parameters_prompt, response)
                    returned_content_str_new = response.replace('null', 'None').replace('None', '"None"')
                    try:
                        returned_content = ast.literal_eval(returned_content_str_new)
                        success = True
                        break
                    except:
                        try:
                            returned_content = json.loads(returned_content_str_new)
                            success = True
                            break
                        except:
                            pass
                except Exception as e:
                    pass
                    #return # 231130 fix 
            print('success or not: ', success)
            if not success:
                [callback.on_agent_action(block_id="log-" + str(self.indexxxx),task=f"GPT can not return valid parameters prediction, please redesign prompt in backend if you want to predict parameters. We will skip parameters prediction currently",task_title="GPT predict Error",) for callback in self.callbacks]
                self.indexxxx += 1
                response = "{}"
                print("GPT can not return valid parameters prediction, please redesign prompt in backend if you want to predict parameters. We will skip parameters prediction currently")
        # generate api_calling
        self.predicted_api_name, api_calling, self.parameters_info_list = generate_api_calling(self.predicted_api_name, self.API_composite[self.predicted_api_name], response)
        print('finished generate api calling')
        if len(self.api_name_json)> len(self.relevant_api_list):
            #assume_class_API = list(set(list(self.api_name_json.keys()))-set(self.relevant_api_list))[0]
            assume_class_API = '.'.join(self.predicted_api_name.split('.')[:-1])
            tmp_class_predicted_api_name, tmp_class_api_calling, tmp_class_parameters_info_list = generate_api_calling(assume_class_API, self.API_composite[assume_class_API], response)
            fix_update = True
            for api in self.api_name_json:
                api_parts = api.split('.')
                maybe_class_name = api_parts[-1]
                maybe_instance_name = maybe_class_name.lower() + "_instance"
                if (maybe_instance_name in self.executor.variables) and (self.API_composite[api]['api_type']=='class'):
                    fix_update = False
                else:
                    pass
            if fix_update:
                self.parameters_info_list['parameters'].update(tmp_class_parameters_info_list['parameters'])
        #print('After GPT predicting parameters, now the produced API calling is : %s', api_calling)
        ####### infer parameters
        # $ param
        self.selected_params = self.executor.select_parameters(self.parameters_info_list['parameters'])
        print("Automatically selected params for $, after selection the parameters are: %s", json.dumps(self.selected_params))
        # $ param if not fulfilled
        none_dollar_value_params = [param_name for param_name, param_info in self.selected_params.items() if param_info["value"] in ['$']]
        print(f'none_dollar_value_params: %s', json.dumps(none_dollar_value_params))
        if none_dollar_value_params:
            print(self.user_states)
            [callback.on_tool_start() for callback in self.callbacks]
            [callback.on_tool_end() for callback in self.callbacks]
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task="However, there are still some parameters with special type undefined. Please start from uploading data, or check your parameter type in json files.",task_title="Missing Parameters: special type") for callback in self.callbacks]
            self.indexxxx+=1
            self.last_user_states = self.user_states
            self.user_states = "initial"
            self.save_state()
            return
        # $ param if multiple choice
        multiple_dollar_value_params = [param_name for param_name, param_info in self.selected_params.items() if ('list' in str(type(param_info["value"]))) and (len(param_info["value"])>1)]
        self.filtered_params = {key: value for key, value in self.parameters_info_list['parameters'].items() if (key in multiple_dollar_value_params)}
        if multiple_dollar_value_params:
            print('==>There exist multiple choice for a special type parameters, start selecting parameters')
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task=f"There are many variables match the expected type. Please determine which one to choose",task_title="Choosing Parameters: special type") for callback in self.callbacks]
            self.indexxxx+=1
            tmp_input_para = ""
            for idx, api in enumerate(self.filtered_params):
                if idx!=0:
                    tmp_input_para+=" and "
                tmp_input_para+="'"+self.filtered_params[api]['description']+ "'"
                tmp_input_para+=f"('{api}': {self.filtered_params[api]['type']}), "
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task=f"The predicted API takes {tmp_input_para} as input. However, there are still some parameters undefined in the query.", task_title="Enter Parameters: special type",color="red") for callback in self.callbacks]
            self.indexxxx+=1
            self.last_user_states = self.user_states
            self.user_states = "run_select_special_params"
            self.run_select_special_params(user_input)
            self.save_state()
            return
        self.run_pipeline_after_select_special_params(user_input)

    def get_success_code_with_val(self, val):
        for i in self.executor.execute_code:
            if i['success']=='True' and val in i['code']:
                return i['code']
        [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task="Can not find the executed code corresponding to the expected parameters", task_title="Error Enter Parameters: special type",color="red") for callback in self.callbacks]
        print('Can not find the executed code corresponding to the expected parameters')
        #raise ValueError
    def run_select_special_params(self, user_input):
        print('==>run_select_special_params')
        if self.last_user_states == "run_select_special_params":
            self.selected_params = self.executor.makeup_for_missing_single_parameter_type_special(params = self.selected_params, param_name_to_update=self.last_param_name, user_input = user_input)
        [callback.on_tool_start() for callback in self.callbacks]
        [callback.on_tool_end() for callback in self.callbacks]
        print(f'self.filtered_params: %s', json.dumps(self.filtered_params))
        if len(self.filtered_params)>1:
            self.last_param_name = list(self.filtered_params.keys())[0]
            candidate_text = ""
            for val in self.selected_params[self.last_param_name]["value"]:
                get_val_code = self.get_success_code_with_val(val)
                candidate_text+=f'{val}: {get_val_code}\n'
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task=f"Which value do you think is appropriate for the parameters '{self.last_param_name}'? We find some candidates:\n {candidate_text}. ", task_title="Enter Parameters: special type",color="red") for callback in self.callbacks]
            self.indexxxx+=1
            self.last_user_states = self.user_states
            self.user_states = "run_select_special_params"
            del self.filtered_params[self.last_param_name]
            print(f'self.filtered_params: %s', json.dumps(self.filtered_params))
            self.save_state()
            return
        elif len(self.filtered_params)==1:
            self.last_param_name = list(self.filtered_params.keys())[0]
            candidate_text = ""
            for val in self.selected_params[self.last_param_name]["value"]:
                get_val_code = self.get_success_code_with_val(val)
                candidate_text+=f'{val}: {get_val_code}\n'
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task=f"Which value do you think is appropriate for the parameters '{self.last_param_name}'? We find some candidates \n {candidate_text}. ", task_title="Enter Parameters: special type",color="red") for callback in self.callbacks]
            self.indexxxx+=1
            self.last_user_states = self.user_states
            self.user_states = "run_pipeline_after_select_special_params"
            del self.filtered_params[self.last_param_name]
            print(f'self.filtered_params: %s', json.dumps(self.filtered_params))
            self.save_state()
        else:
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task="The parameters candidate list is empty", task_title="Error Enter Parameters: basic type",color="red") for callback in self.callbacks]
            self.save_state()
            raise ValueError

    def run_pipeline_after_select_special_params(self,user_input):
        if self.last_user_states == "run_select_special_params":
            self.selected_params = self.executor.makeup_for_missing_single_parameter_type_special(params = self.selected_params, param_name_to_update=self.last_param_name, user_input = user_input)
        # @ param
        print('starting entering basic params')
        none_at_value_params = [param_name for param_name, param_info in self.selected_params.items() if (param_info["value"] in ['@']) and (param_name not in ['path','Path'])]
        self.filtered_params = {key: value for key, value in self.parameters_info_list['parameters'].items() if (value["value"] in ['@']) and (key not in ['path','Path'])}
        self.filtered_pathlike_params = {}
        self.filtered_pathlike_params = {key: value for key, value in self.parameters_info_list['parameters'].items() if (value["value"] in ['@']) and (key in ['path','Path'])}
        # TODO: add condition later: if uploading data files, 
        # avoid asking Path params, assign it as './tmp'
        if none_at_value_params: # TODO: add type PathLike
            print('if exist non path, basic type parameters, start selecting parameters')
            tmp_input_para = ""
            for idx, api in enumerate(self.filtered_params):
                if idx!=0:
                    tmp_input_para+=" and "
                tmp_input_para+=self.filtered_params[api]['description']
                tmp_input_para+=f"('{api}': {self.filtered_params[api]['type']}), "
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task=f"The predicted API takes {tmp_input_para} as input. However, there are still some parameters undefined in the query.", task_title="Enter Parameters: basic type",color="red") for callback in self.callbacks]
            self.indexxxx+=1
            self.user_states = "run_select_basic_params"
            self.run_select_basic_params(user_input)
            self.save_state()
            return
        self.run_pipeline_after_entering_params(user_input)
    
    def run_select_basic_params(self, user_input):
        print('==>run_select_basic_params')
        if self.last_user_states == "run_select_basic_params":
            self.selected_params = self.executor.makeup_for_missing_single_parameter(params = self.selected_params, param_name_to_update=self.last_param_name, user_input = user_input)
        [callback.on_tool_start() for callback in self.callbacks]
        [callback.on_tool_end() for callback in self.callbacks]
        print('self.filtered_params: %s', json.dumps(self.filtered_params))
        if len(self.filtered_params)>1:
            self.last_param_name = list(self.filtered_params.keys())[0]
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task="Which value do you think is appropriate for the parameters '"+self.last_param_name+"'?", task_title="Enter Parameters: basic type",color="red") for callback in self.callbacks]
            self.indexxxx+=1
            self.last_user_states = self.user_states
            self.user_states = "run_select_basic_params"
            del self.filtered_params[self.last_param_name]
            self.save_state()
            return
        elif len(self.filtered_params)==1:
            self.last_param_name = list(self.filtered_params.keys())[0]
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task="Which value do you think is appropriate for the parameters '"+self.last_param_name+"'?", task_title="Enter Parameters: basic type",color="red") for callback in self.callbacks]
            self.indexxxx+=1
            self.last_user_states = self.user_states
            self.user_states = "run_pipeline_after_entering_params"
            del self.filtered_params[self.last_param_name]
            self.save_state()
        else:
            # break out the pipeline
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx), task="The parameters candidate list is empty", task_title="Error Enter Parameters: basic type",color="red") for callback in self.callbacks]
            self.save_state()
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
    def hide_streams(self):
        self.stdout_orig = sys.stdout
        self.stderr_orig = sys.stderr
        self.buf1 = io.StringIO()
        self.buf2 = io.StringIO()
        sys.stdout = self.buf1
        sys.stderr = self.buf2
    def restore_streams(self):
        sys.stdout = self.stdout_orig
        sys.stderr = self.stderr_orig
    def extract_parameters(self, api_name_json, api_info):
        parameters_combined = []
        for api_name in api_name_json:
            details = api_info[api_name]
            parameters = details["Parameters"]
            api_params = {param_name: {"type": param_details["type"]} for param_name, param_details in parameters.items() if (not param_details['optional']) or (param_name=="color" and (("scanpy.pl" in api_name) or ("squidpy.pl" in api_name))) or (param_name=='encodings' and (api_name.startswith('ehrapy.pp') or api_name.startswith('ehrapy.preprocessing'))) or (param_name=='encoded' and (api_name.startswith('ehrapy.')))} # TODO: currently not use optional parameters!!!
            api_params.update({})
            combined_params = {}
            for param_name, param_info in api_params.items():
                if param_name not in combined_params:
                    combined_params[param_name] = param_info
            parameters_combined.append(combined_params)
        return parameters_combined

    def run_pipeline_after_entering_params(self, user_input):
        if self.last_user_states == "run_select_basic_params":
            self.selected_params = self.executor.makeup_for_missing_single_parameter(params = self.selected_params, param_name_to_update=self.last_param_name, user_input = user_input)
        print('==>run pipeline after entering parameters')
        self.last_user_states = self.user_states
        self.user_states = "initial"
        self.image_file_list = self.update_image_file_list()
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
        print('self.selected_params: %s', json.dumps(self.selected_params))
        # split parameters according to multiple API, or class/method API
        parameters_list = self.extract_parameters(self.api_name_json, self.API_composite)
        extracted_params = self.split_params(self.selected_params, parameters_list)
        print(f'==>self.api_name_json: {self.api_name_json}, parameters_list: ', parameters_list)
        print('==>extracted_params: %s', extracted_params)
        extracted_params_dict = {api_name: extracted_param for api_name, extracted_param in zip(self.api_name_json, extracted_params)}
        print('extracted_params_dict: ', extracted_params_dict)
        api_params_list = []
        for idx, api_name in enumerate(self.api_name_json):
            if True:
                #if self.api_name_json[api_name]['type']=='class': # !
                #print('==>assume not start with class API: %s', api_name)
                class_selected_params = {}
                fake_class_api = '.'.join(api_name.split('.')[:-1])
                if fake_class_api in self.api_name_json:
                    if self.api_name_json[fake_class_api]['type']=='class':
                        class_selected_params = extracted_params_dict[fake_class_api]
                # two patches for pandas type data / squidpy parameters
                if ('inplace' in self.API_composite[api_name]['Parameters']) and (api_name.startswith('scanpy') or api_name.startswith('squidpy')):
                    extracted_params[idx]['inplace'] = {
                        "type": self.API_composite[api_name]['Parameters']['inplace']['type'],
                        "value": True,
                        "valuefrom": 'value',
                        "optional": True,
                    }
                if 'shape' in self.API_composite[api_name]['Parameters'] and 'pl.spatial_scatter' in api_name:
                    extracted_params[idx]['shape'] = {
                        "type": self.API_composite[api_name]['Parameters']['shape']['type'],
                        "value": "None",
                        "valuefrom": 'value',
                        "optional": True,
                    }
                # don't include class API, just include class.attribute API
                if self.API_composite[api_name]['api_type']!='class':
                    # when using class.attribute API, only include the API's information.
                    api_params_list.append({"api_name":api_name, 
                    "parameters":extracted_params[idx], 
                    "return_type":self.API_composite[api_name]['Returns']['type'],
                    "class_selected_params":class_selected_params,
                    "api_type":self.API_composite[api_name]['api_type']})
                else: # ==`class`
                    if len(self.api_name_json)==1:
                        # When using class API, only include class API's
                        api_params_list.append({"api_name":api_name, 
                        "parameters":extracted_params[idx], 
                        "return_type":self.API_composite[api_name]['Returns']['type'],
                        "class_selected_params":extracted_params[idx],
                        "api_type":self.API_composite[api_name]['api_type']})
                    else:
                        pass
        print('==>api_params_list: %s', json.dumps(api_params_list))
        # add optional cards
        optional_param = {key: value for key, value in self.API_composite[api_name]['Parameters'].items() if value['optional']}
        print('==>optional_param: %s', json.dumps(optional_param))
        print('len(optional_param) %d', len(optional_param))
        [callback.on_tool_start() for callback in self.callbacks]
        [callback.on_tool_end() for callback in self.callbacks]
        if False: # TODO: if True, to debug the optional card showing
            if len(optional_param)>0:
                print('producing optional param card')
                [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task="Do you want to modify the optional parameters? You can leave it unchange if you don't want to modify the default value.",task_title="Optional cards",) for callback in self.callbacks]
                self.indexxxx+=1
                [callback.on_agent_action(block_id="optional-"+str(self.indexxxx),task=convert_bool_values(correct_bool_values(optional_param)),task_title="Optional cards",) for callback in self.callbacks]
                self.indexxxx+=1
            else:
                pass
        # TODO: real time adjusting execution_code according to optionalcard
        print('api_params_list:', api_params_list)
        self.execution_code = self.executor.generate_execution_code(api_params_list)
        print('==>execution_code: %s', self.execution_code)
        [callback.on_tool_start() for callback in self.callbacks]
        [callback.on_tool_end() for callback in self.callbacks]
        [callback.on_agent_action(block_id="code-"+str(self.indexxxx),task=self.execution_code,task_title="Executed code",) for callback in self.callbacks]
        self.indexxxx+=1
        # LLM response
        summary_prompt = prepare_summary_prompt_full(user_input, self.predicted_api_name, self.API_composite[self.predicted_api_name]['description'], self.API_composite[self.predicted_api_name]['Parameters'],self.API_composite[self.predicted_api_name]['Returns'], self.execution_code)
        response, _ = LLM_response(self.llm, self.tokenizer, summary_prompt, history=[], kwargs={})  
        [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task=response,task_title=f"Task summary before execution",) for callback in self.callbacks]
        self.indexxxx+=1
        [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task="Could you confirm whether this task is what you aimed for, and the code should be executed? Please enter y/n.\nIf you press n, then we will re-direct to the parameter input step",task_title=f"Double Check        ",) for callback in self.callbacks]
        self.indexxxx+=1
        self.last_user_states = self.user_states
        self.user_states = "run_pipeline_after_doublechecking_execution_code"
        self.save_state()
        
    def run_pipeline_after_doublechecking_execution_code(self, user_input):
        [callback.on_tool_start() for callback in self.callbacks]
        [callback.on_tool_end() for callback in self.callbacks]
        # if check, back to the last iteration and status
        if user_input in ['y', 'n']:
            if user_input == 'n':
                print('input n')
                self.last_user_states = self.user_states
                #self.user_states = "initial"
                self.user_states = "run_pipeline_after_doublechecking_API_selection" #TODO: check if exist issue
                [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task="We will redirect to the parameters input",task_title=f"Re-enter the parameters",) for callback in self.callbacks]
                self.indexxxx+=1
                self.save_state()
                self.run_pipeline_after_doublechecking_API_selection('y')
                return
            else:
                print('input y')
                pass
        else:
            print('input not y or n')
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task="The input was not y or n, please enter the correct value.",task_title=f"Index Error",) for callback in self.callbacks]
            self.indexxxx+=1
            self.save_state()
            # user_states didn't change
            return
        # else, continue
        execution_code_list = self.execution_code.split('\n')
        print('execute and obtain figures')
        self.plt_status = plt.get_fignums()
        temp_output_file = "./sub_process_execution.txt"
        process = multiprocessing.Process(target=self.run_pipeline_execution_code_list, args=(execution_code_list, temp_output_file))
        process.start()
        while process.is_alive():
            print('process is alive!')
            time.sleep(1)
            with open(temp_output_file, 'r') as file:
                accumulated_output = file.read() ######?
                print('accumulated_output', accumulated_output)
                [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task=accumulated_output,task_title="Executing results",) for callback in self.callbacks]
        self.indexxxx+=1
        with open("./tmp/tmp_output_run_pipeline_execution_code_list.txt", 'r') as file:
            output_str = file.read()
            result = json.loads(output_str)
        code = result['code']
        output_list = result['output_list']
        self.executor.load_environment("./tmp/tmp_output_run_pipeline_execution_code_variables.pkl")
        print('check:', code, output_list, self.executor.execute_code, self.executor.variables)
        
        if len(execution_code_list)>0:
            self.last_execute_code = self.get_last_execute_code(code)
        else:
            self.last_execute_code = {"code":"", 'success':"False"}
            print('Something wrong with generating code with new API!')
        print('self.executor.variables:')
        print(json.dumps(list(self.executor.variables.keys())))
        print('self.executor.execute_code:')
        print(json.dumps(self.executor.execute_code))
        try:
            content = '\n'.join(output_list)
        except:
            content = ""
        # show the new variable 
        if self.last_execute_code['success']=='True':
            # if execute, visualize value
            code = self.last_execute_code['code']
            vari = [i.strip() for i in code.split('(')[0].split('=')]
            print('-----code: %s', code)
            print('-----vari: %s', vari)
            tips_for_execution_success = True
            if len(vari)>1:
                #if self.executor.variables[vari[0]]['value'] is not None:
                if (vari[0] in self.executor.variables) and ((vari[0].startswith('result_')) or (vari[0].endswith('_instance'))):
                    print_val = vari[0]
                    print_value = self.executor.variables[print_val]['value']
                    print_type = self.executor.variables[print_val]['type']
                    #print('if vari value is not None, return it')
                    [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task="We obtain a new variable: " + str(print_value),task_title="Executed results [Success]",) for callback in self.callbacks]
                    self.indexxxx+=1
                    if print_type=='AnnData':
                        print('if the new variable is of type AnnData, ')
                        visual_attr_list = [i_tmp for i_tmp in list(dir(print_value)) if not i_tmp.startswith('_')]
                        #if len(visual_attr_list)>0:
                        if 'obs' in visual_attr_list:
                            visual_attr = 'obs'#visual_attr_list[0]
                            print('visualize %s attribute', visual_attr)
                            output_table = getattr(self.executor.variables[vari[0]]['value'], "obs", None).head(5).to_csv(index=True, header=True, sep=',', lineterminator='\n')
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
                else:
                    print('Something wrong with variables! success executed variables didnt contain targeted variable')
                tips_for_execution_success = False
            else:
                pass
            print('if generate image, visualize it')
            new_img_list = self.update_image_file_list()
            new_file_list = set(new_img_list)-set(self.image_file_list)
            if new_file_list:
                for new_img in new_file_list:
                    print('send image to frontend')
                    base64_image = convert_image_to_base64(os.path.join(self.image_folder,new_img))
                    if base64_image:
                        [callback.on_agent_action(block_id="log-" + str(self.indexxxx),task="We visualize the obtained figure. Try to zoom in or out the figure.",task_title="Executed results [Success]",imageData=base64_image) for callback in self.callbacks]
                        self.indexxxx += 1
                        tips_for_execution_success = False
            self.image_file_list = new_img_list
            if tips_for_execution_success: # if no output, no new variable, present the log
                [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task=str(content),task_title="Executed results [Success]",) for callback in self.callbacks]
                self.indexxxx+=1
        else:
            print(f'Execution Error: %s', content)
            [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task="\n".join(list(set(output_list))),task_title="Executed results [Fail]",) for callback in self.callbacks] # Execution failed! 
            self.indexxxx+=1
        file_name=f"./tmp/sessions/{str(self.session_id)}_environment.pkl"
        self.executor.save_environment(file_name)
        self.save_state()
        if self.last_execute_code['success']=='True':
            # split tuple variable into individual variables
            ans, new_code = self.executor.split_tuple_variable(self.last_execute_code) # This function verifies whether the new variable is a tuple.
            if ans:
                [callback.on_agent_action(block_id="code-"+str(self.indexxxx),task=new_code,task_title="Executed code",) for callback in self.callbacks]
                self.indexxxx+=1
                [callback.on_agent_action(block_id="log-"+str(self.indexxxx),task="Splitting the returned tuple variable into individual variables",task_title="Executed results [Success]",) for callback in self.callbacks]
                self.indexxxx+=1
            else:
                pass
        else:
            pass
        print("Show current variables in namespace:")
        print(json.dumps(list(self.executor.variables.keys())))
        new_str = []
        for i in self.executor.execute_code:
            new_str.append({"code":i['code'],"execution_results":i['success']})
        print("Currently all executed code: %s", json.dumps(new_str))
        filename = f"./tmp/sessions/{str(self.session_id)}_environment.pkl"
        self.last_user_states = self.user_states
        self.user_states = "initial"
        self.executor.save_environment(filename)
        self.save_state()
    def modify_code_add_tmp(self, code, add_tmp = "tmp"):
        """
        sometimes author make 'return' information wrong
        we want to make up for it automatically by adding `tmp`
        """
        if not code.strip().startswith("result_"):
            find_pos = code.find("(")
            equal_pos = code.find("=")
            if find_pos != -1:
                if equal_pos != -1 and equal_pos < find_pos:
                    return code, False
                elif (equal_pos != -1 and equal_pos > find_pos) or equal_pos == -1:
                    modified_code = add_tmp + " = " + code
                    return modified_code, True
        return code, False
    def run_pipeline_execution_code_list(self, execution_code_list, output_file):
        # initialize the text
        with open(output_file, 'w') as test_file:
            test_file.write("\n")
        #sys.stdout = open(output_file, 'a')
        output_list = []
        for code in execution_code_list:
            ori_code = code
            if 'import' in code:
                add_tmp = None
                pass
            else:
                code, add_tmp = self.modify_code_add_tmp(code) # add `tmp =`
            ans = self.executor.execute_api_call(code, "code", output_file=output_file)
            # process tmp variable, if not None, add it to the 
            if add_tmp:
                if ('tmp' in self.executor.variables):
                    self.executor.counter+=1
                    self.executor.variables['result_'+str(self.executor.counter+1)] = {
                        "type": self.executor.variables['tmp']['type'],
                        "value": self.executor.variables['tmp']['value']
                    }
                    code, _ = self.modify_code_add_tmp(ori_code, 'result_'+str(self.executor.counter+1)) # add `tmp =`
                    ans = self.executor.execute_api_call(code, "code", output_file=output_file)
            print('%s, %s', str(code), str(ans))
            if ans:
                output_list.append(ans)
            if plt.get_fignums()!=self.plt_status:
                output_list.append(self.executor.execute_api_call("from inference.utils import save_plot_with_timestamp", "import"))
                output_list.append(self.executor.execute_api_call("save_plot_with_timestamp()", "code"))
                self.plt_status = plt.get_fignums()
            else:
                pass
        #sys.stdout.close()
        result = json.dumps({'code': code, 'output_list': output_list})
        self.executor.save_environment("./tmp/tmp_output_run_pipeline_execution_code_variables.pkl")
        with open("./tmp/tmp_output_run_pipeline_execution_code_list.txt", 'w') as file:
            file.write(result)
    
    def get_queue(self):
        while not self.queue.empty():
            yield self.queue.get()
    def get_last_execute_code(self, code):
        for i in range(1, len(self.executor.execute_code)+1):
            if self.executor.execute_code[-i]['code']==code:
                return self.executor.execute_code[-i]
            else:
                pass
        print(f'Something wrong with getting execution status by code! Enter wrong code %s', code)

from queue import Queue
import threading
model = Model()
should_stop = threading.Event()

@app.route('/api/stop_generation', methods=['POST'])
def stop_generation():
    global should_stop
    should_stop.set()
    return Response(json.dumps({"status": "stopped"}), status=200, mimetype='application/json')

@app.route('/stream', methods=['GET', 'POST'])
@cross_origin()
def stream():
    data = json.loads(request.data)
    print('='*30)
    print('get data:')
    for key, value in data.items():
        if key not in ['files']:
            print('%s: %s', key, value)
    print('='*30)
    #user_input = data["text"]
    #top_k = data["top_k"]
    #Lib = data["Lib"]
    #conversation_started = data["conversation_started"]
    raw_files = data["files"]
    #session_id = data['session_id']
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
        print('length of files: %d',len(raw_files))
        for i in range(len(raw_files)):
            try:
                print(str(raw_files[i]['data'].split(",")[0]))
                print(str(raw_files[i]['filename']))
            except:
                pass
        files = [save_decoded_file(raw_file) for raw_file in raw_files]
        files = [i for i in files if i] # remove None
    else:
        files = []
    
    global model
    def generate(model):
        global should_stop
        print("Called generate")
        if model.inuse:
            return Response(json.dumps({
                "method_name": "error",
                "error": "Model in use"
            }), status=409, mimetype='application/json')
            return
        model.inuse = True
        """if lib_alias:
            print(lib_alias)
            print('new_lib_doc_url is not none, start installing lib!')
            print('new_lib_doc_url is not none, start installing lib!')
            model.install_lib(data["Lib"], lib_alias, api_html, new_lib_github_url, new_lib_doc_url)"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print('start running pipeline!')
            future = executor.submit(model.run_pipeline, data["text"], data["Lib"], data["top_k"], files, data["conversation_started"], data['session_id'])
            # keep waiting for the queue to be empty
            while True:
                if should_stop.is_set():
                    should_stop.clear()
                    model.inuse = False
                    executor.shutdown(wait=False)
                    yield json.dumps({"method_name": "generation_stopped"})
                    return
                time.sleep(0.1)
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
                    #print(obj)
                    #print(e)
            try:
                future.result()
            except Exception as e:
                model.inuse = False
                #print(e)
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

import os
"""GITHUB_CLIENT_ID = os.environ.get('GITHUB_CLIENT_ID')
GITHUB_CLIENT_SECRET = os.environ.get('GITHUB_CLIENT_SECRET')
"""

signal.signal(signal.SIGINT, handle_keyboard_interrupt)

if __name__ == '__main__':
    app.run(use_reloader=False, host="0.0.0.0", debug=True, port=5000)
