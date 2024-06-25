from queue import Queue
import json, time, importlib, inspect, ast, os, random, io, sys, pickle, shutil, subprocess, re
from sentence_transformers import SentenceTransformer
import multiprocessing
import numpy as np, matplotlib.pyplot as plt
from functools import lru_cache
import asyncio, aiofiles
import traceback

from ..deploy.ServerEventCallback import ServerEventCallback
from ..gpt.utils import get_all_api_json, correct_pred, load_json, save_json, get_retrieved_prompt
from ..inference.utils import json_to_docstring, find_similar_two_pairs
from ..models.model import LLM_response
from ..configs.model_config import *
from ..inference.execution_UI import find_matching_instance
from ..inference.execution_UI import CodeExecutor
from ..inference.retriever_finetune_inference import ToolRetriever
from ..prompt.promptgenerator import PromptFactory
from ..configs.Lib_cheatsheet import CHEATSHEET as LIB_CHEATSHEET
from ..deploy.utils import basic_types, generate_api_calling, download_file_from_google_drive, download_data, save_decoded_file, correct_bool_values, convert_bool_values, infer, dataframe_to_markdown, convert_image_to_base64, change_format, special_types, io_types, io_param_names
from ..models.dialog_classifier import Dialog_Gaussian_classification
from ..inference.param_count_acc import predict_parameters

def generate_function_signature(api_name, parameters_json):
    # Load parameters from JSON string if it's not already a dictionary
    if isinstance(parameters_json, str):
        parameters = json.loads(parameters_json)
    else:
        parameters = parameters_json
    # Start building the function signature
    signature = api_name + "("
    params = []
    for param_name, param_info in parameters.items():
        # Extract parameter details
        param_type = param_info['type']
        default = param_info['default']
        optional = param_info['optional']
        # Format the parameter string
        if optional and default is not None:
            param_str = f"{param_name}: {param_type} = {default}"
        else:
            param_str = f"{param_name}: {param_type}"
        # Append to the params list
        params.append(param_str)
    # Join all parameters with commas and close the function parenthesis
    signature += ", ".join(params) + ")"
    return signature

def color_text(text, color):
    color_codes = {
        'red': '\033[91m',
        'green': '\033[92m',
        'blue': '\033[94m',
        'orange': '\033[93m',
        'reset': '\033[0m'
    }
    return f"{color_codes.get(color, color_codes['reset'])}{text}{color_codes['reset']}"
def label_sentence(sentence, parameters_dict):
    import re
    
    colors = ['red'] # , 'purple', 'blue', 'green', 'orange'
    color_map = {}
    color_index = 0

    def get_color(term):
        nonlocal color_index
        if term not in color_map:
            color_map[term] = colors[color_index % len(colors)]
            color_index += 1
        return color_map[term]
    
    def replace_match(match):
        term = match.group(0)
        color = get_color(term)
        return f' <span style="color:{color}">{term}</span> '
    
    for key, value in parameters_dict.items():
        pattern_key = re.compile(r'\b' + re.escape(key) + r'\b')
        pattern_value = re.compile(r'\b' + re.escape(str(value)) + r'\b')
        sentence = re.sub(pattern_key, replace_match, sentence)
        sentence = re.sub(pattern_value, replace_match, sentence)
    
    return sentence

def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def extract_last_error_sentence(traceback_log):
    # Split the log into individual lines
    lines = traceback_log.strip().split('\n')
    # Initialize a variable to store the last error sentence
    last_error_sentence = ""
    # Iterate over the lines in reverse to find the last error sentence
    for line in reversed(lines):
        if "Error:" in line:
            last_error_sentence = line.strip()
            break
    return last_error_sentence

basic_types.append('Any')

class Model:
    def __init__(self, logger, device, model_llm_type="gpt-3.5-turbo-0125"): # llama3,  # gpt-4-turbo # 
        # IO
        self.image_folder = "./tmp/images/"
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs("./tmp", exist_ok=True)
        os.makedirs("./tmp/states/", exist_ok=True)
        os.makedirs("./tmp/sessions/", exist_ok=True)
        self.LIB = "scanpy"
        with open(f'./data/standard_process/{self.LIB}/centroids.pkl', 'rb') as f:
            self.centroids = pickle.load(f)
        self.user_query_list = []
        self.prompt_factory = PromptFactory()
        self.model_llm_type = model_llm_type
        self.logger = logger
        self.device=device
        self.indexxxx = 1
        self.inuse = False
        self.query_id = 0
        self.queue = Queue()
        self.callbacks = [ServerEventCallback(self.queue)]
        self.occupied = False
        self.args_retrieval_model_path = f'./hugging_models/retriever_model_finetuned/{self.LIB}/assigned'
        self.shot_k = 5 # 5 seed examples for api prediction
        self.retry_execution_limit = 3
        self.retry_execution_count = 0
        self.path_info_list = ['path','Path','PathLike']
        self.args_top_k = 3
        self.param_llm_retry = 1
        self.predict_api_llm_retry = 3
        self.enable_multi_task = True
        self.session_id = ""
        self.last_user_states = ""
        self.user_states = "run_pipeline"
        self.retrieve_query_mode = "nonsimilar"
        self.parameters_info_list = None
        self.initial_goal_description = ""
        self.new_task_planning = True # decide whether re-plan the task
        self.retry_modify_count=0
        self.loaded_files = False
        #load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY', 'sk-test')
        os.environ["GITHUB_TOKEN"] = os.getenv('GITHUB_TOKEN', '')
        self.initialize_executor()
        reset_result = self.reset_lib(self.LIB)
        if reset_result=='Fail':
            self.logger.error('Reset lib fail! Exit the dialog!')
            return
        self.image_file_list = []
        self.image_file_list = self.update_image_file_list()
        #self.get_all_api_json_cache(f"./data/standard_process/{self.LIB}/API_init.json", mode='single')
        self.all_apis, self.all_apis_json = get_all_api_json(f"./data/standard_process/{self.LIB}/API_init.json", mode='single')
        self.logger.info("Server ready")
        self.save_state_enviro()
    async def predict_all_params(self, api_name_tmp, boolean_params, literal_params, int_params, boolean_document, literal_document, int_document):
        predicted_params = {}
        if boolean_params:
            boolean_predicted, response1 = await predict_parameters(boolean_document, self.user_query, list(boolean_params.keys()), api_name_tmp, self.API_composite[api_name_tmp], 'boolean')
            try:
                boolean_predicted = json.loads(response1)
                predicted_params.update(boolean_predicted)
            except:
                boolean_predicted = {}
                self.logger.error('error for predicting boolean parameters: {}', response1)
        if int_params:
            int_predicted, response2 = await predict_parameters(int_document, self.user_query, list(int_params.keys()), api_name_tmp, self.API_composite[api_name_tmp], 'int')
            try:
                int_predicted = json.loads(response2)
                predicted_params.update(int_predicted)
            except:
                int_predicted = {}
                self.logger.error('error for predicting int parameters: {}', response2)
        if literal_params:
            literal_predicted, response3 = await predict_parameters(literal_document, self.user_query, list(literal_params.keys()), api_name_tmp, self.API_composite[api_name_tmp], 'literal')
            try:
                literal_predicted = json.loads(response3)
                predicted_params.update(literal_predicted)
            except:
                literal_predicted = {}
                self.logger.error('error for predicting literal parameters: {}', response3)
        return predicted_params
    #@lru_cache(maxsize=10)
    def get_all_api_json_cache(self,path,mode):
        self.all_apis, self.all_apis_json = get_all_api_json(path, mode)
    def load_multiple_corpus_in_namespace(self, ):
        #self.executor.execute_api_call(f"from data.standard_process.{self.LIB}.Composite_API import *", "import")
        # pyteomics tutorial needs these import libs
        self.executor.execute_api_call(f"import os, gzip, numpy as np, matplotlib.pyplot as plt", "import")
        #self.executor.execute_api_call(f"from urllib.request import urlretrieve", "import")
        #self.executor.execute_api_call(f"from pyteomics import fasta, parser, mass, achrom, electrochem, auxiliary", "import")
        self.executor.execute_api_call(f"import numpy as np", "import")
        self.executor.execute_api_call(f"np.seterr(under='ignore')", "import")
        self.executor.execute_api_call(f"import warnings", "import")
        self.executor.execute_api_call(f"warnings.filterwarnings('ignore')", "import")
    #@lru_cache(maxsize=10)
    def load_bert_model_cache(self, load_mode='unfinetuned_bert'):
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device) if load_mode=='unfinetuned_bert' else SentenceTransformer(f"./hugging_models/retriever_model_finetuned/{self.LIB}/assigned", device=self.device)
    def load_dialog_metrics(self,):
        self.dialog_p_threshold = 0.05
        self.dialog_classifer = Dialog_Gaussian_classification(LIB=self.LIB, threshold=self.dialog_p_threshold)
        self.dialog_classifer.load_mean_std()
        self.dialog_mean, self.dialog_std = self.dialog_classifer.mean, self.dialog_classifer.std
    def reset_lib(self, lib_name):
        #return asyncio.run(self.async_reset_lib(lib_name))
        return self.async_reset_lib(lib_name)
    def async_reset_lib(self, lib_name):
        self.initialize_tool()
        lib_name = lib_name.strip()
        self.logger.info("==>Start reset the Lib {}!", lib_name)
        # reset and reload all the LIB-related data/models
        # suppose that all data&model are prepared already in their path
        try:
            # load the previous variables, execute_code, globals()
            self.args_retrieval_model_path = f'./hugging_models/retriever_model_finetuned/{lib_name}/assigned'
            self.ambiguous_pair = find_similar_two_pairs(f"./data/standard_process/{lib_name}/API_init.json")
            self.ambiguous_api = list(set(api for api_pair in self.ambiguous_pair for api in api_pair))
            #self.load_composite_code_cache(lib_name)
            t1 = time.time()
            retrieval_model_path = self.args_retrieval_model_path
            parts = retrieval_model_path.split('/')
            if len(parts)>=3: # only work for path containing LIB, otherwise, please reenter the path in script
                if not parts[-1]:
                    parts = parts[:-1]
            parts[-2]= lib_name
            new_path = '/'.join(parts)
            retrieval_model_path = new_path
            """tasks = [
                lambda: self.load_data(f"./data/standard_process/{lib_name}/API_composite.json"),
                lambda: self.load_bert_model_cache(),
                lambda: self.load_retriever(lib_name, retrieval_model_path),
                lambda: self.load_multiple_corpus_in_namespace(),
                lambda: self.get_all_api_json_cache(f"./data/standard_process/{lib_name}/API_init.json", mode='single')
            ]
            await asyncio.gather(*[task() for task in tasks])"""
            self.load_data(f"./data/standard_process/{lib_name}/API_composite.json")
            self.load_bert_model_cache()
            self.load_retriever(lib_name, retrieval_model_path)
            self.load_multiple_corpus_in_namespace()
            #self.get_all_api_json_cache(f"./data/standard_process/{lib_name}/API_init.json", mode='single')
            self.all_apis, self.all_apis_json = get_all_api_json(f"./data/standard_process/{lib_name}/API_init.json", mode='single')

            with open(f'./data/standard_process/{lib_name}/centroids.pkl', 'rb') as f:
                self.centroids = pickle.load(f)
            self.executor.execute_api_call(f"import {lib_name}", "import"),
            self.logger.info("==>Successfully loading model! loading model cost: {} s", str(time.time()-t1))
            # load the dialog metrics
            if self.enable_multi_task:
                self.load_dialog_metrics()
            reset_result = "Success"
            self.LIB = lib_name
        except Exception as e:
            e = traceback.format_exc()
            self.logger.error("Error: {}", e)
            reset_result = "Fail"
            self.callback_func('log', f"Something wrong with loading data and model! \n{e}", "Setting error")
        return reset_result
    def load_retriever(self, lib_name, retrieval_model_path):
        self.retriever = ToolRetriever(LIB=lib_name,corpus_tsv_path=f"./data/standard_process/{lib_name}/retriever_train_data/corpus.tsv", model_path=retrieval_model_path, add_base=False)

    def install_lib(self,lib_name, lib_alias, api_html=None, github_url=None, doc_url=None):
        self.install_lib_simple(lib_name, lib_alias, github_url, doc_url, api_html)
        #self.install_lib_full(lib_name, lib_alias, github_url, doc_url, api_html)

    def install_lib_simple(self,lib_name, lib_alias, api_html=None, github_url=None, doc_url=None):
        #from configs.model_config import get_all_variable_from_cheatsheet
        #info_json = get_all_variable_from_cheatsheet(lib_name)
        #API_HTML, TUTORIAL_GITHUB = [info_json[key] for key in ['API_HTML', 'TUTORIAL_GITHUB']]
        self.LIB = lib_name
        self.args_retrieval_model_path = f'./hugging_models/retriever_model_finetuned/{lib_name}/assigned'
        from ..configs.model_config import GITHUB_PATH, ANALYSIS_PATH, READTHEDOC_PATH
        #from configs.model_config import LIB, LIB_ALIAS, GITHUB_LINK, API_HTML
        from ..dataloader.utils.code_download_strategy import download_lib
        from ..dataloader.utils.other_download import download_readthedoc
        from ..dataloader.get_API_init_from_sourcecode import main_get_API_init
        from ..dataloader.get_API_full_from_unittest import merge_unittest_examples_into_API_init
        self.callback_func('installation', "Downloading lib...", "0")
        os.makedirs(f"./data/standard_process/{self.LIB}/", exist_ok=True)
        #self.callback_func('installation', "downloading materials...", "13")
        if github_url: # use git install
            download_lib('git', self.LIB, github_url, lib_alias, GITHUB_PATH)
        else: # use pip install
            subprocess.run(['pip', 'install', f'{lib_alias}'])
        self.callback_func('installation', "Lib downloaded...", "0")
        if doc_url and api_html:
            download_readthedoc(doc_url, api_html)
        self.callback_func('installation', "Preparing API_init.json ...", "26")
        if api_html:
            api_path = os.path.normpath(os.path.join(READTHEDOC_PATH, api_html))
        else:
            api_path = None
        main_get_API_init(self.LIB,lib_alias,ANALYSIS_PATH,api_path)
        self.callback_func('installation', "Preparing API_composite.json ...", "39")
        shutil.copy(f'./data/standard_process/{self.LIB}/API_init.json', f'./data/standard_process/{self.LIB}/API_composite.json')
        self.callback_func('installation', "Preparing instruction generation API_inquiry.json ...", "52")
        command = [
            "python", "-m", "src.dataloader.preprocess_retriever_data",
            "--LIB", self.LIB, "--GPT_model", "gpt3.5"
        ]
        subprocess.Popen(command)
        ###########
        self.callback_func('installation', "Copying chitchat model from multicorpus pretrained chitchat model ...", "65")
        command = [
            "python", "-m", "src.models.chitchat_classification",
            "--LIB", self.LIB, "--ratio_1_to_3", 1.0, "--ratio_2_to_3", 1.0, "--embed_method", "st_untrained"
        ]
        subprocess.Popen(command)
        ###########
        self.callback_func('installation', "Copying retriever from multicorpus pretrained retriever model...", "78")
        subprocess.run(["mkdir", f"./hugging_models/retriever_model_finetuned/{self.LIB}"])
        shutil.copytree(f'./hugging_models/retriever_model_finetuned/multicorpus/assigned', f'./hugging_models/retriever_model_finetuned/{self.LIB}/assigned')
        self.callback_func('installation', "Process done! Please restart the program for usage", "100")
        # TODO: need to add tutorial_github and tutorial_html_path        
        cheatsheet_data = LIB_CHEATSHEET
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
        #cheatsheet_data.update(new_lib_details)
        # save_json(cheatsheet_path, cheatsheet_data)
        # TODO: need to save tutorial_github and tutorial_html_path to cheatsheet
    def save_state_enviro(self):
        self.executor.save_environment(f"./tmp/sessions/{str(self.session_id)}_environment.pkl")
        self.save_state()
    def update_image_file_list(self):
        return [f for f in os.listdir(self.image_folder) if f.endswith(".webp")]
    #@lru_cache(maxsize=10)
    def load_composite_code_cache(self, lib_name):
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
    def initialize_executor(self):
        self.executor = CodeExecutor(self.logger)
        self.executor.callbacks = self.callbacks
        self.clear_globals_with_prefix('result_')
    def clear_globals_with_prefix(self, prefix):
        global_vars = list(globals().keys())
        for var in global_vars:
            if var.startswith(prefix):
                del globals()[var]
    #@lru_cache(maxsize=10)
    def load_data(self, API_file):
        self.API_composite = load_json(API_file)
        self.API_composite.update(load_json("./data/standard_process/base/API_composite.json"))
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
    def initialize_tool(self):
        [callback.on_tool_start() for callback in self.callbacks]
        [callback.on_tool_end() for callback in self.callbacks]
    def callback_func(self, type_task, task, task_title, color="", tableData=None, imageData=None, enhance_indexxxx=True):
        block_id = type_task + "-" + str(self.indexxxx)
        for callback in self.callbacks:
            kwargs = {'block_id': block_id, 'task': task, 'task_title': task_title}
            if color:
                kwargs['color'] = color
            if tableData is not None:
                kwargs['tableData'] = tableData
            if imageData is not None:
                kwargs['imageData'] = imageData
            callback.on_agent_action(**kwargs)
        if enhance_indexxxx: # sometimes we want to deprecate it, when running something slowly
            self.indexxxx += 1
    async def load_single_file(self, file_path):
        file_extension = os.path.splitext(file_path)[1]
        loading_code = self.generate_file_loading_code(file_path, file_extension)
        self.executor.execute_api_call(loading_code, "code")
        return file_path
    async def loading_data_async(self, files, verbose=False):
        tasks = [self.load_single_file(file_path) for file_path in files]
        for ids, task in enumerate(asyncio.as_completed(tasks)):
            file_path = await task
            if verbose:
                self.callback_func('installation', f"uploading files...{ids+1}/{len(files)}", str(int((ids+1)/len(files)*100)))
        self.logger.info("loading data finished!")
        if verbose:
            self.callback_func('installation', "uploading files finished!", "100")
    def loading_data(self, files, verbose=False):
        asyncio.run(self.loading_data_async(files, verbose))
    def save_state(self):
        a = str(self.session_id)
        state = {k: v for k, v in self.__dict__.copy().items() if self.executor.is_picklable(v) and k != 'executor'}
        with open(f"./tmp/states/{a}_state.pkl", 'wb') as file:
            pickle.dump(state, file)
        self.logger.info("State saved to {}", f"./tmp/states/{a}_state.pkl")
    #@lru_cache(maxsize=10)
    def load_state(self, session_id):
        a = str(session_id)
        with open(f"./tmp/states/{a}_state.pkl", 'rb') as file:
            state = pickle.load(file)
        self.__dict__.update(state)
        self.logger.info("State loaded from {}", f"./tmp/states/{a}_state.pkl")
    def run_pipeline_without_files(self, user_input):
        self.initialize_tool()
        #self.logger.info('==> run_pipeline_without_files')
        # if check, back to the last iteration and status
        if user_input in ['y', 'n', 'Y', 'N']:
            if user_input in ['n', 'N']:
                self.update_user_state("run_pipeline")
                self.callback_func('log', "We will start another round. Could you re-enter your inquiry?", "Start another round")
                self.save_state_enviro()
                return
            else:
                self.update_user_state("run_pipeline")
                self.save_state_enviro()
                self.run_pipeline(self.user_query, self.LIB, files=[], conversation_started=False, session_id=self.session_id)
        else:
            self.callback_func('log', "The input was not y or n, please enter the correct value.", "Index Error")
            self.save_state_enviro()
            # user_states didn't change
            return
    def run_pipeline(self, user_input, lib, top_k=3, files=[],conversation_started=True,session_id=""):
        self.initialize_tool()
        self.indexxxx = 2
        self.session_id = session_id
        try:
            self.load_state(session_id)
            a = str(self.session_id)
            self.executor.load_environment(f"./tmp/sessions/{a}_environment.pkl")
        except Exception as e:
            e = traceback.format_exc()
            self.logger.error(e)
            self.initialize_executor()
            self.new_task_planning = True
            self.user_query_list = []
            pass
        # only reset lib when changing lib
        if lib!=self.LIB and lib!='GPT':
            reset_result = self.reset_lib(lib)
            if reset_result=='Fail':
                self.logger.error('Reset lib fail! Exit the dialog!')
                return 
            self.args_retrieval_model_path = f'./hugging_models/retriever_model_finetuned/{lib}/assigned'
            self.LIB = lib
        elif lib=='GPT' and self.last_user_states != "run_pipeline_asking_GPT":
            self.user_query = user_input
            self.update_user_state("run_pipeline_asking_GPT")
        # only clear namespace when starting new conversations
        if conversation_started in ["True", True]:
            self.loaded_files = False
            if len(files)>0:
                self.loaded_files = True
            else:
                pass
            self.new_task_planning = True
            self.user_query_list = []
            #self.logger.info('==>new conversation_started!')
            self.user_states="run_pipeline"
            self.initialize_executor()
            for var_name in list(globals()):
                if var_name.startswith('result_') or (var_name.endswith('_instance')):
                    del globals()[var_name]
            for var_name in list(locals()):
                if var_name.startswith('result_') or (var_name.endswith('_instance')):
                    del locals()[var_name]
        else:
            if len(files)>0:
                self.loaded_files = True
            #self.logger.info('==>old conversation_continued!')
        if self.user_states == "run_pipeline":
            #self.logger.info('start initial!')
            while not self.queue.empty():
                self.queue.get()
            self.initialize_tool()
            self.loading_data(files)
            self.query_id += 1
            if self.new_task_planning:
                self.initial_goal_description = user_input
            self.user_query = user_input
            # chitchat prediction
            predicted_source = infer(user_input, self.bert_model, self.centroids, ['chitchat-data', 'topical-chat', 'api-query'])
            self.logger.info('----query inferred as {}----', predicted_source)
            if predicted_source!='api-query':
                response, _ = LLM_response(user_input, self.model_llm_type, history=[], kwargs={})  # llm
                self.callback_func('log', response, "Non API chitchat")
                return
            else:
                if conversation_started in ["True", True] and len(files)==0:
                    # return and ensure if user go on without uploading some files,
                    # we just ask once!!!!!!
                    self.callback_func('log', 'No data are uploaded! Would you ensure to go on?\nEnter [y]: Go on please.\nEnter [n]: Restart another turn.', 'User Confirmation')
                    self.user_query = user_input
                    self.update_user_state("run_pipeline_without_files")
                    self.save_state_enviro()
                    return
                #pass
            # dialog prediction
            if self.enable_multi_task and self.new_task_planning:
                pred_class = self.dialog_classifer.single_prediction(user_input, self.retriever, self.args_top_k)
                self.logger.info('----query inferred as {}----', pred_class)
                #if pred_class not in ['single']:
                if True: # set as multiple for comparing
                    self.logger.info('start multi-task!')
                    prompt = self.prompt_factory.create_prompt("multi_task", self.LIB, user_input, files)
                    #TODO: try with max_trials times
                    response, _ = LLM_response(prompt, self.model_llm_type, history=[], kwargs={})
                    self.logger.info('multi task prompt: {}, response: {}', prompt, response)
                    try:
                        steps_list = ast.literal_eval(response)['plan']
                    except:
                        try:
                            steps_list = json.loads(response)['plan']
                        except:
                            try:
                                steps_list = re.findall(r'\"(.*?)\"', response)
                            except:
                                steps_list = []
                    #self.logger.info('steps_list: {}', steps_list)
                    if not steps_list:
                        self.callback_func('log', "LLM can not return valid steps list, please redesign your prompt.", "LLM predict Error")
                        return
                    else:
                        pass
                        #self.callback_func('log', "\n - " + "\n - ".join(steps_list), "Multi step Task Planning") # "LLM return valid steps list, start executing...\n" + 
                    self.add_query(steps_list)
                    self.callback_func('log', "Ongoing subtask and remaining subtasks: \n → "+ '\n - '.join(self.user_query_list), "Task Planning")
                    sub_task = self.get_query()
                    if not sub_task:
                        raise ValueError("sub_task is empty!")
                    self.new_task_planning = False
                    self.first_task_start = True
                else:
                    sub_task = user_input
            else:
                sub_task = user_input
            # we correct the subtask description before retrieving API
            if len([i['code'] for i in self.executor.execute_code if i['success']=='True'])>0: # for non-first subtasks
                retrieved_apis = self.retriever.retrieving(sub_task, top_k=3)
                prompt = self.prompt_factory.create_prompt("modify_subtask_correction", sub_task, 
                    '\n'.join([i['code'] for i in self.executor.execute_code if i['success']=='True']), 
                    json.dumps({str(key): str(value) for key, value in self.executor.variables.items() if value['type'] not in ['function', 'module', 'NoneType']}), 
                    "\n".join(['def '+generate_function_signature(api, self.API_composite[api]['Parameters'])+':\n"""'+self.API_composite[api]['Docstring'] + '"""' for api in retrieved_apis])
                )                
                self.logger.info('modified sub_task prompt: {}', prompt)
                sub_task, _ = LLM_response(prompt, self.model_llm_type, history=[], kwargs={})
                self.logger.info('modified sub_task: {}', sub_task)
                #self.callback_func('log', 'we modify the subtask as '+sub_task, 'Modify subtask description')
                self.callback_func('log', sub_task, 'Polished subtask')
            else:
                pass
            #self.logger.info("start retrieving names!")
            # get sub_task after dialog prediction
            self.user_query = sub_task
            #self.logger.info('we filter those API with IO parameters!')
            retrieved_names = self.retriever.retrieving(self.user_query, top_k=self.args_top_k+20)
            # filter out APIs
            #self.logger.info('first_task_start: {}, self.loaded_files: {}', self.first_task_start, self.loaded_files)
            if self.first_task_start and (not self.loaded_files): # need to consider only the builtin dataset
                retrieved_names = [
                    api_name for api_name in retrieved_names
                    if all(
                        (
                            (not param['optional'] and not any(special_type in str(param['type']) for special_type in special_types) and not any(io_type in str(param['type']) for io_type in io_types) and param_name not in io_param_names)
                            or 
                            param['optional']
                        )
                        for param_name, param in self.API_composite[api_name]['Parameters'].items()
                    )
                ]
                # TODO: 240623: try smarter way, as there are some saved path instead of loading path parameters, e.g. squidpy.datasets.visium_fluo_adata
                self.logger.info('there not exist files, retrieved_names are: {}', retrieved_names)
            else: # for the first API, it is assumed to be loading data (not setting), if no files provided, must use builtin dataset,
                retrieved_names = [api_name for api_name in retrieved_names if all((not any(special_type in str(param['type']) for special_type in special_types)) for param_name, param in self.API_composite[api_name]['Parameters'].items())]
                self.logger.info('there exist files or we have already load some dataset, retrieved_names are: {}', retrieved_names)
            retrieved_names = retrieved_names[:self.args_top_k]
            self.first_task_start = False
            self.logger.info("retrieved names: {}!", retrieved_names)
            # start retrieving names
            # produce prompt
            if self.retrieve_query_mode=='similar':
                #self.logger.info('start retrieving similar queries!')
                instruction_shot_example = self.retriever.retrieve_similar_queries(self.user_query, shot_k=self.shot_k)
            else:
                #self.logger.info('start retrieving shuffled queries!')
                sampled_shuffled = random.sample(self.retriever.shuffled_data, 5)
                instruction_shot_example = "".join(["\nInstruction: " + ex['query'] + "\nFunction: " + ex['gold'] for ex in sampled_shuffled])
                similar_queries = ""
                idx = 0
                for iii in sampled_shuffled:
                    instruction = iii['query']
                    tmp_retrieved_api_list = self.retriever.retrieving(instruction, top_k=top_k)
                    # ensure the order won't affect performance
                    tmp_retrieved_api_list = random.sample(tmp_retrieved_api_list, len(tmp_retrieved_api_list))
                    # ensure the example is correct
                    if iii['gold'] in tmp_retrieved_api_list:
                        if idx<self.shot_k:
                            idx+=1
                            # only retain shot_k number of sampled_shuffled
                            tmp_str = "Instruction: " + instruction + "\nFunction: [" + iii['gold'] + "]"
                            new_function_candidates = [f"{api}, description: "+self.all_apis_json[api].replace('\n',' ') for i, api in enumerate(tmp_retrieved_api_list)] # {i}:
                            similar_queries += f"function candidates: {tmp_retrieved_api_list}\n" + "\n".join(new_function_candidates) + '\n' + tmp_str + "\n---\n"
                instruction_shot_example = similar_queries
            #self.logger.info('start predicting API!')
            api_predict_init_prompt = get_retrieved_prompt()
            #self.logger.info('api_predict_init_prompt: {}', api_predict_init_prompt)
            #print(self.all_apis_json.keys())
            retrieved_apis_prepare = ""
            retrieved_apis_prepare += str(retrieved_names) + "\n"
            for idx, api in enumerate(retrieved_names):
                retrieved_apis_prepare+=api+": "+self.all_apis_json[api].replace('\n',' ')+"\n"
            self.logger.info('retrieved_apis_prepare: {}', retrieved_apis_prepare)
            api_predict_prompt = api_predict_init_prompt.format(query=self.user_query, retrieved_apis=retrieved_apis_prepare, similar_queries=instruction_shot_example)
            self.logger.info('api_predict_prompt: {}', api_predict_prompt)
            #self.logger.info('==>start predicting API! Ask LLM: {}', api_predict_prompt)
            success = False
            for idxxx_api in range(self.predict_api_llm_retry):
                if idxxx_api>0:
                    api_predict_prompt += "\nInaccurate response:" + response + " Never respond this fake API again. Please select from the provided function candidates."
                try:
                    ori_response, _ = LLM_response(api_predict_prompt, self.model_llm_type, history=[], kwargs={})  # llm
                    #self.logger.info('==>start predicting API! LLM response: {}, {}', api_predict_prompt, ori_response)
                    # hack for if LLM answers this or that
                    """response = response.split(',')[0].split("(")[0].split(' or ')[0]
                    response = response.replace('{','').replace('}','').replace('"','').replace("'",'')
                    response = response.split(':')[0]# for robustness, sometimes llm will return api:description"""
                    response = correct_pred(ori_response, self.LIB)
                    response = response.replace('"','').replace("'","").strip()
                    #self.logger.info('self.all_apis_json keys: {}', self.all_apis_json.keys())
                    if len(response.split(','))>1:
                        response = response.split(',')[0].strip()
                    self.logger.info('==>start predicting API! api_predict_prompt, {}, correct response: {}, respose: {}', api_predict_prompt, ori_response, response)
                    if response in self.all_apis_json:
                        self.logger.info('response in self.all_apis_json')
                        self.predicted_api_name = response
                        success = True
                        break
                    else:
                        self.logger.info('use another way to parse')
                        def extract_api_calls(text, library):
                            pattern = rf'\b{library}(?:\.\w+)*\b'
                            matches = re.findall(pattern, text)
                            return [i for i in matches if i not in [library]]
                        from ..configs.model_config import get_all_variable_from_cheatsheet
                        info_json = get_all_variable_from_cheatsheet(self.LIB)
                        self.logger.info('info_json: {}', info_json)
                        lib_alias = info_json["LIB_ALIAS"]
                        extracted_api_calls = extract_api_calls(ori_response, lib_alias)
                        self.logger.info('extracted_api_calls: {}', extracted_api_calls)
                        if extracted_api_calls:
                            response = extracted_api_calls[0]
                            if response in self.all_apis_json:
                                self.predicted_api_name = response
                                success = True
                                break
                except Exception as e:
                    print('error during api prediction:', e)
                    e = traceback.format_exc()
                    self.logger.error('error during api prediction: {}', e)
            if not success:
                self.initialize_tool()
                self.callback_func('log', "LLM can not return valid API name prediction, please redesign your prompt.", "LLM predict Error")
                return
            # if the predicted API is in ambiguous API list, then show those API and select one from them
            if self.predicted_api_name in self.ambiguous_api:
                filtered_pairs = [api_pair for api_pair in self.ambiguous_pair if self.predicted_api_name in api_pair]
                self.filtered_api = [i for i in list(set(api for api_pair in filtered_pairs for api in api_pair)) if i!=self.predicted_api_name]
                #next_str = ""
                next_str = "We have retrieved an API, but we found that there may be several similar or related APIs. Please choose one of the following options:\n"
                idx_api = 1
                next_str += f"Enter [1] Retrieved API: {self.predicted_api_name}"
                description_1 = self.API_composite[self.predicted_api_name]['Docstring'].split("\n")[0]
                next_str+='\n'+description_1.replace('`','')  + '\n'
                idx_api+=1
                for api in self.filtered_api:
                    next_str += f"Enter [{idx_api}] Ambiguous API: {api}"
                    #next_str+=f"Candidate [{idx_api}]: {api}"
                    description_1 = self.API_composite[api]['Docstring'].split("\n")[0]
                    next_str+='\n'+description_1.replace('`','').replace('_', '')  + '\n'
                    idx_api+=1
                self.filtered_api = [self.predicted_api_name] + self.filtered_api
                next_str += "Enter [-1]: No appropriate API, input inquiry manually\n"
                next_str += "Enter [-2]: Skip to the next subtask"
                # for ambiguous API, we think that it might be executed more than once as ambiguous API sometimes work together
                # user can exit by entering -1
                # so we add it back to the subtask list to execute it again
                self.add_query([self.user_query], mode='pre')
                self.update_user_state("run_pipeline_after_ambiguous")
                self.initialize_tool()
                self.callback_func('log', next_str, f"Can you confirm which of the following {len(self.filtered_api)} candidates")
                self.save_state_enviro()
                return
            else:
                self.update_user_state("run_pipeline_after_fixing_API_selection")
                self.run_pipeline_after_fixing_API_selection(self.user_query)
        elif self.user_states == "run_pipeline_after_ambiguous":
            self.initialize_tool()
            ans = self.run_pipeline_after_ambiguous(user_input)
            if ans in ['break']:
                return
            self.run_pipeline_after_fixing_API_selection(user_input)
        elif self.user_states in ["run_pipeline_after_doublechecking_execution_code", "run_pipeline_after_entering_params", "run_select_basic_params", "run_pipeline_after_select_special_params", "run_select_special_params", "run_pipeline_after_doublechecking_API_selection", "run_pipeline_asking_GPT", "run_pipeline_without_files"]:
            self.initialize_tool()
            self.handle_state_transition(user_input)
        else:
            self.initialize_tool()
            self.logger.error('Unknown user state: {}', self.user_states)
            raise ValueError
    def run_pipeline_asking_GPT(self,user_input):
        self.initialize_tool()
        #self.logger.info('==>run_pipeline_asking_GPT')
        self.retry_execution_count +=1
        if self.retry_execution_count>self.retry_execution_limit:
            self.logger.error('retry_execution_count exceed the limit! Exit the dialog!')
            self.callback_func('log', 'code generation using GPT has exceed the limit! Please choose other lib and re-enter the inquiry! You can use GPT again once you have executed code successfully through our tool!', 'Error')
            self.update_user_state('run_pipeline')
            self.save_state_enviro()
            return
        prompt = self.prompt_factory.create_prompt(
            'LLM_code_generation',
            self.user_query,
            str(self.executor.execute_code),
            str({str(key): str(value) for key, value in self.executor.variables.items() if value['type'] not in ['function', 'module', 'NoneType']}),
            self.LIB
        )
        response, _ = LLM_response(prompt, self.model_llm_type, history=[], kwargs={})
        if '```python' in response and response.endswith('```'):
            response = response.split('```python')[1].split('```')[0]
        if '\"\"\"' in response:
            response = response.replace('\"\"\"', '')
        newer_code = response
        self.execution_code = newer_code
        self.callback_func('code', self.execution_code, "Executed code")
        # LLM response
        api_docstring = 'def '+generate_function_signature(self.predicted_api_name, self.API_composite[self.predicted_api_name]['Parameters'])+':\n"""'+self.API_composite[self.predicted_api_name]['Docstring'] + '"""'
        summary_prompt = self.prompt_factory.create_prompt('summary_full', self.user_query, api_docstring, self.execution_code)
        response, _ = LLM_response(summary_prompt, self.model_llm_type, history=[], kwargs={})
        self.callback_func('log', response, "Task summary")
        self.callback_func('log', "Could you confirm should this task be executed?\nEnter [y]: Go on please.\nEnter [n]: Re-generate the code\nEnter [r], Restart another turn", "User Confirmation")
        self.update_user_state("run_pipeline_after_doublechecking_execution_code")
        self.save_state_enviro()
        return 
        
    def handle_unknown_state(self, user_input):
        self.logger.info("Unknown state: {}", self.user_states)

    def handle_state_transition(self, user_input):
        method = getattr(self, self.user_states, self.handle_unknown_state)
        return method(user_input)
    
    def run_pipeline_after_ambiguous(self,user_input):
        self.initialize_tool()
        #self.logger.info('==>run_pipeline_after_ambiguous')
        user_input = user_input.strip()
        try:
            user_index = int(user_input)
        except ValueError:
            self.callback_func('log', "Error: the input is not a number.\nPlease re-enter the index", "Index Error")
            self.update_user_state("run_pipeline_after_ambiguous")
            return 'break'
        if user_index==-1:
            self.update_user_state("run_pipeline")
            self.callback_func('log', "We will start another round. Could you re-enter your inquiry?", "Start another round")
            self.save_state_enviro()
            return 'break'
        if user_index==-2:
            sub_task = self.get_query()
            if self.user_query_list:
                self.callback_func('log', "Ongoing subtask and remaining subtasks: \n → "+ '\n - '.join(self.user_query_list), "Task Planning")
                sub_task = self.get_query()
                self.user_query = sub_task
            self.update_user_state("run_pipeline")
            self.save_state_enviro()
            self.run_pipeline(sub_task, self.LIB, top_k=3, files=[],conversation_started=False,session_id=self.session_id)
            return
        try:
            self.filtered_api[user_index-1]
        except IndexError:
            self.callback_func('log', "Error: the input index exceed the maximum length of ambiguous API list\nPlease re-enter the index", "Index Error")
            self.update_user_state("run_pipeline_after_ambiguous")
            return 'break'
        self.update_user_state("run_pipeline_after_fixing_API_selection")
        self.predicted_api_name = self.filtered_api[user_index-1]
        self.save_state_enviro()
    def process_api_info(self, api_info, single_api_name):
        relevant_apis = api_info.get(single_api_name, {}).get("relevant APIs")
        if not relevant_apis:
            return [{single_api_name: {'type': api_info[single_api_name]['api_type']}}]
        else:
            return [{relevant_api_name: {'type': api_info.get(relevant_api_name, {}).get("api_type")}} for relevant_api_name in relevant_apis]
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
    def update_user_state(self, new_state):
        self.last_user_states = self.user_states
        self.user_states = new_state
        #print(f"Updated state from {self.last_user_states} to {self.user_states}")
    def run_pipeline_after_fixing_API_selection(self,user_input):
        self.initialize_tool()
        # check if composite API/class method API, return the relevant APIs
        if isinstance(self.predicted_api_name, str):
            self.relevant_api_list = self.process_api_info(self.API_composite, self.predicted_api_name)
        elif isinstance(self.predicted_api_name, list) and len(self.predicted_api_name) == 1:
            self.relevant_api_list = self.process_api_info(self.API_composite, self.predicted_api_name[0])
        elif isinstance(self.predicted_api_name, list) and len(self.predicted_api_name) > 1:
            raise ValueError("Predicting Multiple APIs during one inference is not supported yet")
            self.relevant_api_list = []
            for api_name in self.predicted_api_name:
                self.relevant_api_list.extend(self.process_api_info(self.API_composite, api_name))
        else:
            self.relevant_api_list = []
            self.logger.error('Invalid type or empty list for self.predicted_api_name: {}', type(self.predicted_api_name))
            # TODO: should return
        self.api_name_json = self.check_and_insert_class_apis(self.API_composite, self.relevant_api_list)# also contains class API
        self.logger.info('self.api_name_json: {}', self.api_name_json)
        #self.update_user_state("run_pipeline")
        # summary task
        api_docstring = 'def '+generate_function_signature(self.predicted_api_name, self.API_composite[self.predicted_api_name]['Parameters'])+':\n"""'+self.API_composite[self.predicted_api_name]['Docstring'] + '"""'
        summary_prompt = self.prompt_factory.create_prompt('summary', user_input, api_docstring)
        response, _ = LLM_response(summary_prompt, self.model_llm_type, history=[], kwargs={})
        self.logger.info(f'summary_prompt: {summary_prompt}, summary_prompt response: {response}')
        self.callback_func('log', response, f"Predicted API: {self.predicted_api_name}")
        self.callback_func('log', "Could you confirm whether this API should be called?\nEnter [y]: Go on please.\nEnter [n]: Restart another turn", "User Confirmation")
        self.update_user_state("run_pipeline_after_doublechecking_API_selection")
        self.save_state_enviro()
    
    def run_pipeline_after_doublechecking_API_selection(self, user_input):
        self.initialize_tool()
        #self.logger.info('==>run_pipeline_after_doublechecking_API_selection')
        user_input = str(user_input)
        if user_input in ['n', 'N']:
            if self.new_task_planning or self.retry_modify_count>=3: # if there is no task planning
                self.update_user_state("run_pipeline")
                self.callback_func('log', "We will start another round. Could you re-enter your inquiry?", "Start another round")
                self.retry_modify_count = 0
                self.save_state_enviro()
            else: # if there is task planning, we just update this subtask
                self.retry_modify_count += 1
                self.callback_func('log', "As this subtask is not exactly what you want, we polish the subtask and re-run the code generation pipeline", "Continue to the same subtask")
                #sub_task = self.get_query()
                # polish and modify the sub_task
                retrieved_apis = self.retriever.retrieving(user_input, top_k=3)
                prompt = self.prompt_factory.create_prompt("modify_subtask_correction", 
                    self.user_query, 
                    '\n'.join([i['code'] for i in self.executor.execute_code if i['success']=='True']), 
                    json.dumps({str(key): str(value) for key, value in self.executor.variables.items() if value['type'] not in ['function', 'module', 'NoneType']}), 
                    "\n".join(['def '+generate_function_signature(api, self.API_composite[api]['Parameters'])+':\n"""'+self.API_composite[api]['Docstring'] + '"""' for api in retrieved_apis])
                )
                self.user_query, _ = LLM_response(prompt, self.model_llm_type, history=[], kwargs={})
                #self.logger.info('Polished subtask: {}', self.user_query)
                self.callback_func('log', self.user_query, 'Polished subtask')
                self.update_user_state("run_pipeline")
                self.save_state_enviro()
                self.run_pipeline(self.user_query, self.LIB, top_k=3, files=[],conversation_started=False,session_id=self.session_id)
            return
        elif user_input in ['y', 'Y']:
            pass
        else:
            self.callback_func('log', "The input was not y or n, please enter the correct value.", "Index Error")
            self.save_state_enviro()
            # user_states didn't change
            return
        self.logger.info('self.predicted_api_name: {}', self.predicted_api_name)
        if len([i['code'] for i in self.executor.execute_code if i['success']=='True'])>0: # for non-first subtasks
            prompt = self.prompt_factory.create_prompt("modify_subtask_parameters", self.user_query, 
                '\n'.join([i['code'] for i in self.executor.execute_code if i['success']=='True']), 
                json.dumps({str(key): str(value) for key, value in self.executor.variables.items() if value['type'] not in ['function', 'module', 'NoneType']}), 
                "\n"+ 'def '+generate_function_signature(self.predicted_api_name, self.API_composite[self.predicted_api_name]['Parameters'])+':\n"""'+self.API_composite[self.predicted_api_name]['Docstring'] + '"""')
            self.user_query, _ = LLM_response(prompt, self.model_llm_type, history=[], kwargs={})
            self.logger.info('modified sub_task prompt: {}', prompt)
            self.logger.info('modified sub_task: {}', self.user_query)
            #self.callback_func('log', 'we modify the subtask as '+sub_task, 'Modify subtask description')
            #self.callback_func('log', self.user_query, 'Polished subtask')
            # we leave this to the `highlight` card
        else:
            pass
        combined_params = {}
        #self.logger.info('self.api_name_json: {}', self.api_name_json)
        # if the class API has already been initialized, then skip it
        executor_variables = {var_name: var_info["value"] for var_name, var_info in self.executor.variables.items() if str(var_info["value"]) not in ["None"]}
        #self.logger.info('executor_variables: {}', executor_variables)
        try:
            for api in self.api_name_json:
                maybe_class_name = api.split('.')[-1]
                maybe_instance_name = maybe_class_name.lower() + "_instance"
                # 240520: modified, support for variable with none xx_instance name
                if self.API_composite[api]['api_type'] in ['class', 'unknown']:
                    from ..inference.execution_UI import find_matching_instance
                    matching_instance, is_match = find_matching_instance(api, executor_variables)
                    #self.logger.info(f'executor_variables: {executor_variables}, api: {api}, matching_instance: {matching_instance}, is_match: {is_match}')
                    if is_match:
                        maybe_instance_name = matching_instance
                        continue
                else:
                    pass
                combined_params.update(self.API_composite[api]['Parameters'])
        except Exception as e:
            self.logger.info('error in combining parametesr: {}', e)
        #self.logger.info('combined_params: {}', combined_params)
        parameters_name_list = [key for key, value in combined_params.items() if (key not in self.path_info_list)] # if (not value['optional'])
        #self.logger.info('parameters_name_list: {}', parameters_name_list)
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
        #filter out special type parameters, do not infer them using LLM
        api_parameters_information = [param for param in api_parameters_information if any(basic_type in param['type'] for basic_type in basic_types)]
        parameters_name_list = [param_info['name'] for param_info in api_parameters_information]
        #
        # we only support one API for now
        assert len(self.relevant_api_list)==1
        api_name_tmp_list = self.relevant_api_list[0]
        api_name_tmp = list(api_name_tmp_list.keys())[0]
        apis_name = api_name_tmp
        apis_description = self.API_composite[api_name_tmp]['description']
        # 240531 added, predict parameters in chunked setting
        param_tmp = {i:self.API_composite[apis_name]['Parameters'][i] for i in self.API_composite[apis_name]['Parameters'] if (self.API_composite[apis_name]['Parameters'][i]['description'] is not None) and (not any(special_type in str(self.API_composite[apis_name]['Parameters'][i]['type']) for special_type in special_types)) and (str(self.API_composite[apis_name]['Parameters'][i]['type']) not in io_types) and (i not in io_param_names)}
        
        boolean_params = {k: v for k, v in param_tmp.items() if 'boolean' in str(v['type']) or 'bool' in str(v['type'])}
        literal_params = {k: v for k, v in param_tmp.items() if 'literal' in str(v['type']) or 'Literal' in str(v['type'])}
        int_params = {k: v for k, v in param_tmp.items() if k not in boolean_params and k not in literal_params}
        boolean_document = json_to_docstring(apis_name, self.API_composite[apis_name]["description"], boolean_params)
        literal_document = json_to_docstring(apis_name, self.API_composite[apis_name]["description"], literal_params)
        int_document = json_to_docstring(apis_name, self.API_composite[apis_name]["description"], int_params)
        try:
            predicted_params = asyncio.run(self.predict_all_params(api_name_tmp, boolean_params, literal_params, int_params, boolean_document, literal_document, int_document))
            #self.logger.info('predicted_parameters async run finised! {}', predicted_params)
            param_success = True
        except Exception as e:
            e = traceback.format_exc()
            self.logger.error('error for parameters: {}', e)
            param_success = False
        if predicted_params:
            pass
        else:
            predicted_params = {}
        # filter out the parameters which value is same as their default value
        self.logger.info('now filter out the parameters which value is same as their default value!')
        self.logger.info('predicted_parameters: {}', predicted_params)
        predicted_parameters = {k: v for k, v in predicted_params.items() if ((k in param_tmp) and (str(v) != param_tmp[k]["default"]) and (str(v) not in [None, "None", "null", 'NoneType']))}
        self.logger.info('predicted_parameters after filtering: {}', predicted_parameters)
        if len(parameters_name_list)==0:
            #self.logger.info('if there is no required parameters, skip using LLM')
            response = "[]"
            predicted_parameters = {}
        else:
            #self.logger.info('there exists required parameters, using LLM')
            if not param_success:
                #self.logger.info('param_success is False')
                self.callback_func('log', "LLM can not return valid parameters prediction, please redesign prompt in backend if you want to predict parameters. We will skip parameters prediction currently", "LLM predict Error")
                response = "{}"
                predicted_parameters = {}
        #self.logger.info('filter predicted_parameters: {}', predicted_parameters)
        required_param_list = [param_name for param_name, param_info in self.API_composite[apis_name]['Parameters'].items() if param_info['type'] in special_types or param_info['type'] in io_types or param_name in io_param_names]
        predicted_parameters = {key: value for key, value in predicted_parameters.items() if value not in [None, "None", "null"] or key in required_param_list}
        self.logger.info('after filtering, predicted_parameters: {}', predicted_parameters)
        colored_sentence = label_sentence(self.user_query, predicted_parameters)
        self.callback_func('log', colored_sentence, 'Highlight parameters value in polished subtask description')
        #self.logger.info('colored_sentence: {}', colored_sentence)
        # generate api_calling
        self.predicted_api_name, api_calling, self.parameters_info_list = generate_api_calling(self.predicted_api_name, self.API_composite[self.predicted_api_name], predicted_parameters)
        self.logger.info('parameters_info_list: {}', self.parameters_info_list)
        if len(self.api_name_json)> len(self.relevant_api_list):
            #assume_class_API = list(set(list(self.api_name_json.keys()))-set(self.relevant_api_list))[0]
            assume_class_API = '.'.join(self.predicted_api_name.split('.')[:-1])
            tmp_class_predicted_api_name, tmp_class_api_calling, tmp_class_parameters_info_list = generate_api_calling(assume_class_API, self.API_composite[assume_class_API], predicted_parameters)
            self.logger.info(f'assume_class_API: {assume_class_API}, tmp_class_predicted_api_name: {tmp_class_predicted_api_name}, tmp_class_api_calling: {tmp_class_api_calling}, tmp_class_parameters_info_list: {tmp_class_parameters_info_list}')
            fix_update = True
            for api in self.api_name_json:
                maybe_class_name = api.split('.')[-1]
                maybe_instance_name = maybe_class_name.lower() + "_instance"
                # 240520: modified, 
                if self.API_composite[api]['api_type'] in ['class', 'unknown']:
                    executor_variables = {}
                    for var_name, var_info in self.executor.variables.items():
                        var_value = var_info["value"]
                        if str(var_value) not in ["None"]:
                            executor_variables[var_name] = var_value
                    try:
                        from ..inference.execution_UI import find_matching_instance
                        matching_instance, is_match = find_matching_instance(api, executor_variables)
                    except Exception as e:
                        e = traceback.format_exc()
                        self.logger.error('error during matching_instance: {}', e)
                    self.logger.info(f'executor_variables: {executor_variables}, matching_instance: {matching_instance}, is_match: {is_match}')
                    if is_match:
                        maybe_instance_name = matching_instance
                        fix_update = False
                else:
                    pass
            if fix_update:
                self.parameters_info_list['parameters'].update(tmp_class_parameters_info_list['parameters'])
        ####### infer parameters
        # $ param
        self.selected_params = self.executor.select_parameters(self.parameters_info_list['parameters'])
        # $ param if not fulfilled
        none_dollar_value_params = [param_name for param_name, param_info in self.selected_params.items() if param_info["value"] in ['$']]
        self.logger.info('Automatically selected params for $, after selection the parameters are: {}, none_dollar_value_params: {}', json.dumps(self.selected_params), json.dumps(none_dollar_value_params))
        if none_dollar_value_params:
            self.callback_func('log', "However, there are still some parameters with special type undefined. Please start from uploading data, or check your parameter type in json files.", "Missing Parameters: special type")
            self.update_user_state("run_pipeline")
            self.save_state_enviro()
            return
        # $ param if multiple choice
        multiple_dollar_value_params = [param_name for param_name, param_info in self.selected_params.items() if ('list' in str(type(param_info["value"]))) and (len(param_info["value"])>1)]
        self.filtered_params = {key: value for key, value in self.parameters_info_list['parameters'].items() if (key in multiple_dollar_value_params)}
        if multiple_dollar_value_params:
            self.callback_func('log', "There are many variables match the expected type. Please determine which one to choose", "Choosing Parameters: special type")
            tmp_input_para = ""
            for idx, api in enumerate(self.filtered_params):
                if idx!=0:
                    tmp_input_para+=" and "
                tmp_input_para+="'"+self.filtered_params[api]['description']+ "'"
                tmp_input_para+=f"('{api}': {self.filtered_params[api]['type']}), "
            self.callback_func('log', f"The predicted API takes {tmp_input_para} as input. However, there are still some parameters undefined in the query.", "Enter Parameters: special type", "red")
            self.update_user_state("run_select_special_params")
            self.run_select_special_params(user_input)
            self.save_state_enviro()
            return
        self.run_pipeline_after_select_special_params(user_input)

    def get_success_code_with_val(self, val):
        for i in self.executor.execute_code:
            if i['success']=='True' and val in i['code']:
                return i['code']
        self.callback_func('log', "Can not find the executed code corresponding to the expected parameters", "Error Enter Parameters: special type","red")
    def run_select_special_params(self, user_input):
        self.initialize_tool()
        #self.logger.info('==>run_select_special_params')
        if self.last_user_states == "run_select_special_params":
            self.selected_params = self.executor.makeup_for_missing_single_parameter_type_special(params = self.selected_params, param_name_to_update=self.last_param_name, user_input = user_input)
        #print('self.filtered_params: {}', json.dumps(self.filtered_params))
        if len(self.filtered_params)>1:
            self.last_param_name = list(self.filtered_params.keys())[0]
            candidate_text = ""
            for val in self.selected_params[self.last_param_name]["value"]:
                get_val_code = self.get_success_code_with_val(val)
                candidate_text+=f'{val}: {get_val_code}\n'
            self.callback_func('log', f"Which value do you think is appropriate for the parameters '{self.last_param_name}'? We find some candidates:\n {candidate_text}. ", "Enter Parameters: special type", "red")
            self.update_user_state("run_select_special_params")
            del self.filtered_params[self.last_param_name]
            #print('self.filtered_params: {}', json.dumps(self.filtered_params))
            self.save_state_enviro()
            return
        elif len(self.filtered_params)==1:
            self.last_param_name = list(self.filtered_params.keys())[0]
            candidate_text = ""
            for val in self.selected_params[self.last_param_name]["value"]:
                get_val_code = self.get_success_code_with_val(val)
                candidate_text+=f'{val}: {get_val_code}\n'
            self.callback_func('log', f"Which value do you think is appropriate for the parameters '{self.last_param_name}'? We find some candidates \n {candidate_text}. ", "Enter Parameters: special type", "red")
            self.update_user_state("run_pipeline_after_select_special_params")
            del self.filtered_params[self.last_param_name]
            self.save_state_enviro()
        else:
            self.callback_func('log', "The parameters candidate list is empty", "Error Enter Parameters: basic type", "red")
            self.save_state_enviro()
            raise ValueError

    def run_pipeline_after_select_special_params(self,user_input):
        if self.last_user_states == "run_select_special_params":
            self.selected_params = self.executor.makeup_for_missing_single_parameter_type_special(params = self.selected_params, param_name_to_update=self.last_param_name, user_input = user_input)
        # @ param
        #self.logger.info('==> run_pipeline_after_select_special_params')
        none_at_value_params = [param_name for param_name, param_info in self.selected_params.items() if (param_info["value"] in ['@']) and (param_name not in self.path_info_list)]
        self.filtered_params = {key: value for key, value in self.parameters_info_list['parameters'].items() if (value["value"] in ['@']) and (key not in self.path_info_list)}
        self.filtered_pathlike_params = {key: value for key, value in self.parameters_info_list['parameters'].items() if (value["value"] in ['@']) and (key in self.path_info_list)}
        # TODO: add condition later: if uploading data files, 
        # avoid asking Path params, assign it as './tmp'
        if none_at_value_params: # TODO: add type PathLike
            #self.logger.info('if exist non path, basic type parameters, start selecting parameters')
            tmp_input_para = ""
            for idx, api in enumerate(self.filtered_params):
                if idx!=0:
                    tmp_input_para+=" and "
                tmp_input_para+=self.filtered_params[api]['description']
                tmp_input_para+=f"('{api}': {self.filtered_params[api]['type']}), "
            self.callback_func('log', f"The predicted API takes {tmp_input_para} as input. However, there are still some parameters undefined in the query.", "Enter Parameters: basic type", "red")
            self.user_states = "run_select_basic_params"
            self.run_select_basic_params(user_input)
            self.save_state_enviro()
            return
        self.run_pipeline_after_entering_params(user_input)
    
    def run_select_basic_params(self, user_input):
        self.initialize_tool()
        #self.logger.info('==>run_select_basic_params')
        if self.last_user_states == "run_select_basic_params":
            self.selected_params = self.executor.makeup_for_missing_single_parameter(params = self.selected_params, param_name_to_update=self.last_param_name, user_input = user_input)
        self.logger.info('self.filtered_params: {}', json.dumps(self.filtered_params))
        if len(self.filtered_params)>1:
            self.last_param_name = list(self.filtered_params.keys())[0]
            self.callback_func('log', "Which value do you think is appropriate for the parameters '" + self.last_param_name + "'?", "Enter Parameters: basic type","red")
            self.update_user_state("run_select_basic_params")
            del self.filtered_params[self.last_param_name]
            self.save_state_enviro()
            return
        elif len(self.filtered_params)==1:
            self.last_param_name = list(self.filtered_params.keys())[0]
            self.callback_func('log', "Which value do you think is appropriate for the parameters '" + self.last_param_name + "'?", "Enter Parameters: basic type", "red")
            self.update_user_state("run_pipeline_after_entering_params")
            del self.filtered_params[self.last_param_name]
            self.save_state_enviro()
        else:
            # break out the pipeline
            self.callback_func('log', "The parameters candidate list is empty", "Error Enter Parameters: basic type","red")
            self.save_state_enviro()
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
    def extract_parameters(self, api_name_json, api_info, selected_params):
        parameters_combined = []
        for api_name in api_name_json:
            details = api_info[api_name]
            parameters = details["Parameters"]
            api_params = {param_name: {"type": param_details["type"]} for param_name, param_details in parameters.items() if (param_name in selected_params) or (not param_details['optional']) or (param_name=="color" and (("scanpy.pl" in api_name) or ("squidpy.pl" in api_name))) or (param_name=='encodings' and (api_name.startswith('ehrapy.pp') or api_name.startswith('ehrapy.preprocessing'))) or (param_name=='encoded' and (api_name.startswith('ehrapy.')))} # TODO: currently not use optional parameters!!!
            # TODO: add which have been predicted in selected_params
            api_params.update({})
            combined_params = {}
            #for param_name, param_info in api_params.items():
            #    if param_name not in combined_params:
            #        combined_params[param_name] = param_info
            combined_params = {param_name: param_info for param_name, param_info in api_params.items() if param_name not in combined_params}
            parameters_combined.append(combined_params)
        return parameters_combined

    def run_pipeline_after_entering_params(self, user_input):
        self.initialize_tool()
        if self.last_user_states == "run_select_basic_params":
            self.selected_params = self.executor.makeup_for_missing_single_parameter(params = self.selected_params, param_name_to_update=self.last_param_name, user_input = user_input)
        #self.logger.info('==>run pipeline after entering parameters')
        self.update_user_state("run_pipeline")
        self.image_file_list = self.update_image_file_list()
        if self.filtered_pathlike_params:
            # add 'tmp' 
            for key in self.filtered_pathlike_params:
                param_info = self.filtered_pathlike_params[key]
                self.selected_params[key] = {
                    "type": param_info["type"],
                    "value": "./tmp",
                    "valuefrom": 'userinput',
                    "optional": param_info["optional"]
                }
        # split parameters according to multiple API, or class/method API
        tmp_api_info = {api: details for api, details in self.API_composite.items() if api in self.api_name_json}
        parameters_list = self.extract_parameters(self.api_name_json, tmp_api_info, self.selected_params)
        self.logger.info('==>self.selected_params: {}, tmp_api_info: {}, parameters_list: {}', json.dumps(self.selected_params), json.dumps(tmp_api_info), json.dumps(parameters_list))
        extracted_params = self.split_params(self.selected_params, parameters_list)
        extracted_params_dict = {api_name: extracted_param for api_name, extracted_param in zip(self.api_name_json, extracted_params)}
        self.logger.info('==>self.api_name_json: {}, parameters_list: {}, extracted_params: {}, extracted_params_dict: {}',str(self.api_name_json), str(parameters_list), str(extracted_params), str(extracted_params_dict))
        api_params_list = []
        for idx, api_name in enumerate(self.api_name_json):
            #if self.api_name_json[api_name]['type'] in ['class', 'unknown']: # !
            class_selected_params = {}
            fake_class_api = '.'.join(api_name.split('.')[:-1])
            if fake_class_api in self.api_name_json:
                if self.api_name_json[fake_class_api]['type'] in ['class', 'unknown']:
                    class_selected_params = extracted_params_dict[fake_class_api]
            api_data_single = self.API_composite[api_name]
            # two patches for pandas type data / squidpy parameters
            if ('inplace' in api_data_single['Parameters']) and (api_name.startswith('scanpy') or api_name.startswith('squidpy')):
                extracted_params[idx]['inplace'] = {
                    "type": api_data_single['Parameters']['inplace']['type'],
                    "value": True,
                    "valuefrom": 'value',
                    "optional": True,
                }
            if 'shape' in api_data_single['Parameters'] and 'pl.spatial_scatter' in api_name:
                extracted_params[idx]['shape'] = {
                    "type": api_data_single['Parameters']['shape']['type'],
                    "value": "None",
                    "valuefrom": 'value',
                    "optional": True,
                }
            # don't include class API, just include class.attribute API
            if api_data_single['api_type'] not in ['class', 'unknown']:
                # when using class.attribute API, only include the API's information.
                api_params_list.append({"api_name":api_name, 
                "parameters":extracted_params[idx], 
                "return_type":api_data_single['Returns']['type'],
                "class_selected_params":class_selected_params,
                "api_type":api_data_single['api_type']})
            else: # ==`class`
                if len(self.api_name_json)==1:
                    # When using class API, only include class API's
                    api_params_list.append({"api_name":api_name, 
                    "parameters":extracted_params[idx], 
                    "return_type":api_data_single['Returns']['type'],
                    "class_selected_params":extracted_params[idx],
                    "api_type":api_data_single['api_type']})
                else:
                    pass
        # add optional cards
        optional_param = {key: value for key, value in api_data_single['Parameters'].items() if value['optional']}
        self.logger.info('==>api_params_list: {}, ==>optional_param: {}, len(optional_param) {}', json.dumps(api_params_list), json.dumps(optional_param), len(optional_param))
        if False: # TODO: if True, to debug the optional card showing
            if len(optional_param)>0:
                self.callback_func('log', "Do you want to modify the optional parameters? You can leave it unchange if you don't want to modify the default value.", "Optional cards")
                self.callback_func('optional', convert_bool_values(correct_bool_values(optional_param)), "Optional cards")
            else:
                pass
        # TODO: real time adjusting execution_code according to optionalcard
        self.execution_code = self.executor.generate_execution_code(api_params_list)
        self.logger.info('==>api_params_list: {}, execution_code: {}', api_params_list, self.execution_code)
        self.callback_func('code', self.execution_code, "Executed code")
        # LLM response
        api_data_single = self.API_composite[self.predicted_api_name]
        api_docstring = 'def '+generate_function_signature(self.predicted_api_name, self.API_composite[self.predicted_api_name]['Parameters'])+':\n"""'+self.API_composite[self.predicted_api_name]['Docstring'] + '"""'
        summary_prompt = self.prompt_factory.create_prompt('summary_full', user_input, api_docstring, self.execution_code)
        response, _ = LLM_response(summary_prompt, self.model_llm_type, history=[], kwargs={})
        self.callback_func('log', response, "Task summary")
        self.callback_func('log', "Could you confirm whether this task is what you aimed for, and the code should be executed?\nEnter [y]: Go on please\nEnter [n]: Re-direct to the parameter input step\nEnter [r]: Restart another turn", "User Confirmation")
        self.update_user_state("run_pipeline_after_doublechecking_execution_code")
        self.save_state_enviro()
        
    def run_pipeline_after_doublechecking_execution_code(self, user_input):
        self.initialize_tool()
        #self.logger.info('==> run_pipeline_after_doublechecking_execution_code')
        # if check, back to the last iteration and status
        if user_input in ['y', 'n', 'r', 'Y', 'N', 'R']:
            if user_input in ['n', 'N']:
                if self.last_user_states=='run_pipeline_asking_GPT':
                    self.update_user_state("run_pipeline_asking_GPT")
                    self.callback_func('log', "We will redirect to the LLM model to re-generate the code", "Re-generate the code")
                    self.save_state_enviro()
                    self.run_pipeline_asking_GPT(self.user_query)
                    return
                else:
                    self.update_user_state("run_pipeline_after_doublechecking_API_selection")#TODO: check if exist issue
                    self.callback_func('log', "We will redirect to the parameters input", "Re-enter the parameters")
                    self.save_state_enviro()
                    self.run_pipeline_after_doublechecking_API_selection('y')
                    return
            elif user_input in ['r', 'R']:
                self.update_user_state("run_pipeline")
                self.callback_func('log', "We will start another round. Could you re-enter your inquiry?", "Start another round")
                self.save_state_enviro()
                return
            else:
                pass
        else:
            self.callback_func('log', "The input was not y or n or r, please enter the correct value.", "Index Error")
            self.save_state_enviro()
            # user_states didn't change
            return
        # else, continue
        execution_code_list = self.execution_code.split('\n')
        self.plt_status = plt.get_fignums()
        temp_output_file = "./sub_process_execution.txt"
        process = multiprocessing.Process(target=self.run_pipeline_execution_code_list, args=(execution_code_list, temp_output_file))
        process.start()
        #process.join()
        while process.is_alive():
            #self.logger.info('process is alive!')
            time.sleep(1)
            with open(temp_output_file, 'r') as file:
                accumulated_output = file.read() ######?
                #self.logger.info('accumulated_output: {}', accumulated_output)
                self.callback_func('log', accumulated_output, "Executing results", enhance_indexxxx=False)
        self.indexxxx+=1
        with open("./tmp/tmp_output_run_pipeline_execution_code_list.txt", 'r') as file:
            output_str = file.read()
            result = json.loads(output_str)
        code = result['code']
        output_list = result['output_list']
        self.executor.load_environment("./tmp/tmp_output_run_pipeline_execution_code_variables.pkl")
        self.logger.info('check: {}, {}', code, output_list)
        if len(execution_code_list)>0:
            self.last_execute_code = self.get_last_execute_code(code)
        else:
            self.last_execute_code = {"code":"", 'success':"False"}
        self.logger.info('self.executor.variables: {}, self.executor.execute_code: {}, self.last_execute_code: {}', json.dumps(list(self.executor.variables.keys())), json.dumps(self.executor.execute_code), str(self.last_execute_code))
        try:
            content = '\n'.join(output_list)
        except:
            try:
                content = self.last_execute_code['error']
            except Exception as e:
                self.logger.error('error for loading content: {}', e)
                content = ""
        self.logger.info('content: {}', content)
        # show the new variable 
        if self.last_execute_code['code'] and self.last_execute_code['success']=='True':
            # if execute, visualize value
            code = self.last_execute_code['code']
            vari = [i.strip() for i in code.split('(')[0].split('=')]
            self.logger.info('-----code: {} -----vari: {}', code, vari)
            tips_for_execution_success = True
            if len(vari)>1:
                #if self.executor.variables[vari[0]]['value'] is not None:
                if (vari[0] in self.executor.variables) and ((vari[0].startswith('result_')) or (vari[0].endswith('_instance'))):
                    print_val = vari[0]
                    print_value = self.executor.variables[print_val]['value']
                    print_type = self.executor.variables[print_val]['type']
                    self.callback_func('log', "We obtain a new variable: " + str(print_value), "Executed results [Success]")
                    if print_type=='AnnData': #TODO: 'AnnData' in print_type
                        #self.logger.info('if the new variable is of type AnnData, ')
                        visual_attr_list = [i_tmp for i_tmp in list(dir(print_value)) if not i_tmp.startswith('_')]
                        #if len(visual_attr_list)>0:
                        if 'obs' in visual_attr_list:
                            visual_attr = 'obs' # TODO: visual_attr_list[0]
                            self.logger.info('visualize {} attribute', visual_attr)
                            output_table = getattr(self.executor.variables[vari[0]]['value'], "obs", None).head(5).to_csv(index=True, header=True, sep=',', lineterminator='\n')
                            # if exist \n in the last index, remove it
                            last_newline_index = output_table.rfind('\n')
                            if last_newline_index != -1:
                                output_table = output_table[:last_newline_index] + '' + output_table[last_newline_index + 1:]
                            else:
                                pass
                            self.callback_func('log', "We visualize the first 5 rows of the table data", "Executed results [Success]", tableData=output_table)
                        else:
                            pass
                    try:
                        #self.logger.info('if exist table, visualize it')
                        output_table = self.executor.variables[vari[0]]['value'].head(5).to_csv(index=True, header=True, sep=',', lineterminator='\n')
                        last_newline_index = output_table.rfind('\n')
                        if last_newline_index != -1:
                            output_table = output_table[:last_newline_index] + '' + output_table[last_newline_index + 1:]
                        else:
                            pass
                        self.callback_func('log', "We visualize the first 5 rows of the table data", "Executed results [Success]", tableData=output_table)
                    except:
                        pass
                else:
                    self.logger.info('Something wrong with variables! success executed variables didnt contain targeted variable')
                tips_for_execution_success = False
            else:
                pass
            #self.logger.info('if generate image, visualize it')
            new_img_list = self.update_image_file_list()
            new_file_list = set(new_img_list)-set(self.image_file_list)
            if new_file_list:
                for new_img in new_file_list:
                    base64_image = convert_image_to_base64(os.path.join(self.image_folder,new_img))
                    if base64_image:
                        self.callback_func('log', "We visualize the obtained figure. Try to zoom in or out the figure.", "Executed results [Success]", imageData=base64_image)
                        tips_for_execution_success = False
            self.image_file_list = new_img_list
            if tips_for_execution_success: # if no output, no new variable, present the log
                self.callback_func('log', str(content), "Executed results [Success]")
            self.retry_execution_count = 0
        else:
            self.logger.info('Execution Error: {}', content)
            try:
                tmp_output = extract_last_error_sentence("\n".join(list(set(output_list))))
            except:
                tmp_output = extract_last_error_sentence(content)
            self.callback_func('log', tmp_output, "Executed results [Fail]")
            if self.retry_execution_count<self.retry_execution_limit:
                self.retry_execution_count +=1
                # 240521: automatically regenerated code by LLM
                #prompt = self.prompt_factory.create_prompt('execution_correction', self.user_query, str(self.executor.execute_code), self.last_execute_code['code'], content, str(self.executor.variables), self.LIB)
                # 240531: add newer execution prompt, which combines information from docstring examples, api_callings, and github issue solutions, and traceback informations
                # remove tracebackerror because the information is nonsense, only keep the line of 'ValueError: '
                if output_list:
                    executor_info = "\n".join(list(set([str(iii) for iii in output_list])))
                else:
                    executor_info = ""
                error_index = next((index for index, value in enumerate(output_list) if 'Error:' in value), None)
                if error_index is not None:
                    filtered_output_list = output_list[error_index:]
                else:
                    filtered_output_list = []
                if filtered_output_list:
                    executor_info = "\n".join(filtered_output_list)
                else:
                    executor_info = ""
                # append executor_info to the self.executor.execute_code
                self.executor.execute_code[-1]['traceback'] = executor_info
                from ..models.query_issue_corpus import retrieved_issue_solution, search_github_issues
                from ..gpt.get_summarize_tutorial import extract_imports, get_sub_API
                # use github issue retriever
                #possible_solution = retrieved_issue_solution(self.LIB, 3, executor_info, "sentencebert", "issue_title") # issue_description
                possible_solution = search_github_issues(self.LIB, 2, executor_info)
                self.logger.info('executor_info: {}, possible_solution: {}', executor_info, possible_solution)
                try:
                    if isinstance(possible_solution, list):
                        possible_solution = '\n'.join(possible_solution)
                    else:
                        possible_solution = str(possible_solution)
                except:
                    possible_solution = str(possible_solution)
                # do not use github issue retriever
                #possible_solution = ""
                self.logger.info("possible_solution: {}", possible_solution)
                example_json = {}
                api_callings = {}
                parameters_json = {}
                success_history_code = '\n'.join([i['code'] for i in self.executor.execute_code if i['success']=='True'])
                last_success_index = max(idx for idx, item in enumerate(self.executor.execute_code) if item['success'] == 'True')
                error_code_after_last_success = '\n'.join(
                    [i['code'] for i in self.executor.execute_code[last_success_index + 1:] if i['success'] == 'False']
                )
                error_code_with_info = '\n'.join(
                    [i['code'] + ', its traceback:' + i.get('traceback', '') + '\n' for i in self.executor.execute_code[last_success_index + 1:] if i['success'] == 'False']
                )
                #error_code = '\n'.join([i['code'] for i in self.executor.execute_code if i['success']=='False'])
                error_code = error_code_after_last_success
                self.logger.info("success_history_code: {}, error_code: {}", success_history_code, error_code)
                whole_code = success_history_code+'\n' + error_code
                # get all previous API in the code, then collect the examples and put into prompt
                imports = extract_imports(whole_code)
                ori_relevant_API, relevant_API = get_sub_API(error_code, imports, self.LIB) # ALIAS
                for api in relevant_API:
                    if api in self.API_composite:
                        example = self.API_composite[api]['example']
                        if example:
                            example_json[api] = example
                        api_callings[api] = generate_function_signature(api, self.API_composite[api]['Parameters'])
                        parameters_json[api] = self.API_composite[api]['Parameters']
                    else:
                        self.logger.error('there exist error that some APIs are not in API_init.json')
                # collect parameters and put into prompt
                api_docstring = 'def '+generate_function_signature(self.predicted_api_name, self.API_composite[self.predicted_api_name]['Parameters'])+':\n"""'+self.API_composite[self.predicted_api_name]['Docstring'] + '"""'
                execution_prompt = self.prompt_factory.create_prompt('executor_correction', api_docstring, json.dumps({str(key): str(value) for key, value in self.executor.variables.items() if value['type'] not in ['function', 'module', 'NoneType']}), error_code_with_info, possible_solution, json.dumps(example_json), success_history_code, self.user_query)
                response, _ = LLM_response(execution_prompt, self.model_llm_type, history=[], kwargs={})  # llm
                self.logger.info('execution_prompt: {}, response: {}', execution_prompt, response)
                tmp_retry_count = 0
                while tmp_retry_count<5:
                    tmp_retry_count+=1
                    try:
                        clean_response = response.replace('```json', '').replace('```', '').strip()
                        newer_code = json.loads(clean_response)['code']
                        newer_analysis = json.loads(clean_response)['analysis']
                        if ast.parse(newer_code): # must be valid code
                            break
                    except Exception as e:
                        self.logger.error('Error: {}', e)
                        newer_code = ""
                        newer_analysis = ""
                self.logger.info('clean_response: {}, newer_code: {}, newer_analysis: {}', clean_response, newer_code, newer_analysis)
                #newer_code = response.replace('\"\"\"', '')
                if newer_analysis:
                    self.callback_func('log', newer_analysis, "Error Analysis," +" retry count: "+str(self.retry_execution_count) + "/"+str(self.retry_execution_limit))
                if newer_code:
                    self.execution_code = newer_code
                    self.callback_func('code', self.execution_code, "Executed code")
                else:
                    # TODO: should return to another round
                    self.callback_func('log', "LLM didn't correct code as we expected.", "Execution correction [Fail]")
                # LLM response
                api_docstring = 'def '+generate_function_signature(self.predicted_api_name, self.API_composite[self.predicted_api_name]['Parameters'])+':\n"""'+self.API_composite[self.predicted_api_name]['Docstring'] + '"""'
                summary_prompt = self.prompt_factory.create_prompt('summary_full', user_input, api_docstring, self.execution_code)
                response, _ = LLM_response(summary_prompt, self.model_llm_type, history=[], kwargs={})
                self.callback_func('log', response, "Task summary")
                self.callback_func('log', "Could you confirm whether this task is what you aimed for, and the code should be executed?\nEnter [y]: Go on please\nEnter [n]: Re-direct to the parameter input step\nEnter [r]: Restart another turn", "User Confirmation")
                self.update_user_state("run_pipeline_after_doublechecking_execution_code")
                self.save_state_enviro()
                return 
            else:
                #self.callback_func('log', "The execution failed multiple times. Please re-enter the inquiry and start a new turn.", "Executed results [Fail]")
                self.callback_func('log', "The execution failed multiple times. Please re-enter the inquiry for current task, we will try again and continue the remaining subtasks.", "Executed results [Fail]")
                self.retry_execution_count = 0
                self.update_user_state("run_pipeline")
                self.save_state_enviro()
                return
            
        self.save_state_enviro()
        if self.last_execute_code['code'] and self.last_execute_code['success']=='True':
            # split tuple variable into individual variables
            ans, new_code = self.executor.split_tuple_variable(self.last_execute_code) # This function verifies whether the new variable is a tuple.
            if ans:
                self.callback_func('code', new_code, "Executed code")
                self.callback_func('log', "Splitting the returned tuple variable into individual variables", "Executed results [Success]")
            else:
                pass
        else:
            pass
        new_str = []
        for i in self.executor.execute_code:
            new_str.append({"code":i['code'],"execution_results":i['success']})
        self.logger.info("namespace vari: {}, Currently all executed code: {}", json.dumps(list(self.executor.variables.keys())), json.dumps(new_str))
        # 240604 add logic, if there exist sub task in self.user_query_list, then do not return, go ahead to the next sub task
        if self.user_query_list:
            #self.callback_func('log', "Remaining subtasks: \n → "+ '\n  - '.join(self.user_query_list), "Continue to the next subtask")
            self.callback_func('log', "Ongoing subtask and remaining subtasks: \n → "+ '\n - '.join(self.user_query_list), "Task Planning")
            sub_task = self.get_query()
            self.update_user_state("run_pipeline")
            self.save_state_enviro()
            self.run_pipeline(sub_task, self.LIB, top_k=3, files=[],conversation_started=False,session_id=self.session_id)
            return
        else:
            self.callback_func('log', "All subtasks are predicted and executed.", "Start another round.")
            self.new_task_planning = True
            self.user_query_list = []
        self.update_user_state("run_pipeline")
        self.save_state_enviro()
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
            self.logger.info('{}, {}', str(code), str(ans))
            if ans:
                output_list.append(ans)
            if plt.get_fignums()!=self.plt_status:
                output_list.append(self.executor.execute_api_call("from src.inference.utils import save_plot_with_timestamp", "import"))
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
        self.logger.info('Something wrong with getting execution status by code! Enter wrong code {}', code)
        return None
    def add_query(self, queries, mode='aft'):
        if mode=='aft':
            self.user_query_list.extend(queries)
        else:
            self.user_query_list = queries + self.user_query_list
    def get_query(self):
        if self.user_query_list:
            return self.user_query_list.pop(0)
        else:
            self.initialize_tool()
            self.callback_func('log', "There is no subtask for code generation, please re-enter your inquiry", "Start another round")
            self.update_user_state("run_pipeline")
            self.save_state_enviro()
            return None

