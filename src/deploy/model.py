from queue import Queue
import json, time, importlib, inspect, ast, os, random, io, sys, pickle, shutil, subprocess
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
from ..inference.execution_UI import CodeExecutor, find_matching_instance
from ..inference.retriever_finetune_inference import ToolRetriever
from ..prompt.promptgenerator import PromptFactory
from ..configs.Lib_cheatsheet import CHEATSHEET as LIB_CHEATSHEET
from ..deploy.utils import basic_types, generate_api_calling, download_file_from_google_drive, download_data, save_decoded_file, correct_bool_values, convert_bool_values, infer, dataframe_to_markdown, convert_image_to_base64, change_format, parse_json_safely, post_process_parsed_params, special_types, io_types, io_param_names
from ..models.dialog_classifier import Dialog_Gaussian_classification

class Model:
    def __init__(self, logger, device, model_llm_type="gpt-3.5-turbo-0125"): # llama3
        print('start initialization!')
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
        self.LIB = "scanpy"
        self.args_retrieval_model_path = f'./hugging_models/retriever_model_finetuned/{self.LIB}/assigned'
        self.shot_k = 5 # 5 seed examples for api prediction
        self.retry_execution_limit = 2
        self.retry_execution_count = 0
        self.path_info_list = ['path','Path','PathLike']
        self.args_top_k = 3
        self.param_llm_retry = 1
        self.predict_api_llm_retry = 3
        self.enable_multi_task = False
        self.session_id = ""
        #load_dotenv()
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-test')
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        self.initialize_executor()
        reset_result = self.reset_lib(self.LIB)
        if reset_result=='Fail':
            print('Reset lib fail! Exit the dialog!')
            return
        self.last_user_states = ""
        self.user_states = "run_pipeline"
        self.parameters_info_list = None
        self.image_folder = "./tmp/images/"
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs("./tmp", exist_ok=True)
        os.makedirs("./tmp/states/", exist_ok=True)
        os.makedirs("./tmp/sessions/", exist_ok=True)
        self.image_file_list = []
        self.image_file_list = self.update_image_file_list()
        #with open(f'./data/standard_process/{self.LIB}/vectorizer.pkl', 'rb') as f:
        #    self.vectorizer = pickle.load(f)
        with open(f'./data/standard_process/{self.LIB}/centroids.pkl', 'rb') as f:
            self.centroids = pickle.load(f)
        self.retrieve_query_mode = "random"
        self.get_all_api_json_cache(f"./data/standard_process/{self.LIB}/API_init.json", mode='single')
        print("Server ready")
    @lru_cache(maxsize=10)
    async def get_all_api_json_cache(self,path,mode):
        self.all_apis, self.all_apis_json = get_all_api_json(path, mode)
    async def load_multiple_corpus_in_namespace(self, ):
        #self.executor.execute_api_call(f"from data.standard_process.{self.LIB}.Composite_API import *", "import")
        # pyteomics tutorial needs these import libs
        self.executor.execute_api_call(f"import os, gzip, numpy as np, matplotlib.pyplot as plt", "import")
        #self.executor.execute_api_call(f"from urllib.request import urlretrieve", "import")
        #self.executor.execute_api_call(f"from pyteomics import fasta, parser, mass, achrom, electrochem, auxiliary", "import")
        self.executor.execute_api_call(f"import numpy as np", "import")
        self.executor.execute_api_call(f"np.seterr(under='ignore')", "import")
        self.executor.execute_api_call(f"import warnings", "import")
        self.executor.execute_api_call(f"warnings.filterwarnings('ignore')", "import")
    @lru_cache(maxsize=10)
    async def load_bert_model_cache(self, load_mode='unfinetuned_bert'):
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device) if load_mode=='unfinetuned_bert' else SentenceTransformer(f"./hugging_models/retriever_model_finetuned/{self.LIB}/assigned", device=self.device)
    async def compute_dialog_metrics(self,):
        annotate_path = f'data/standard_process/{self.LIB}/API_inquiry_annotate.json'
        annotate_data = load_json(annotate_path)
        info_json = get_all_variable_from_cheatsheet(self.LIB)
        LIB_ALIAS = info_json['LIB_ALIAS']
        self.dialog_p_threshold = 0.05
        data_source = "single_query_train"
        self.dialog_classifer = Dialog_Gaussian_classification(threshold=self.dialog_p_threshold)
        scores_train, outliers = self.dialog_classifer.compute_accuracy_filter_compositeAPI(self.LIB, self.retriever, annotate_data, self.args_top_k, name=data_source, LIB_ALIAS=LIB_ALIAS)
        self.dialog_mean, self.dialog_std = self.dialog_classifer.fit_gaussian(scores_train['rank_1'])
    def reset_lib(self, lib_name):
        return asyncio.run(self.async_reset_lib(lib_name))
    async def async_reset_lib(self, lib_name):
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
            tasks = [
                self.load_data(f"./data/standard_process/{lib_name}/API_composite.json"),
                self.load_bert_model_cache(),
                self.load_retriever(lib_name, retrieval_model_path),
                self.load_multiple_corpus_in_namespace(),
                self.get_all_api_json_cache(f"./data/standard_process/{self.LIB}/API_init.json", mode='single')
            ]
            await asyncio.gather(*tasks)
            with open(f'./data/standard_process/{self.LIB}/centroids.pkl', 'rb') as f:
                self.centroids = pickle.load(f)
            self.executor.execute_api_call(f"import {lib_name}", "import"),
            self.logger.info("==>Successfully loading model! loading model cost: {} s", str(time.time()-t1))
            reset_result = "Success"
            self.LIB = lib_name
            # compute the dialog metrics
            await self.compute_dialog_metrics()
        except Exception as e:
            e = traceback.format_exc()
            self.logger.error("Error: {}", e)
            reset_result = "Fail"
            self.callback_func('log', f"Something wrong with loading data and model! \n{e}", "Setting error")
        return reset_result
    async def load_retriever(self, lib_name, retrieval_model_path):
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
    @lru_cache(maxsize=10)
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
    @lru_cache(maxsize=10)
    async def load_data(self, API_file):
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
        file_name = f"./tmp/states/{a}_state.pkl"
        state = {k: v for k, v in self.__dict__.copy().items() if self.executor.is_picklable(v) and k != 'executor'}
        with open(file_name, 'wb') as file:
            pickle.dump(state, file)
        self.logger.info("State saved to {}", file_name)
    @lru_cache(maxsize=10)
    def load_state(self, session_id):
        a = str(session_id)
        file_name = f"./tmp/states/{a}_state.pkl"
        with open(file_name, 'rb') as file:
            state = pickle.load(file)
        self.__dict__.update(state)
        self.logger.info("State loaded from {}", file_name)
    def run_pipeline(self, user_input, lib, top_k=3, files=[],conversation_started=True,session_id=""):
        self.initialize_tool()
        self.indexxxx = 2
        if True:
            self.session_id = session_id
            try:
                self.load_state(session_id)
                a = str(self.session_id)
                self.executor.load_environment(f"./tmp/sessions/{a}_environment.pkl")
            except Exception as e:
                e = traceback.format_exc()
                self.logger.error(e)
                self.initialize_executor()
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
            self.logger.info('==>new conversation_started!')
            self.user_states="run_pipeline"
            self.initialize_executor()
            for var_name in list(globals()):
                if var_name.startswith('result_') or (var_name.endswith('_instance')):
                    del globals()[var_name]
            for var_name in list(locals()):
                if var_name.startswith('result_') or (var_name.endswith('_instance')):
                    del locals()[var_name]
        else:
            self.logger.info('==>old conversation_continued!')
        if self.user_states == "run_pipeline":
            self.logger.info('start initial!')
            while not self.queue.empty():
                self.queue.get()
            self.loading_data(files)
            self.query_id += 1
            self.user_query = user_input
            # chitchat prediction
            predicted_source = infer(self.user_query, self.bert_model, self.centroids, ['chitchat-data', 'topical-chat', 'api-query'])
            self.logger.info('----query inferred as {}----', predicted_source)
            if predicted_source!='api-query':
                response, _ = LLM_response(user_input, self.model_llm_type, history=[], kwargs={})  # llm
                self.callback_func('log', response, "Non API chitchat")
                return
            else:
                pass
            # dialog prediction
            pred_class = self.dialog_classifer.single_prediction(user_input, self.retriever, self.args_top_k)
            self.logger.info('----query inferred as {}----', pred_class)
            if self.enable_multi_task:
                if pred_class not in ['single']:
                    self.logger.info('start multi-task!')
                    prompt = self.prompt_factory.create_prompt("multi_task", user_input, files)
                    #TODO: try with max_trials times
                    response, _ = LLM_response(prompt, self.model_llm_type, history=[], kwargs={})
                    try:
                        steps_list = ast.literal_eval(response)
                    except:
                        try:
                            steps_list = json.loads(response)
                        except:
                            try:
                                steps_list = re.findall(r'\"(.*?)\"', response)
                            except:
                                steps_list = []
                    if not steps_list:
                        self.callback_func('log', "LLM can not return valid steps list, please redesign your prompt.", "LLM predict Error")
                        return
                    # TODO: retrieve API for each task, return a tuple of (task, retrieved_apis) and run the remain pipeline
                    for tmp_input in steps_list:
                        retrieved_names = self.retriever.retrieving(tmp_input, top_k=self.args_top_k)
                    pass
            # start retrieving names
            retrieved_names = self.retriever.retrieving(user_input, top_k=self.args_top_k)
            # produce prompt
            if self.retrieve_query_mode=='similar':
                instruction_shot_example = self.retriever.retrieve_similar_queries(user_input, shot_k=self.shot_k)
            else:
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
                            new_function_candidates = [f"{i}:{api}, description: "+self.all_apis_json[api].replace('\n',' ') for i, api in enumerate(tmp_retrieved_api_list)]
                            similar_queries += "function candidates:\n" + "\n".join(new_function_candidates) + '\n' + tmp_str + "\n---\n"
                instruction_shot_example = similar_queries
            api_predict_init_prompt = get_retrieved_prompt()
            retrieved_apis_prepare = ""
            for idx, api in enumerate(retrieved_names):
                retrieved_apis_prepare+=f"{idx}:" + api+", description: "+self.all_apis_json[api].replace('\n',' ')+"\n"
            api_predict_prompt = api_predict_init_prompt.format(query=user_input, retrieved_apis=retrieved_apis_prepare, similar_queries=instruction_shot_example)
            success = False
            for _ in range(self.predict_api_llm_retry):
                try:
                    response, _ = LLM_response(api_predict_prompt, self.model_llm_type, history=[], kwargs={})  # llm
                    self.logger.info('==>Ask LLM: {}\n==>LLM response: {}', api_predict_prompt, response)
                    # hack for if LLM answers this or that
                    """response = response.split(',')[0].split("(")[0].split(' or ')[0]
                    response = response.replace('{','').replace('}','').replace('"','').replace("'",'')
                    response = response.split(':')[0]# for robustness, sometimes llm will return api:description"""
                    response = correct_pred(response, self.LIB)
                    response = response.strip()
                    #self.logger.info('self.all_apis_json keys: {}', self.all_apis_json.keys())
                    if len(response.split(','))>1:
                        response = response.split(',')[0]
                    self.all_apis_json[response]
                    self.predicted_api_name = response
                    success = True
                    break
                except Exception as e:
                    e = traceback.format_exc()
                    self.logger.error('error during api prediction: {}', e)
            if not success:
                self.callback_func('log', "LLM can not return valid API name prediction, please redesign your prompt.", "LLM predict Error")
                return
            # if the predicted API is in ambiguous API list, then show those API and select one from them
            if self.predicted_api_name in self.ambiguous_api:
                filtered_pairs = [api_pair for api_pair in self.ambiguous_pair if self.predicted_api_name in api_pair]
                self.filtered_api = list(set(api for api_pair in filtered_pairs for api in api_pair))
                next_str = ""
                idx_api = 1
                for api in self.filtered_api:
                    if idx_api>1:
                        next_str+='\n'
                    next_str+=f"Candidate [{idx_api}]: {api}"
                    description_1 = self.API_composite[api]['Docstring'].split("\n")[0]
                    next_str+='\n'+description_1
                    idx_api+=1
                next_str+="\n"+f"Candidate [-1]: No appropriate candidate, restart another inquiry by input -1"
                
                self.update_user_state("run_pipeline_after_ambiguous")
                self.callback_func('log', next_str, f"Can you confirm which of the following {len(self.filtered_api)} candidates")
                self.save_state_enviro()
            else:
                self.update_user_state("run_pipeline_after_fixing_API_selection")
                self.run_pipeline_after_fixing_API_selection(user_input)
        elif self.user_states == "run_pipeline_after_ambiguous":
            ans = self.run_pipeline_after_ambiguous(user_input)
            if ans in ['break']:
                return
            self.run_pipeline_after_fixing_API_selection(user_input)
        elif self.user_states in ["run_pipeline_after_doublechecking_execution_code", "run_pipeline_after_entering_params", "run_select_basic_params", "run_pipeline_after_select_special_params", "run_select_special_params", "run_pipeline_after_doublechecking_API_selection", "run_pipeline_asking_GPT"]:
            self.handle_state_transition(user_input)
        else:
            self.logger.error('Unknown user state: {}', self.user_states)
            raise ValueError
    def run_pipeline_asking_GPT(self,user_input):
        self.initialize_tool()
        self.logger.info('==>run_pipeline_asking_GPT')
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
            str(self.executor.variables),
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
        summary_prompt = self.prompt_factory.create_prompt('summary_full', self.user_query, self.predicted_api_name, self.API_composite[self.predicted_api_name]['description'], self.API_composite[self.predicted_api_name]['Parameters'],self.API_composite[self.predicted_api_name]['Returns'], self.execution_code)
        response, _ = LLM_response(summary_prompt, self.model_llm_type, history=[], kwargs={})
        self.callback_func('log', response, "Task summary before execution")
        self.callback_func('log', "Could you confirm whether this task is what you aimed for, and the code should be executed? Please enter y/n.\nIf you press n, we will re-direct to the parameter input step\nIf you press r, we will restart another turn", "Double Check")
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
        self.logger.info('==>run_pipeline_after_ambiguous')
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
        self.logger.info('==>run_pipeline_after_fixing_API_selection')
        # check if composite API/class method API, return the relevant APIs
        self.relevant_api_list = self.process_api_info(self.API_composite, self.predicted_api_name) # only contains predicted API
        self.api_name_json = self.check_and_insert_class_apis(self.API_composite, self.relevant_api_list)# also contains class API
        self.update_user_state("run_pipeline")
        api_description = self.API_composite[self.predicted_api_name]['description']
        # summary task
        summary_prompt = self.prompt_factory.create_prompt('summary', user_input, self.predicted_api_name, api_description, self.API_composite[self.predicted_api_name]['Parameters'],self.API_composite[self.predicted_api_name]['Returns'])
        response, _ = LLM_response(summary_prompt, self.model_llm_type, history=[], kwargs={})
        self.logger.info(f'summary_prompt: {summary_prompt}, summary_prompt response: {response}')
        self.callback_func('log', response, f"Predicted API: {self.predicted_api_name}")
        self.callback_func('log', "Could you confirm whether this API should be called? Please enter y/n.", "Double Check")
        self.update_user_state("run_pipeline_after_doublechecking_API_selection")
        self.save_state_enviro()
    
    def run_pipeline_after_doublechecking_API_selection(self, user_input):
        self.initialize_tool()
        self.logger.info('==>run_pipeline_after_doublechecking_API_selection')
        user_input = str(user_input)
        if user_input == 'n':
            self.update_user_state("run_pipeline")
            self.callback_func('log', "We will start another round. Could you re-enter your inquiry?", "Start another round")
            self.save_state_enviro()
            return
        elif user_input == 'y':
            pass
        else:
            self.callback_func('log', "The input was not y or n, please enter the correct value.", "Index Error")
            self.save_state_enviro()
            # user_states didn't change
            return
        #self.logger.info("==>Need to collect all parameters for a composite API")
        combined_params = {}
        self.logger.info('self.api_name_json: {}', self.api_name_json)
        # if the class API has already been initialized, then skip it
        executor_variables = {var_name: var_info["value"] for var_name, var_info in self.executor.variables.items() if str(var_info["value"]) not in ["None"]}
        for api in self.api_name_json:
            maybe_class_name = api.split('.')[-1]
            maybe_instance_name = maybe_class_name.lower() + "_instance"
            # 240520: modified, support for variable with none xx_instance name
            if self.API_composite[api]['api_type'] in ['class', 'unknown']:
                matching_instance, is_match = find_matching_instance(api, executor_variables)
                self.logger.info(f'executor_variables: {executor_variables}, api: {api}, matching_instance: {matching_instance}, is_match: {is_match}')
                if is_match:
                    maybe_instance_name = matching_instance
                    continue
            else:
                pass
            combined_params.update(self.API_composite[api]['Parameters'])
        parameters_name_list = [key for key, value in combined_params.items() if (key not in self.path_info_list)] # if (not value['optional'])
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
        try:
            tmp_api_parameters_information = self.API_composite[apis_name]['Parameters']
            api_docstring = json_to_docstring(apis_name, apis_description, tmp_api_parameters_information)###TODO: here only works for one api, if we add compositeAPI or classAPI in the future, we need to buildup a parameters selection for multiple API!!!
            parameters_prompt = self.prompt_factory.create_prompt('parameters',self.user_query, api_docstring, parameters_name_list)
        except Exception as e:
            e = traceback.format_exc()
            self.logger.error('error for parameters: {}', e)
        if len(parameters_name_list)==0:
            # if there is no required parameters, skip using LLM
            response = "[]"
            predicted_parameters = {}
        else:
            success = False
            for _ in range(self.param_llm_retry):
                try:
                    response, _ = LLM_response(parameters_prompt, self.model_llm_type, history=[], kwargs={})
                    self.logger.info('==>Asking LLM: {}, ==>LLM response: {}', parameters_prompt, response)
                    returned_content_str_new = response.replace('null', 'None').replace('None', '"None"')
                    # 240519 fix
                    pred_params, success = parse_json_safely(returned_content_str_new)
                    predicted_parameters = post_process_parsed_params(pred_params, apis_name, self.API_composite)
                except Exception as e:
                    e = traceback.format_exc()
                    self.logger.error('error during parameters prediction: {}', e)
                    pass
            if not success:
                self.callback_func('log', "LLM can not return valid parameters prediction, please redesign prompt in backend if you want to predict parameters. We will skip parameters prediction currently", "LLM predict Error")
                response = "{}"
                predicted_parameters = {}
        # filter predicted_parameters
        required_param_list = [param_name for param_name, param_info in self.API_composite[apis_name]['Parameters'].items() if param_info['type'] in special_types or param_info['type'] in io_types or param_name in io_param_names]
        predicted_parameters = {key: value for key, value in predicted_parameters.items() if value not in [None, "None", "null"] or key in required_param_list}
        self.logger.info('after filtering, predicted_parameters: {}', predicted_parameters)
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
                    self.logger.info('executor_variables: {}', executor_variables)
                    try:
                        matching_instance, is_match = find_matching_instance(api, executor_variables)
                    except Exception as e:
                        e = traceback.format_exc()
                        self.logger.error('error during matching_instance: {}', e)
                    self.logger.info(f'matching_instance: {matching_instance}, is_match: {is_match}')
                    if is_match:
                        maybe_instance_name = matching_instance
                        fix_update = False
                else:
                    pass
            if fix_update:
                self.logger.info('fix_update')
                self.parameters_info_list['parameters'].update(tmp_class_parameters_info_list['parameters'])
        ####### infer parameters
        # $ param
        self.selected_params = self.executor.select_parameters(self.parameters_info_list['parameters'])
        self.logger.info("Automatically selected params for $, after selection the parameters are: {}", json.dumps(self.selected_params))
        # $ param if not fulfilled
        none_dollar_value_params = [param_name for param_name, param_info in self.selected_params.items() if param_info["value"] in ['$']]
        self.logger.info('none_dollar_value_params: {}', json.dumps(none_dollar_value_params))
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
        self.logger.info('==>run_select_special_params')
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
            #print('self.filtered_params: {}', json.dumps(self.filtered_params))
            self.save_state_enviro()
        else:
            self.callback_func('log', "The parameters candidate list is empty", "Error Enter Parameters: basic type", "red")
            self.save_state_enviro()
            raise ValueError

    def run_pipeline_after_select_special_params(self,user_input):
        if self.last_user_states == "run_select_special_params":
            self.selected_params = self.executor.makeup_for_missing_single_parameter_type_special(params = self.selected_params, param_name_to_update=self.last_param_name, user_input = user_input)
        # @ param
        self.logger.info('==> run_pipeline_after_select_special_params')
        none_at_value_params = [param_name for param_name, param_info in self.selected_params.items() if (param_info["value"] in ['@']) and (param_name not in self.path_info_list)]
        self.filtered_params = {key: value for key, value in self.parameters_info_list['parameters'].items() if (value["value"] in ['@']) and (key not in self.path_info_list)}
        self.filtered_pathlike_params = {key: value for key, value in self.parameters_info_list['parameters'].items() if (value["value"] in ['@']) and (key in self.path_info_list)}
        # TODO: add condition later: if uploading data files, 
        # avoid asking Path params, assign it as './tmp'
        if none_at_value_params: # TODO: add type PathLike
            self.logger.info('if exist non path, basic type parameters, start selecting parameters')
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
        self.logger.info('==>run_select_basic_params')
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
            for param_name, param_info in api_params.items():
                if param_name not in combined_params:
                    combined_params[param_name] = param_info
            combined_params = {param_name: param_info for param_name, param_info in api_params.items() if param_name not in combined_params}

            parameters_combined.append(combined_params)
        return parameters_combined

    def run_pipeline_after_entering_params(self, user_input):
        self.initialize_tool()
        if self.last_user_states == "run_select_basic_params":
            self.selected_params = self.executor.makeup_for_missing_single_parameter(params = self.selected_params, param_name_to_update=self.last_param_name, user_input = user_input)
        self.logger.info('==>run pipeline after entering parameters')
        self.update_user_state("run_pipeline")
        self.image_file_list = self.update_image_file_list()
        if self.filtered_pathlike_params:
            # add 'tmp' 
            self.selected_params.update({
                key: {
                    "type": param_info["type"],
                    "value": "./tmp",
                    "valuefrom": 'userinput',
                    "optional": param_info["optional"]
                }
                for key, param_info in self.filtered_pathlike_params.items()
            })
        self.logger.info('self.selected_params: {}', json.dumps(self.selected_params))
        # split parameters according to multiple API, or class/method API
        parameters_list = self.extract_parameters(self.api_name_json, self.API_composite, self.selected_params)
        self.logger.info('==>parameters_list: {}', json.dumps(parameters_list))
        extracted_params = self.split_params(self.selected_params, parameters_list)
        extracted_params_dict = {api_name: extracted_param for api_name, extracted_param in zip(self.api_name_json, extracted_params)}
        self.logger.info('==>self.api_name_json: {}, parameters_list: {}, extracted_params: {}, extracted_params_dict: {}',str(self.api_name_json), str(parameters_list), str(extracted_params), str(extracted_params_dict))
        api_params_list = []
        for idx, api_name in enumerate(self.api_name_json):
            #if self.api_name_json[api_name]['type'] in ['class', 'unknown']: # !
            #print('==>assume not start with class API: {}', api_name)
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
                self.logger.info('producing optional param card')
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
        summary_prompt = self.prompt_factory.create_prompt('summary_full', user_input, self.predicted_api_name, api_data_single['description'], api_data_single['Parameters'],api_data_single['Returns'], self.execution_code)        
        response, _ = LLM_response(summary_prompt, self.model_llm_type, history=[], kwargs={})
        self.callback_func('log', response, "Task summary before execution")
        self.callback_func('log', "Could you confirm whether this task is what you aimed for, and the code should be executed? Please enter y/n.\nIf you press n, we will re-direct to the parameter input step\nIf you press r, we will restart another turn", "Double Check")
        self.update_user_state("run_pipeline_after_doublechecking_execution_code")
        self.save_state_enviro()
        
    def run_pipeline_after_doublechecking_execution_code(self, user_input):
        self.initialize_tool()
        self.logger.info('==> run_pipeline_after_doublechecking_execution_code')
        # if check, back to the last iteration and status
        if user_input in ['y', 'n', 'r']:
            if user_input == 'n':
                self.update_user_state("run_pipeline_after_doublechecking_API_selection")#TODO: check if exist issue
                self.callback_func('log', "We will redirect to the parameters input", "Re-enter the parameters")
                self.save_state_enviro()
                self.run_pipeline_after_doublechecking_API_selection('y')
                return
            elif user_input == 'r':
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
            self.logger.info('process is alive!')
            time.sleep(1)
            with open(temp_output_file, 'r') as file:
                accumulated_output = file.read() ######?
                self.logger.info('accumulated_output: {}', accumulated_output)
                self.callback_func('log', accumulated_output, "Executing results", enhance_indexxxx=False)
        self.indexxxx+=1
        with open("./tmp/tmp_output_run_pipeline_execution_code_list.txt", 'r') as file:
            output_str = file.read()
            result = json.loads(output_str)
        code = result['code']
        output_list = result['output_list']
        self.executor.load_environment("./tmp/tmp_output_run_pipeline_execution_code_variables.pkl")
        self.logger.info('check: {}, {}, {}, {}', code, output_list, self.executor.execute_code, self.executor.variables)
        
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
            except:
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
                    if print_type=='AnnData':
                        self.logger.info('if the new variable is of type AnnData, ')
                        visual_attr_list = [i_tmp for i_tmp in list(dir(print_value)) if not i_tmp.startswith('_')]
                        #if len(visual_attr_list)>0:
                        if 'obs' in visual_attr_list:
                            visual_attr = 'obs'#visual_attr_list[0]
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
                        self.logger.info('if exist table, visualize it')
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
            self.logger.info('if generate image, visualize it')
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
                self.callback_func('log', "\n".join(list(set(output_list))), "Executed results [Fail]")
            except:
                self.callback_func('log', content, "Executed results [Fail]")
            if self.retry_execution_count<self.retry_execution_limit:
                self.retry_execution_count +=1
                self.callback_func('log', "retry count: "+str(self.retry_execution_count), "Retry Execution by LLM")
                # 240521: automatically regenerated code by LLM
                prompt = self.prompt_factory.create_prompt('execution_correction', self.user_query, str(self.executor.execute_code), self.last_execute_code['code'], content, str(self.executor.variables), self.LIB)
                response, _ = LLM_response(prompt, self.model_llm_type, history=[], kwargs={})  # llm
                newer_code = response.replace('\"\"\"', '')
                self.execution_code = newer_code
                self.callback_func('code', self.execution_code, "Executed code")
                # LLM response
                summary_prompt = self.prompt_factory.create_prompt('summary_full', user_input, self.predicted_api_name, self.API_composite[self.predicted_api_name]['description'], self.API_composite[self.predicted_api_name]['Parameters'],self.API_composite[self.predicted_api_name]['Returns'], self.execution_code)
                response, _ = LLM_response(summary_prompt, self.model_llm_type, history=[], kwargs={})
                self.callback_func('log', response, "Task summary before execution")
                self.callback_func('log', "Could you confirm whether this task is what you aimed for, and the code should be executed? Please enter y/n.\nIf you press n, we will re-direct to the parameter input step\nIf you press r, we will restart another turn", "Double Check")
                self.update_user_state("run_pipeline_after_doublechecking_execution_code")
                self.save_state_enviro()
                return 
            else:
                self.callback_func('log', "The execution failed multiple times. Please re-enter the inquiry and start a new turn.", "Executed results [Fail]")
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
        self.logger.info("Show current variables in namespace: {}", json.dumps(list(self.executor.variables.keys())))
        new_str = []
        for i in self.executor.execute_code:
            new_str.append({"code":i['code'],"execution_results":i['success']})
        self.logger.info("Currently all executed code: {}", json.dumps(new_str))
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

