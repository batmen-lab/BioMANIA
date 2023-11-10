import os
import torch.cuda
import torch.backends
import platform
import json

LIB = 'squidpy' # qiime2 scikit-bio pyteomics scanpy

########### User Input
with open('./configs/Lib_cheatsheet.json', 'r') as file:
    USER_INPUT = json.load(file)
if LIB not in USER_INPUT:
    raise ValueError(f"'{LIB}' is not available in USER_INPUT. Please select a valid library.")
################

CURRENT_PATH = os.getcwd()
RESOURCES_PATH  = os.path.join(os.getcwd(),'..','..','resources')
READTHEDOC_PATH = os.path.join(RESOURCES_PATH,'readthedoc_files')
ANALYSIS_PATH = os.path.join(RESOURCES_PATH,'json_analysis')
GITHUB_PATH = os.path.join(RESOURCES_PATH,'github_code')

API_HTML = USER_INPUT[LIB]['API_HTML_PATH']
if API_HTML:
    API_HTML_PATH = os.path.normpath(os.path.join(READTHEDOC_PATH, API_HTML))
else:
    API_HTML_PATH = None
GITHUB_LINK = USER_INPUT[LIB]['GITHUB_LINK']
READTHEDOC_LINK = USER_INPUT[LIB]['READTHEDOC_LINK']
LIB_ALIAS = USER_INPUT[LIB]['LIB_ALIAS']
if USER_INPUT[LIB]['TUTORIAL_HTML_PATH']:
    TUTORIAL_HTML_PATH = os.path.normpath(os.path.join(READTHEDOC_PATH, USER_INPUT[LIB]['TUTORIAL_HTML_PATH']))
else:
    TUTORIAL_HTML_PATH = None
TUTORIAL_GITHUB = USER_INPUT[LIB]['TUTORIAL_GITHUB']

LIB_ANALYSIS_PATH = os.path.join(ANALYSIS_PATH,LIB)
if not os.path.exists(LIB_ANALYSIS_PATH):
    os.makedirs(LIB_ANALYSIS_PATH)
LIB_GITHUB_PATH = os.path.join(GITHUB_PATH,LIB)
if not os.path.exists(LIB_GITHUB_PATH):
    os.makedirs(LIB_GITHUB_PATH)

# hugging_models
if platform.system() == 'Linux':
    HUGGINGPATH = "/home/z6dong/BioChat/hugging_models"
elif platform.system() == 'Darwin':
    HUGGINGPATH = "/Users/doradong/hugging_models"
else:
    HUGGINGPATH = "/home/z6dong/BioChat/hugging_models"
hugging_model = {
    "chatglm-6b":"THUDM/chatglm-6b",
    "chatglm2-6b":"THUDM/chatglm2-6b",
}
llm_model_dict = {
    "openai":{
        "platform":"OPENAI",
    },
    "gorilla-7b-hf-v1":{
        "platform":"GORILLA",
    },
    "THUDM/chatglm2-6b": {
        "platform":"HUGGINGFACE",
        "LOCALPATH":os.path.join(HUGGINGPATH,'chatglm2-6b'),
    },
    "guanaco":{
        "platform":"PEFT",
        "LOCALPATH":None,
        "pretrained_model":""
    },
    "tiiuae/falcon-7b":{
        "platform":"HUGGINGFACE",
        "LOCALPATH":os.path.join(HUGGINGPATH,'falcon-7b'),
    },
    "llama-2-7b-chat-hf":{
        "platform":"PEFT",
        "LOCALPATH":os.path.join(HUGGINGPATH,'llama-2-7b-chat-hf'),
    },
}
LLM_MODEL = "openai"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.01
LLM_HISTORY_LEN = 20
FP16 = True
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-test')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Base cheatsheet
with open('./configs/Base_cheatsheet.json', 'r') as file:
    CHEATSHEET = json.load(file)
