import os
import torch.cuda
import torch.backends
import platform

LIB = 'biotite' # eletoolkit scvi-tools qiime2 scikit-bio pyteomics scanpy squidpy scenicplus pyopenms biopython deap biotite
# emperor gneiss tskit MIOSTONE

CURRENT_PATH = os.getcwd()
RESOURCES_PATH  = os.path.join(os.getcwd(),'..','..','resources')
READTHEDOC_PATH = os.path.join(RESOURCES_PATH,'readthedoc_files')
ANALYSIS_PATH = os.path.join(RESOURCES_PATH,'json_analysis')
GITHUB_PATH = os.path.join(RESOURCES_PATH,'github_code')

def get_all_variable_from_cheatsheet(LIB):
    from ..configs.Lib_cheatsheet import CHEATSHEET
    USER_INPUT = CHEATSHEET
    if LIB not in USER_INPUT:
        raise ValueError(f"'{LIB}' is not available in USER_INPUT. Please select a valid library.")
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
    info_json = {
        "API_HTML": API_HTML,
        "API_HTML_PATH": API_HTML_PATH,
        "GITHUB_LINK": GITHUB_LINK,
        "READTHEDOC_LINK": READTHEDOC_LINK,
        "LIB_ALIAS": LIB_ALIAS,
        "TUTORIAL_HTML_PATH": TUTORIAL_HTML_PATH,
        "TUTORIAL_GITHUB": TUTORIAL_GITHUB,
        "LIB_ANALYSIS_PATH": LIB_ANALYSIS_PATH,
        "LIB_GITHUB_PATH": LIB_GITHUB_PATH,
        "TUTORIAL_HTML": USER_INPUT[LIB]['TUTORIAL_HTML_PATH'],
        "LIB_DATA_PATH": os.path.join('data','standard_process',LIB),
        "BASE_DATA_PATH": os.path.join('data','standard_process','base'),
    }
    return info_json

info_json = get_all_variable_from_cheatsheet(LIB)
API_HTML, API_HTML_PATH, GITHUB_LINK, READTHEDOC_LINK, LIB_ALIAS, TUTORIAL_HTML_PATH, TUTORIAL_GITHUB, LIB_ANALYSIS_PATH, LIB_GITHUB_PATH, TUTORIAL_HTML, LIB_DATA_PATH, BASE_DATA_PATH = [info_json[key] for key in ['API_HTML', 'API_HTML_PATH', 'GITHUB_LINK', 'READTHEDOC_LINK', 'LIB_ALIAS', 'TUTORIAL_HTML_PATH', 'TUTORIAL_GITHUB', 'LIB_ANALYSIS_PATH', 'LIB_GITHUB_PATH', 'TUTORIAL_HTML', 'LIB_DATA_PATH', 'BASE_DATA_PATH']]

def get_all_basic_func_from_cheatsheet():
    # Base cheatsheet
    from ..configs.Base_cheatsheet import CHEATSHEET
    return CHEATSHEET

# hugging_models
if platform.system() == 'Linux':
    HUGGINGPATH = "./hugging_models"
elif platform.system() == 'Darwin':
    HUGGINGPATH = "./hugging_models"
else:
    HUGGINGPATH = "./hugging_models"

EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_NEW_TOKENS = 800
TEMPERATURE = 0.01
LLM_HISTORY_LEN = 20
FP16 = True
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-test')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
