import json
from inference.execution import CodeExecutor
import os, datetime
import matplotlib.pyplot as plt

from collections import defaultdict

def find_similar_two_pairs(lib_name):
    from collections import defaultdict
    with open(f"./data/standard_process/{lib_name}/API_init.json", "r") as file:
        api_data = json.load(file)
    api_data = {key:api_data[key] for key in api_data if api_data[key]['api_type']!='class'}
    # 1: description
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    def find_similar_api_pairs(api_descriptions):
        descriptions = list(api_descriptions.values())
        api_names = list(api_descriptions.keys())
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
        similar_pairs = []
        for i in range(len(api_names)):
            for j in range(i+1, len(api_names)):
                if cosine_similarities[i, j] > 0.999:  # threshold can be adjusted
                    similar_pairs.append((api_names[i], api_names[j]))
        return similar_pairs
    import re, os
    from string import punctuation
    end_of_docstring_summary = re.compile(r'[{}\n]+'.format(re.escape(punctuation)))
    all_apis = {x: end_of_docstring_summary.split(api_data[x]['Docstring'])[0].strip() for x in api_data}
    all_apis = list(all_apis.items())
    all_apis_json = {i[0]:i[1] for i in all_apis}
    #all_apis_json = {api_name:api_data[api_name]['Docstring'].split('.')[0] for api_name in api_data}
    similar_api_pairs = find_similar_api_pairs(all_apis_json)
    # 2: 
    require_same_depth=False
    api_list = list(api_data.keys())
    groups = defaultdict(list)
    for api in api_list:
        parts = api.split('.')
        if require_same_depth:
            key = (parts[-1], len(parts))
        else:
            key = parts[-1]
        groups[key].append(api)
    similar_pairs = [group for group in groups.values() if len(group) > 1]
    list_1 = similar_api_pairs
    list_2 = similar_pairs
    pairs_from_list_2 = [(apis[i], apis[j]) for apis in list_2 for i in range(len(apis)) for j in range(i+1, len(apis))]
    print('information of the ambiguous pair:', len(list_1), len(list_2), len(pairs_from_list_2))
    merged_pairs = list(set(list_1 + pairs_from_list_2))
    return merged_pairs

def find_similar_pairs(lib, require_same_depth=False):
    # find similar name pairs
    with open(f'./data/standard_process/{lib}/API_init.json', 'r') as file:
        data = json.load(file)
    api_list = list(data.keys())
    groups = defaultdict(list)
    for api in api_list:
        parts = api.split('.')
        if require_same_depth:
            key = (parts[-1], len(parts))
        else:
            key = parts[-1]
        groups[key].append(api)
    similar_pairs = [group for group in groups.values() if len(group) > 1]# Filter out groups that only contain 1 API (no similar pairs).
    for pair in similar_pairs:
        print(pair)
    return similar_pairs

if not os.path.exists("./tmp/images"):
    os.makedirs("./tmp/images", exist_ok=True)
def save_plot_with_timestamp(folder="./tmp/images", prefix="img", format="png"):
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    file_name = f"{prefix}_{timestamp}.{format}"
    save_path = os.path.join(folder, file_name)
    plt.savefig(save_path)
    return save_path

def compress_api_str_from_list_query_version(api):
    api_name = api['api_calling'][0].split('(')[0]
    api_desc_truncated = api['api_description'].split('\n')[0]
    req_params = json.dumps(api['required_parameters'])
    opt_params = json.dumps(api['optional_parameters'])
    return_schema = json.dumps(api['Returns'])
    compressed_str = f"{api_name}, {api_desc_truncated}, required_params: {req_params}, optional_params: {opt_params}, return_schema: {return_schema}"
    return compressed_str

def process_retrieval_document_query_version(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in documents_df.itertuples():
        doc = json.loads(row.document_content)
        ir_corpus[row.docid] = compress_api_str_from_list_query_version(doc)
        corpus2tool[compress_api_str_from_list_query_version(doc)] = doc['api_calling'][0].split('(')[0]
    return ir_corpus, corpus2tool

def compress_api_str_from_list(api):
    api_name = api['api_calling'][0].split('(')[0]
    api_desc_truncated = api['description'].split('\n')[0]
    if api['Parameters']:
        req_params = json.dumps({key:api['Parameters'][key] for key in api['Parameters'] if not api['Parameters'][key]['optional']})
        opt_params = json.dumps({key:api['Parameters'][key] for key in api['Parameters'] if api['Parameters'][key]['optional']})
    else:
        req_params = json.dumps({})
        opt_params = json.dumps({})
    return_schema = json.dumps(api['Returns'])
    compressed_str = f"{api_name}, {api_desc_truncated}, required_params: {req_params}, return_schema: {return_schema}" # optional_params: {opt_params},
    return compressed_str 

def process_retrieval_document(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in documents_df.itertuples():
        doc = json.loads(row.document_content)
        ir_corpus[row.docid] = compress_api_str_from_list(doc)
        corpus2tool[compress_api_str_from_list(doc)] = doc['api_calling'][0].split('(')[0]
    return ir_corpus, corpus2tool

def get_all_types_in_API(filepath="./data/standard_process/squidpy/API_composite.json"):
    with open(filepath, "r") as file:
        api_data = json.load(file)
    types = set()
    variables_list = []
    for api_name, api_details in api_data.items():
        for param_name, param_details in api_details.get("Parameters", {}).items():
            types.add(param_details["type"])
        #param_details = api_details.get("Returns", {})
        #if param_details:
        #    types.add(param_details["type"])
    return types
types = get_all_types_in_API(filepath="./data/standard_process/squidpy/API_composite.json")
print(types)

def fast_get_environment(pre_code):
    executor = CodeExecutor()
    executor.save_directory = './tmp'
    executor.execute_api_call(pre_code)
    executor.save_environment("pre_env.pkl")
    executor.save_variables_to_json()
    return executor

if __name__=='__main__':
    pre_code = """
import squidpy as sq\n
a = sq.datasets.four_i()\n
b = sq.datasets.visium()\n
arr = np.ones((100, 100, 3))\n
arr[40:60, 40:60] = [0, 0.7, 1]\n
img = sq.im.ImageContainer(arr, layer="img1")\n
    """
    executor = fast_get_environment(pre_code)
    executor.variables = {}
    executor.load_environment()
    executor.load_variables_to_json()
    print(executor.variables)
    executor.execute_api_call('print(a)')

