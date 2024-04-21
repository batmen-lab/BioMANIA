import json
import os, datetime
import matplotlib.pyplot as plt

from collections import defaultdict
from gpt.utils import get_all_api_json, find_similar_api_pairs, is_pair_in_merged_pairs, find_similar_two_pairs, get_ambiguous_pairs
import numpy as np

def json_to_docstring(api_name, description, parameters):
    params_list = ', '.join([
        f"{param}: {parameters[param]['type']} = {parameters[param]['default']}" if parameters[param]['optional'] else f"{param}: {parameters[param]['type']}"
        for param in parameters
    ])
    function_signature = f"def {api_name}({params_list}):"
    docstring = f"\"\"\"{description}\n\n"
    if len(parameters) > 0:
        docstring += "Parameters\n----------\n"
    for parameter in parameters:
        info = parameters[parameter]
        if info['description'] is not None:  # Skip empty descriptions
            if info['description'].strip():
                docstring += f"{parameter}\n    {info['description']}\n"
    docstring += "\"\"\""
    return function_signature + "\n" + docstring.strip()

def predict_by_similarity(user_query_vector, centroids, labels):
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = [cosine_similarity(user_query_vector, centroid.reshape(1, -1)) for centroid in centroids]
    return labels[np.argmax(similarities)]

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
from PIL import Image
#def compress_and_save_image(image_path, quality=85):
#    with Image.open(image_path) as img:
#        img.save(image_path, "PNG", optimize=True, quality=quality)
"""def compress_and_save_image(image_path, output_path=None, optimize=True, scale_factor=0.5):
    with Image.open(image_path) as img:
        if scale_factor != 1:
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img = img.resize(new_size, Image.ANTIALIAS)
        img = img.convert("P", palette=Image.ADAPTIVE)
        if not output_path:
            output_path = image_path
        img.save(output_path, "PNG", optimize=optimize)"""
def compress_and_save_image(image_path, output_path=None):
    if not output_path:
        output_path = image_path
    reader = png.Reader(image_path)
    w, h, pixels, metadata = reader.asDirect()
    output = open(output_path, 'wb')
    writer = png.Writer(w, h, greyscale=metadata['greyscale'], alpha=metadata['alpha'], bitdepth=8)
    writer.write_array(output, pixels)
    output.close()
def save_plot_with_timestamp(folder="./tmp/images", prefix="img", format="webp", save_pdf=False):
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    # save 
    temp_png_path = os.path.join(folder, f"{prefix}_{timestamp}.png")
    plt.savefig(temp_png_path, bbox_inches='tight')
    # compress
    if format == 'webp':
        webp_path = os.path.join(folder, f"{prefix}_{timestamp}.webp")
        with Image.open(temp_png_path) as img:
            img.save(webp_path, 'WEBP')
        os.remove(temp_png_path)
        save_path = webp_path
    elif format == 'png':
        compress_and_save_image(save_path)
    else:
        save_path = temp_png_path
    if save_pdf:
        file_name = f"{prefix}_{timestamp}.pdf"
        save_path = os.path.join(folder, file_name)
        plt.savefig(save_path)
    return save_path

def compress_api_str_from_list_query_version(api):
    api_name = api['api_calling'][0].split('(')[0]
    api_desc_truncated = api['api_description'].split('\n')[0]
    req_params = json.dumps(api['required_parameters'])
    opt_params = json.dumps(api['optional_parameters'])
    return_schema = json.dumps(api['Returns'])
    #compressed_str = f"{api_name}, {api_desc_truncated}, required_params: {req_params}, optional_params: {opt_params}, return_schema: {return_schema}"
    compressed_str = f"{api_name}, {api_desc_truncated}, required_params: {req_params}, optional_params: {opt_params}"
    return compressed_str

def process_retrieval_desc(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in documents_df.itertuples():
        doc = json.loads(row.document_content)
        ir_corpus[row.docid] = compress_desc(doc)
        corpus2tool[row.docid] = doc['api_name']
    return ir_corpus, corpus2tool

def compress_desc(doc):
    api_description = doc['api_description'].split('\n')[0]
    compressed_str = f"{api_description}"
    return compressed_str

def process_retrieval_document_query_version(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in documents_df.itertuples():
        doc = json.loads(row.document_content)
        ir_corpus[row.docid] = compress_api_str_from_list_query_version(doc)
        corpus2tool[row.docid] = doc['api_calling'][0].split('(')[0]
    return ir_corpus, corpus2tool

def compress_tut_version(doc):
    text = doc['text']
    code = doc['code']
    compressed_str = f"tutorial text: {text}, code: {code}"
    return compressed_str

def process_tut_version(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in documents_df.itertuples():
        doc = json.loads(row.document_content)
        ir_corpus[row.docid] = compress_tut_version(doc)
        corpus2tool[row.docid] = doc['relevant_API']
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
    #compressed_str = f"{api_name}, {api_desc_truncated}, required_params: {req_params}, return_schema: {return_schema}" # optional_params: {opt_params},
    compressed_str = f"API description: {api_desc_truncated}, required_params: {req_params}" # optional_params: {opt_params},
    return compressed_str 

def process_retrieval_document(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in documents_df.itertuples():
        doc = json.loads(row.document_content)
        ir_corpus[row.docid] = compress_api_str_from_list(doc)
        corpus2tool[row.docid] = doc['api_calling'][0].split('(')[0]
    return ir_corpus, corpus2tool

def get_all_types_in_API(LIB):
    filepath=f"./data/standard_process/{LIB}/API_composite.json"
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

def fast_get_environment(pre_code):
    from inference.execution_UI import CodeExecutor
    executor = CodeExecutor()
    executor.save_directory = './tmp'
    executor.execute_api_call(pre_code)
    executor.save_environment("pre_env.pkl")
    executor.save_variables_to_json()
    return executor

def sentence_transformer_embed(model, texts):
    embeddings = model.encode(texts, convert_to_tensor=True)
    embeddings = embeddings.cpu().detach().numpy()
    return embeddings

def bert_embed(model,tokenizer,text, device='cpu'):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(1).squeeze().detach().cpu().numpy()

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    a, b = get_all_api_json(f"./data/standard_process/scanpy/API_init.json")
    '''pre_code = """
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
    executor.execute_api_call('print(a)')'''

