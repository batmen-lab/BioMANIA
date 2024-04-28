import base64, ast, os, requests, subprocess
from datetime import datetime
import numpy as np
from ..inference.utils import sentence_transformer_embed, predict_by_similarity
from urllib.parse import urlparse

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

'''def save_decoded_file(encoded_file, filename):
    """Decode and save the base64 encoded file"""
    file_path = os.path.join('tmp', filename)
    with open(file_path, "wb") as file:
        file.write(base64.b64decode(encoded_file.split(",")[1]))
    return file_path'''

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

def infer(query, model, centroids, labels):
    # 240125 modified chitchat model
    user_query_vector = np.array([sentence_transformer_embed(model, query)])
    try:
        predicted_label = predict_by_similarity(user_query_vector, centroids, labels)
    except Exception as e:
        print(e)
    return predicted_label

def dataframe_to_markdown(df):
    headers = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for index, row in df.iterrows():
        rows.append("| " + " | ".join(row.values.astype(str)) + " |")
    table_markdown = "\n".join([headers, separator] + rows)
    return table_markdown

def convert_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        print("Converting image to Base64 successfully!")
        return base64_image
    except Exception as e:
        print("Error converting image to Base64:", str(e))
        return None

def change_format(input_params, param_name_list):
    """
    Get a subset of input parameters dictionary
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


import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))])) + [sentence_transformer_embed, predict_by_similarity, urlparse, basic_types]
