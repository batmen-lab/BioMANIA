import base64, ast, os, requests, subprocess
from datetime import datetime
import numpy as np
from ..inference.utils import sentence_transformer_embed, predict_by_similarity
from urllib.parse import urlparse
import re, json

basic_types = ['str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set', 'List', 'Dict', 'Any', 'any', 'Path', 'path', 'Pathlike']
basic_types.extend(['_AvailShapes']) # extend for squidpy `shape` type

special_types = {'AnnData', 'ndarray', 'spmatrix', 'DataFrame', 'recarray', 'Axes'}
io_types = {'PathLike', 'Path', 'path'}
io_param_names = {'filename'}

# post_process_parsed_params, parse_json_safely are saved in param_count_acc_just_test.py
def correct_param_type(param_name, param_value, api_data, api_name):
    if param_name not in api_data[api_name]['Parameters']:
        return param_value
    param_type = api_data[api_name]['Parameters'][param_name]['type']
    if param_type in [None, "None"]:
        return param_value
    if 'List' in param_type or 'list' in param_type:
        if param_value.startswith('(') and param_value.endswith(')'):
            param_value = param_value.replace('(', '[').replace(')', ']')
        # Convert string representation of numbers to list
        elif re.match(r'^\d+(,\d+)*$', param_value):
            param_value = f'[{param_value}]'
    elif 'Tuple' in param_type or 'tuple' in param_type:
        if param_value.startswith('[') and param_value.endswith(']'):
            param_value = param_value.replace('[', '(').replace(']', ')')
        # Convert string representation of numbers to tuple
        elif re.match(r'^\d+(,\d+)*$', param_value):
            param_value = f'({param_value})'
    if 'Iterable' in param_type or 'List' in param_type or 'Tuple' in param_type:
        if not param_value.startswith('[') and not param_value.startswith('('):
            param_value = '[' + param_value + ']'
    return param_value

def generate_api_calling(api_name, api_details, predicted_parameters):
    """
    Generates an API call and formats output based on provided API details and returned content string.
    """
    parameters_dict = {}
    parameters_info_list = []
    for param_name, param_details in api_details['Parameters'].items():
        # only include required parameters and optional parameters found from response, and a patch for color in scanpy/squidpy pl APIs
        if (param_name in predicted_parameters) or (not param_details['optional']) or (param_name=='color' and (api_name.startswith('scanpy.pl') or api_name.startswith('squidpy.pl'))) or (param_name=='encodings' and (api_name.startswith('ehrapy.pp') or api_name.startswith('ehrapy.preprocessing'))) or (param_name=='encoded' and (api_name.startswith('ehrapy.'))):
            #print(param_name, param_name in predicted_parameters, not param_details['optional'])
            param_type = param_details['type']
            if param_type in [None, 'None', 'NoneType']:
                param_type = "Any"
            param_description = param_details['description']
            param_value = param_details['default']
            param_optional = param_details['optional']
            if predicted_parameters:
                if param_name in predicted_parameters:
                    param_value = predicted_parameters[param_name]
                    #if param_type is not None and ('str' in param_type or 'PathLike' in param_type):
                    #    if ('"' not in param_type and "'" not in param_type) and (param_value not in ['None', None]):
                    #        param_value = "'"+str(param_value)+"'"
            # added condition to differentiate between basic and non-basic types
            if any(item in param_type for item in basic_types):
                param_value = param_value if ((param_value not in [None,  'None']) and param_value) else "@"
            # add some general rules for basic types.
            elif ('openable' in param_type) or ('filepath' in param_type) or ('URL' in param_type):
                param_value = param_value if ((param_value not in [None, 'None']) and param_value) else "@"
            else:
                param_value = param_value if ((param_value not in [None,  'None']) and param_value) else "$"
            parameters_dict[param_name] = param_value
            parameters_info_list.append({
                'name': param_name,
                'type': param_type,
                'value': param_value,
                'description': param_description,
                'optional': param_optional
            })
    parameters_str = ", ".join(f"{k}={v}" for k, v in parameters_dict.items())
    if api_details['api_type'] not in ['property', 'constant']:
        api_calling = f"{api_name}({parameters_str})"
    else:
        api_calling = f"{api_name}"
    output = {
        "api_name": api_name,
        "parameters": {
            param['name']: {
                "type": param['type'],
                "description": param['description'],
                "value": format_string_list(str(param['value']), "[", "]") if '[' in str(param['value']) else format_string_list(str(param['value']), "(", ")") if '(' in str(param['value']) else str(param['value']), # TODO: 240519, add patches if gpt response is not formatted correctly
                "optional": param['optional']
            } for param in parameters_info_list
        },
        "return_type": api_details['Returns']['type']
    }
    return api_name, api_calling, output

def format_string_list(input_string, left_bracket="[", right_bracket="]"):
    # Safely evaluate the string to convert it into a Python list
    # ast.literal_eval is used here for safety to prevent execution of arbitrary code
    try:
        # Convert the input string to a valid list string by adding quotes around elements
        safe_input = input_string.replace(left_bracket, f'{left_bracket}"').replace(right_bracket, f'"{right_bracket}').replace(', ', '", "')
        data_list = ast.literal_eval(safe_input)
    except Exception as e:
        # f"Error parsing input: {str(e)}"
        return input_string
    formatted_list = [f"'{item}'" if isinstance(item, str) and "'" not in item else str(item) for item in data_list]
    # Convert list back to string representation
    result_string = left_bracket + ", ".join(formatted_list) + right_bracket
    return result_string

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
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))])) + [sentence_transformer_embed, predict_by_similarity, urlparse, basic_types, special_types, io_types, io_param_names
]
