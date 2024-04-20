import base64

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
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))
