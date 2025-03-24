import os
from copy import deepcopy
from ..dataloader.get_API_data_using_gpt import MODULE_NAME_TO_CLASS_NAME
from ..gpt.utils import load_json, save_json


# Borrowed from src.dataloader.get_API_init_from_sourcecode.generate_api_callings
def get_optional_value(parameter: dict) -> bool:
    basic_types = ["int", "float", "str", "bool", "list", "tuple", "dict", "set", "None"]
    param_type = parameter.get("type")
    if param_type is not None:  # Add this line to check for None
        return not any(basic_type in param_type for basic_type in basic_types)
    else:
        return True  # or False, depending on your default logic

def preprocess_API_data_into_API_init(api_data: dict) -> dict:
    api_init = {}
    for module_name in api_data:
        for api_name in api_data[module_name]:
            class_name = MODULE_NAME_TO_CLASS_NAME[module_name]
            full_api_name = f"bioservices.{class_name}.{api_name}"
            api_init[full_api_name] = {
                "Parameters": {},
                "Returns": {},
                "Docstring": "",
                "description": "",
                "example": "",
                "api_type": "",
                "api_calling": [],
                "relevant APIs": [],
                "type": "singleAPI"
            }
            for parameter in api_data[module_name][api_name]["Parameters"]:
                api_init[full_api_name]["Parameters"][parameter["name"]] = {
                    "type": parameter["type"],
                    "default": parameter["default"],
                    "optional": parameter["optional"],
                    "description": parameter["description"],
                    "optional_value": get_optional_value(parameter),
                }
            api_init[full_api_name]["Returns"] = api_data[module_name][api_name]["Returns"]
            api_init[full_api_name]["Docstring"] = api_data[module_name][api_name]["Docstring"]
            api_init[full_api_name]["description"] = api_data[module_name][api_name]["Docstring"]
            api_init[full_api_name]["example"] = api_data[module_name][api_name]["example"]
            api_init[full_api_name]["api_type"] = api_data[module_name][api_name]["api_type"]
            api_init[full_api_name]["api_calling"] = [api_data[module_name][api_name]["api_calling"]]
    return api_init

def process_API_data_into_API_inquiry(api_data: dict) -> tuple[list, list]:
    api_inquiry = []
    api_inquiry_annotate = []
    query_id = 0
    for module_name in api_data:
        for api_name in api_data[module_name]:
            class_name = MODULE_NAME_TO_CLASS_NAME[module_name]
            full_api_name = f"bioservices.{class_name}.{api_name}"
            api_definition = {
                "Parameters": {},
                "Returns": {},
                "Docstring": "",
                "description": "",
                "example": "",
                "api_type": "",
                "api_calling": [],
                "relevant APIs": [],
                "type": "singleAPI",
                "query": "",
                "query_code": "",
                "query_id": query_id,
            }
            for parameter in api_data[module_name][api_name]["Parameters"]:
                api_definition["Parameters"][parameter["name"]] = {
                    "type": parameter["type"],
                    "default": parameter["default"],
                    "optional": parameter["optional"],
                    "description": parameter["description"],
                }
            api_definition["Returns"] = api_data[module_name][api_name]["Returns"]
            api_definition["Docstring"] = api_data[module_name][api_name]["Docstring"]
            api_definition["description"] = api_data[module_name][api_name]["Docstring"]
            api_definition["example"] = api_data[module_name][api_name]["example"]
            api_definition["api_type"] = api_data[module_name][api_name]["api_type"]
            api_definition["api_calling"] = [api_data[module_name][api_name]["api_calling"]]
            query_id += 1
            api_inquiry.append(api_definition)
            api_definition_annotate = deepcopy(api_definition)
            api_definition_annotate["api_name"] = full_api_name
            api_inquiry_annotate.append(api_definition_annotate)

    return api_inquiry, api_inquiry_annotate

if __name__ == "__main__":
    INPUT_DIR = os.path.join('data','standard_process','bioservices')
    INPUT_FILE = os.path.join(INPUT_DIR, f"API_data.json")
    api_data = load_json(INPUT_FILE)
    OUTPUT_DIR = os.path.join('data','standard_process','bioservices')
    OUTPUT_API_INIT = os.path.join(OUTPUT_DIR, f"API_init.json")
    OUTPUT_API_COMPOSITE = os.path.join(OUTPUT_DIR, f"API_composite.json")
    api_init = preprocess_API_data_into_API_init(api_data)
    save_json(OUTPUT_API_INIT, api_init)
    save_json(OUTPUT_API_COMPOSITE, api_init)
    OUTPUT_API_INQUIRY = os.path.join(OUTPUT_DIR, f"API_inquiry.json")
    OUTPUT_API_INQUIRY_ANNOTATE = os.path.join(OUTPUT_DIR, f"API_inquiry_annotate.json")
    api_inquiry, api_inquiry_annotate = process_API_data_into_API_inquiry(api_data)
    save_json(OUTPUT_API_INQUIRY, api_inquiry)
    save_json(OUTPUT_API_INQUIRY_ANNOTATE, api_inquiry_annotate)
