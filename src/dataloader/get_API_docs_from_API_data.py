import os
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

if __name__ == "__main__":
    INPUT_DIR = os.path.join('data','standard_process','bioservices')
    INPUT_FILE = os.path.join(INPUT_DIR, f"API_data_bigg.json")
    api_data = load_json(INPUT_FILE)
    OUTPUT_DIR = os.path.join('data','standard_process','bioservices')
    OUTPUT_API_INIT = os.path.join(OUTPUT_DIR, f"API_init_bigg.json")
    OUTPUT_API_COMPOSITE = os.path.join(OUTPUT_DIR, f"API_composite_bigg.json")
    api_init = preprocess_API_data_into_API_init(api_data)
    save_json(OUTPUT_API_INIT, api_init)
    save_json(OUTPUT_API_COMPOSITE, api_init)
