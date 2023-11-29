import pickle
import types
import pandas as pd
import importlib
import json
import inspect
import os

class CodeExecutor:
    def __init__(self):
        self.variables = {}
        self.save_directory = "./tmp"
        self.generate_code = []
        self.execute_code = []
        self.api_execution_count = 0  # Counter for executed APIs
        self.counter = 0
        self.callbacks = []
    def is_picklable(self, obj):
        """Check if an object is picklable."""
        try:
            pickle.dumps(obj)
            return True
        except Exception:
            return False
    def save_environment(self, env_name="environment.pkl") -> None:
        serializable_vars = {k: v for k, v in self.variables.items() if isinstance(v, (pd.DataFrame, pd.Series, int, float, str, bool))}
        with open(os.path.join(self.save_directory,env_name), "wb") as file:
            pickle.dump(serializable_vars, file)
    def load_environment(self, env_name="environment.pkl") -> None:
        with open(os.path.join(self.save_directory,env_name), "rb") as file:
            loaded_vars = pickle.load(file)
            self.variables.update(loaded_vars)
            globals().update(loaded_vars)
    def select_parameters(self, params):
        print('Start selecting parameters for $!')
        matching_params = {}
        if params:
            for param_name, param_info in params.items():
                if param_info["value"] not in ['$']:
                    matching_params[param_name] = {
                        "type": param_info["type"],
                        "value": param_info["value"],
                        "valuefrom": "value",
                        "optional": param_info["optional"],
                    }
                else:
                    param_type_str = param_info["type"]
                    possible_matches = [var_name for var_name, var_info in self.variables.items() if var_info["type"] in param_type_str] # change from == to in, since there might be Union[a,b]
                    if len(possible_matches) == 1:
                        matching_params[param_name] = {
                            "type": param_info["type"],
                            "value": possible_matches[0],
                            "valuefrom": 'match', 
                            "optional": param_info["optional"],
                        }
                    elif len(possible_matches) > 1:
                        """result_choices = [(int(choice.split('_')[1]), choice) for choice in possible_matches if choice.startswith('result_')]
                        if result_choices:
                            choice = max(result_choices, key=lambda x: x[0])[1]
                        else:
                            choice = possible_matches[0]"""
                        choice = possible_matches
                        matching_params[param_name] = {
                            "type": param_info["type"],
                            "value": choice,
                            "valuefrom": 'match',
                            "optional": param_info["optional"],
                        }
                    else:
                        matching_params[param_name] = {
                            "type": param_info["type"],
                            "value": param_info["value"],  # If not find matched parameters, set value as default
                            "valuefrom": 'userinput',
                            "optional": param_info["optional"],
                        }
        print('End selecting parameters for $!')
        return matching_params
    def makeup_for_missing_parameters(self, params, user_input):
        input_values = user_input.split(' ')
        input_iterator = iter(input_values)
        for param_name, param_info in params.items():
            if param_info["value"] in ['@']:
                try:
                    value = next(input_iterator)
                    #value = self.format_value(param_info['value'], param_info['type'])
                    params[param_name] = {
                        "type": param_info["type"],
                        "value": value,
                        "valuefrom": 'userinput',
                        "optional": param_info["optional"],
                    }
                except StopIteration:
                    print(f"Insufficient values provided in user_input for parameter '{param_name}'")
                    # You might want to handle this error more gracefully, depending on your requirements
        return params
    def makeup_for_missing_single_parameter(self, params, param_name_to_update, user_input, param_spec_type='@'):
        # Check if the given parameter name is valid and its value is '@'
        if param_name_to_update not in params or params[param_name_to_update]["value"] != param_spec_type:
            print(f"Invalid parameter name '{param_name_to_update}' or the parameter doesn't need a value.")
            return params
        param_info = params[param_name_to_update]
        value = user_input  # Since user_input is for a single parameter, we directly use it
        # Type conversion if needed
        value = self.format_value(value, param_info['type'])
        # Update the parameter information
        params[param_name_to_update] = {
            "type": param_info["type"],
            "value": value,
            "valuefrom": 'userinput',
            "optional": param_info["optional"],
        }
        return params
    def makeup_for_missing_single_parameter_type_special(self, params, param_name_to_update, user_input):
        # Check if the given parameter name is valid and its value is list type
        if param_name_to_update not in params or ('list' not in str(type(params[param_name_to_update]["value"]))):
            print(f"Invalid parameter name '{param_name_to_update}' or the parameter doesn't have multiple choice.")
            return params
        param_info = params[param_name_to_update]
        value = user_input  # Since user_input is for a single parameter, we directly use it
        # we didn't need to add '' for the value, its from namespace variable
        #value = self.format_value(value, param_info['type'])
        # Update the parameter information
        params[param_name_to_update] = {
            "type": param_info["type"],
            "value": value,
            "valuefrom": 'userinput',
            "optional": param_info["optional"],
        }
        return params
    def get_import_code(self, api_name):
        print(f'==>start importing code for {api_name}')
        if '.' not in api_name:
            return "", 'function'
        try:
            importlib.import_module(api_name)
            return f"import {api_name}", type(api_name).__name__
        except ModuleNotFoundError:
            pass
        parts = api_name.split('.')
        for i in range(len(parts) - 1, 0, -1):
            module_name = '.'.join(parts[:i])
            attr_name = parts[i]
            try:
                module = importlib.import_module(module_name)
                attr = getattr(module, attr_name, None)
                if attr:
                    type_name = 'class' if inspect.isclass(attr) else attr.__name__
                    return f"from {module_name} import {attr_name}", type_name
            except ModuleNotFoundError:
                continue
        print(f"# Error: Could not generate import code for {api_name}")
        return "", ""
    def format_value(self, value, value_type):
        if "str" in value_type:
            value = str(value).strip()
            if value.startswith("("): # if user input tuple parameters, return directly
                return value # (('tuple' in value) or ('Tuple' in value)) and 
            elif value.startswith("["): # if user input tuple parameters, return directly
                return value # (('list' in value) or ('List' in value)) and 
            elif value.startswith("{"): # if user input tuple parameters, return directly
                return value # (('dict' in value) or ('Dict' in value)) and 
            elif (value.startswith("'")) and (value.endswith("'")):
                return value
            elif value.startswith('"') and value.endswith('"'):
                return value
            else:
                return f"'{value}'"
        elif value_type in ["int", "float", "bool"]:
            return str(value)
        else:
            return str(value)
    def generate_execution_code(self, api_params_list):
        generated_code = []
        for api_info in api_params_list:
            api_name = api_info['api_name']
            selected_params = api_info['parameters']
            return_type = api_info['return_type']
            class_selected_params = api_info['class_selected_params']
            code_for_one_api = self.generate_execution_code_for_one_api(api_name, selected_params, return_type, class_selected_params)
            generated_code.append(code_for_one_api)
        return '\n'.join(generated_code)
    def format_arguments(self, selected_params):
        positional_args = []
        keyword_args = []
        for k, v in selected_params.items():
            if v['optional']:
                keyword_args.append(f"{k}={self.format_value(v['value'], v['type'])}")
            else:
                positional_args.append(self.format_value(v['value'], v['type']))
        params_formatted = ', '.join(positional_args + keyword_args)
        return params_formatted

    def generate_execution_code_for_one_api(self, api_name, selected_params, return_type, class_selected_params={}):
        import_code, type_api = self.get_import_code(api_name)
        if import_code in [i['code'] for i in self.execute_code if i['code_type']=='import' and i['success']=='True']:
            # if already imported
            print('api already imported!')
            pass
        else:
            print('api not imported, import now!')
            print(import_code)
            tmp_result = self.execute_api_call(import_code, "import")
            if tmp_result:
                print(f'Error during importing of api calling! {tmp_result}')
        api_parts = api_name.split('.')
        # Convert the parameters to the format 'param_name=param_value' or 'param_value' based on optionality
        params_formatted = self.format_arguments(selected_params)
        if class_selected_params:
            class_params_formatted = self.format_arguments(class_selected_params)
        # Class type API need to be initialized first, then used
        if type_api == "class":
            # double check for API type
            if not class_selected_params:
                raise ValueError
            final_api_name = api_parts[-1]
            maybe_class_name = api_parts[-2]
            maybe_instance_name = maybe_class_name.lower() + "_instance"
            if maybe_instance_name in self.variables:
                api_call = f"{maybe_instance_name}.{final_api_name}({params_formatted})"
            else:
                api_call = f"{maybe_instance_name} = {maybe_class_name}({class_params_formatted})\n"
                # Ensure that the method is called on the new instance
                api_call += f"{maybe_instance_name}.{final_api_name}({params_formatted})"
            class_API = maybe_instance_name
        else:
            final_api_name = api_parts[-1]
            api_call = f"{final_api_name}({params_formatted})"
        # generate return information
        if (return_type not in ["NoneType", None]) and (not return_type.startswith('Optional')):
            self.counter += 1
            if '=' in api_call:
                return import_code+'\n'+f"{api_call}"
            else:
                return_var = f"result_{self.counter} = "
                self.generate_code.append(f"{return_var}{api_call}")
                return import_code+'\n'+f"{return_var}{api_call}"
        else:
            self.generate_code.append(f"{api_call}")
            return import_code+'\n'+f"{api_call}"
    def split_tuple_variable(self, ):
        # generate splitting code if the results is a tuple
        # split result_n into result_n+1, reuslt_n+2, result_n+3 = result_n
        if self.execute_code[-1]['success']:
            last_code = self.execute_code[-1]
            # Check if the last code snippet ends with 'result'
            if last_code['code'].strip().startswith('result'):
                # Extract the variable name that starts with 'result'
                result_name_tuple = last_code['code'].strip().split('=')[0].strip()
                result_variable = self.variables[result_name_tuple]
                # Check if the variable is a tuple
                if 'tuple' in str(type(result_variable['value'])):
                    print('==>indeed split tuple variables!')
                    length = len(result_variable['value'])
                    new_variables = [f"result_{self.counter + i + 1}" for i in range(length)]
                    new_code = ', '.join(new_variables) + f" = {result_name_tuple}"
                    # execute new api call
                    self.execute_api_call(last_code['code']+'\n'+new_code, last_code['code_type'])
                    # Update the count
                    self.counter += length
                    print(f'==>generate new code: {new_code}')
        else:
            pass

    def execute_api_call(self, api_call_code, code_type):
        try:
            globals_before = set(globals().keys())
            exec(api_call_code, globals())
            globals_after = set(globals().keys())
            new_vars = globals_after - globals_before
            for var_name in new_vars:
                var_value = globals()[var_name]
                var_type = type(var_value).__name__
                self.variables[var_name] = {
                    "type": var_type,
                    "value": var_value
                }
            self.execute_code.append({'code':api_call_code,'code_type':code_type, 'success':'True', 'error':''})
            return ''
        except Exception as e:
            error = f"{e}"
            print('error in execute api call:', error)
            self.execute_code.append({'code':api_call_code,'code_type':code_type, 'success':'False', 'error': error})
            return error
    def save_variables_to_json(self):
        save_data = {name: details["type"] for name, details in self.variables.items()}
        with open(os.path.join(self.save_directory,"variables.json"), "w") as file:
            json.dump(save_data, file)
    def load_variables_to_json(self):
        with open(os.path.join(self.save_directory,"variables.json"), "r") as file:
            saved_vars = json.load(file)
        variables = {}
        for var_name, var_type in saved_vars.items():
            if var_name in globals():
                variables[var_name] = {
                    "type": var_type,
                    "value": globals()[var_name]
                }
        self.variables = variables
    def execute_one_pass(self, api_info):
        api_name = api_info['api_name']
        params = api_info['parameters']
        print('Automatically/Manually Selected params for $:')
        selected_params = self.select_parameters(params)
        # TODO: modify the selected_params, there exist multiple possible_Choices, need to fix one.
        print('After selecting parameters: ', selected_params)
        none_value_params = [param_name for param_name, param_info in selected_params.items() if param_info["value"] in ['@']]
        if none_value_params:
            print("Parameters @ with value unassigned are:", none_value_params)
            selected_params = self.makeup_for_missing_parameters(selected_params, 'user_input_placeholder')
            print('After Entering parameters: ', selected_params)
        return_type = api_info['return_type']
        api_params_list = [{"api_name":api_name, 
                            "class_selected_params":selected_params, 
                            "return_type":return_type,
                            "parameters":api_info['parameters']}]
        execution_code = self.generate_execution_code(api_params_list)
        print(execution_code)
        execution_code_list = execution_code.split('\n')
        for code in execution_code_list:
            self.execute_api_call(code, "code")
    
if __name__=='__main__':
    # Step 1: Provide the complete test_apis list
    test_apis = [
        {
            "api_name": "",
            "parameters": {
                "X": {
                    "type": "float",
                    "description": "Data to scale",
                    "value": 1.0,
                    "optional":False,
                },
                "X2": {
                    "type": "float",
                    "description": "Data to scale",
                    "value": 1.0,
                    "optional":False,
                },
                "X3": {
                    "type": "float",
                    "description": "Data to scale",
                    "value": 1.0,
                    "optional":False,
                },
                },
            "return_type": "tuple"
        },
        {
            "api_name": "sklearn.datasets.load_iris",
            "parameters": {},
            "return_type": "tuple"
        },
        {
            "api_name": "sklearn.preprocessing.StandardScaler.fit_transform",
            "parameters": {
                "X": {
                    "type": "ndarray",
                    "description": "Data to scale",
                    "value": "$",
                    "optional":False,
                }
            },
            "return_type": "ndarray"
        },
        {
            "api_name": "sklearn.decomposition.PCA.fit_transform",
            "parameters": {
                "X": {
                    "type": "ndarray",
                    "description": "Data to transform",
                    "value": "$",
                    "optional":False,
                }
            },
            "return_type": "ndarray"
        },
        {
            "api_name": "sklearn.model_selection.train_test_split",
            "parameters": {
                "X": {
                    "type": "ndarray",
                    "description": "Data to split",
                    "value": "$",
                    "optional":False,
                },
                "y": {
                    "type": "ndarray",
                    "description": "Labels",
                    "value": "$",
                    "optional":False,
                },
                "test_size": {
                    "type": "float",
                    "description": "Size of the test subset",
                    "value": 0.25,
                    "optional":True,
                }
            },
            "return_type": "tuple"
        },
        {
            "api_name": "sklearn.svm.SVC.fit",
            "parameters": {
                "X": {
                    "type": "ndarray",
                    "description": "Training data",
                    "value": "$",
                    "optional":False,
                },
                "y": {
                    "type": "ndarray",
                    "description": "Target values",
                    "value": "$",
                    "optional":False,
                }
            },
            "return_type": "object"
        },
        {
            "api_name": "sklearn.svm.SVC.predict",
            "parameters": {
                "X": {
                    "type": "ndarray",
                    "description": "Data to predict",
                    "value": "$",
                    "optional":False,
                }
            },
            "return_type": "ndarray"
        },
        {
            "api_name": "sklearn.metrics.accuracy_score",
            "parameters": {
                "y_true": {
                    "type": "ndarray",
                    "description": "True labels",
                    "value": "$",
                    "optional":False,
                },
                "y_pred": {
                    "type": "ndarray",
                    "description": "Predicted labels",
                    "value": "$",
                    "optional":False,
                }
            },
            "return_type": "float"
        },
        {
            "api_name": "numpy.savetxt",
            "parameters": {
                "fname": {
                    "type": "str",
                    "description": "File name",
                    "value": "@",
                    "optional":False,
                },
                "X": {
                    "type": "ndarray",
                    "description": "Data to save",
                    "value": "$",
                    "optional":False,
                },
                "delimiter": {
                    "type": "str",
                    "description": "Delimiter for the data",
                    "value": "@",
                    "optional":True,
                }
            },
            "return_type": "None"
        }
    ]
    ##########
    # Initializing executor
    executor = CodeExecutor()
    # Execute the initial setup as in Test1
    executor.execute_api_call("import numpy as np", "import")
    executor.execute_api_call("import pandas as pd", "import")
    executor.execute_api_call("data = np.array([[1, 2], [3, 4], [5, 6]])", "code")
    executor.execute_api_call("labels = np.array([0, 1, 0])", "code")
    for api_info in test_apis:
        executor.execute_one_pass(api_info)
        executor.split_tuple_variable()
    print('-'*10)
    print('Current variables in namespace:')
    print(json.dumps(str(executor.variables)))
    print('All successfully executed code:')
    #print('code:    success or not:')
    for i in executor.execute_code:
        if i['success']=='True':
            print(i['code'])
    print('Save variable json:')
    executor.save_variables_to_json()
