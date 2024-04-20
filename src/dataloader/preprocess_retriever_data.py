# instruction_generation
# prepare for retriever data
import json, os, re, copy, ast, random, time, cProfile, pstats, argparse, asyncio
from tqdm import tqdm as tqdm_normal
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
from sklearn.utils import shuffle
from dotenv import load_dotenv
from models.model import LLM_response, LLM_model
#from configs.model_config import LIB
from prompt.instruction import make_instruction_generation_prompt
from inference.utils import process_retrieval_document, compress_api_str_from_list, json_to_docstring, process_retrieval_desc
from dataloader.get_API_init_from_sourcecode import parse_content_list
from inference.retriever_finetune_inference import ToolRetriever
from inference.utils import is_pair_in_merged_pairs, get_all_api_json, find_similar_api_pairs, find_similar_two_pairs, get_ambiguous_pairs
from typing import Any, Tuple, Optional

def unify_response_format(response: str) -> list:
    """
    Converts a JSON-like response string into a unified list of dictionaries.
    Handles cases where the response may not be properly formatted as valid JSON.

    Parameters
    ----------
    response : str
        The JSON-like response string to process.

    Returns
    -------
    list
        A list of dictionaries representing the parsed JSON objects.
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        list_pattern = re.compile(r'\[\{.*?\}\]', re.DOTALL)
        matched_lists = list_pattern.findall(response)
        unified_response = []
        for single_response in matched_lists:
            try:
                response_list = ast.literal_eval(single_response)
                unified_response.extend(response_list)
            except (ValueError, SyntaxError):
                pass
        return unified_response

def preprocess_json_string(response: str) -> str:
    """
    Preprocesses a JSON string, replacing single quotes with double quotes for JSON keys
    and ensuring internal single quotes within strings are preserved.

    Parameters
    ----------
    response : str
        The JSON string to preprocess.

    Returns
    -------
    str
        The preprocessed JSON string suitable for JSON parsing.
    """
    # Replace single quotes with double quotes while preserving internal single quotes in strings
    in_string = False
    processed_response = ""
    for char in response:
        if char == "'" and not in_string:
            processed_response += '"'
        elif char == '"' and not in_string:
            # Entering a string
            in_string = True
            processed_response += char
        elif char == '"' and in_string:
            # Exiting a string
            in_string = False
            processed_response += char
        else:
            processed_response += char
    return processed_response

def parse_unformatted_jsons(response: str) -> list:
    """
    Attempts to parse a string containing multiple JSON objects even if not properly formatted.

    Parameters
    ----------
    response : str
        The string containing potentially multiple unformatted JSON objects.

    Returns
    -------
    list
        A list of dictionaries, each representing a parsed JSON object.
    """
    response = preprocess_json_string(response)  # Preprocess to handle single quotes
    jsons = []
    current_json = ''
    brace_count = 0
    for char in response:
        if char == '{':
            brace_count += 1
            current_json += char
        elif char == '}':
            brace_count -= 1
            current_json += char
            if brace_count == 0:
                jsons.append(current_json)
                current_json = ''
        elif brace_count > 0:
            current_json += char
    parsed_jsons = []
    for json_str in jsons:
        try:
            parsed_json = json.loads(json_str)
            parsed_jsons.append(parsed_json)
        except json.JSONDecodeError:
            pass
    return parsed_jsons

def parse_json_response(response: str) -> list:
    """
    Parses a string response into JSON format and filters out valid JSON objects.

    Parameters
    ----------
    response : str
        The response string containing JSON objects.

    Returns
    -------
    list
        A list of valid JSON objects.
    """
    parsed_response = parse_unformatted_jsons(response)
    valid_responses = []
    for d in parsed_response:
        valid_responses.append(d)
    return valid_responses

async def async_LLM_response(llm: Any, tokenizer: Any, prompt: str, GPT_model: str, history: list = [], kwargs: dict = {}) -> Tuple[str, list]:
    """
    Asynchronously sends a prompt to a language model and awaits the response.

    Parameters
    ----------
    llm : Any
        The language learning model.
    tokenizer : Any
        The tokenizer for processing text.
    prompt : str
        The prompt to send to the language model.
    GPT_model : str
        The specific GPT model version to use.
    history : list, optional
        Previous interactions with the language model, if applicable.
    kwargs : dict, optional
        Additional keyword arguments for the language model interaction.

    Returns
    -------
    Tuple[str, list]
        A tuple containing the model's response and the interaction history.
    """
    model_version = "gpt-4-0125-preview" if args.GPT_model == 'gpt4' else "gpt-3.5-turbo-0125"
    loop = asyncio.get_event_loop()
    response, history = await loop.run_in_executor(None, LLM_response, llm, tokenizer, prompt, model_version, history, kwargs)
    return response, history

async def process_prompt_async(desc_retriever: Any, API_init: dict, api_name: str, api: dict, llm: Any, tokenizer: Any, tmp_docstring: str, progress: tqdm_normal, similar_api_same_desc: dict, similar_api_same_funcname: dict, GPT_model: str) -> list:
    """
    Processes a prompt asynchronously by preparing the prompt, sending it to a language model, and processing the response.

    Parameters
    ----------
    desc_retriever : Any
        The descriptor retriever for pulling similar API descriptions.
    API_init : dict
        Initial API data loaded as a dictionary.
    api_name : str
        The name of the API being processed.
    api : dict
        The API data dictionary.
    llm : Any
        The language learning model.
    tokenizer : Any
        The tokenizer for processing text.
    tmp_docstring : str
        The temporary docstring prepared for the prompt.
    progress : tqdm_normal
        The progress bar instance for tracking progress.
    similar_api_same_desc : dict
        A dictionary of similar API descriptions.
    similar_api_same_funcname : dict
        A dictionary of similar API function names.
    GPT_model : str
        The specific GPT model version to use.

    Returns
    -------
    list
        A list of dictionaries containing the processed results from the language model.
    """
    prompt1, prompt2 = make_instruction_generation_prompt(api_name, tmp_docstring)
    retrieved_apis = desc_retriever.retrieving(query=API_init[api_name]['description'].split('\n')[0],top_k=20)
    assert len(retrieved_apis) == len(set(retrieved_apis)), 'repeated apis in retrieved_apis'+retrieved_apis
    # remove target api
    if api_name in retrieved_apis:
        retrieved_apis.remove(api_name)
    assert api_name not in retrieved_apis, 'api name not in retrieved apis: '+json.dumps(retrieved_apis) + ', ' + api_name
    # remove compositeAPI and classAPI
    # remove same description type ambiguous API
    #filtered_apis = [api for api in retrieved_apis if not is_pair_in_merged_pairs(api_name, api, similar_api_same_desc)]
    filtered_apis = [api for api in retrieved_apis if (not is_pair_in_merged_pairs(api_name, api, similar_api_same_funcname)) and (not is_pair_in_merged_pairs(api_name, api, similar_api_same_desc))]
    #if len(filtered_apis)<len(retrieved_apis):
    #    print('remove some apis: ', set(retrieved_apis)-set(filtered_apis))
    filtered_apis = filtered_apis[:5]
    assert len(filtered_apis)==5
    retrieved_descriptions = [API_init[i]['description'].split('\n')[0].replace('\n',' ') for i in filtered_apis]
    unique_descriptions = list(set(retrieved_descriptions))
    target_description = API_init[api_name]['description'].split('\n')[0].replace('\n',' ')
    #return
    retrieved_desc_apis = ''
    index = 0
    for tmp_i in unique_descriptions:
        #retrieved_desc_apis+=str(index)+': '+tmp_i+'\n'
        retrieved_desc_apis+='"'+tmp_i+'",'
        index+=1
    #prompt_ans = compress_api_str_from_list(api)
    descriptions = f"""Target description:{target_description}\nContrasting descriptions: \n[{retrieved_desc_apis}]."""
    prompt = f"""{prompt1}\nNow I provide my target and contrasting descriptions to you, {descriptions}\n{prompt2} Now generate the 10 examples following the Instruction."""
    #prompt = f"""{prompt_template}Target description:{target_description}. Now generate the 10 examples following the Instruction."""
    retry_count = 0
    MAX_trial = 3
    valid_response = None
    while retry_count < MAX_trial:
        response, _ = await async_LLM_response(llm, tokenizer, prompt, GPT_model)
        try:
            response_list = parse_json_response(response)
            if response_list and isinstance(response_list, list) and isinstance(response_list[0], dict) and all('instruction' in response for response in response_list) and len(response_list)>=10:
                valid_response = response_list
                break
        except:
            pass
        retry_count += 1
    #print('Prompt: ', prompt)
    #print('GPT response:', response)
    if not valid_response:
        return []
    results = []
    for response_dict in response_list:
        api_tmp = copy.deepcopy(api)
        if api_name not in response_dict['instruction']:  # filter out the response which contains API
            api_tmp['query'] = response_dict['instruction']
            api_tmp['query_code'] = response_dict['code']
            results.append(api_tmp)
        del api_tmp
    # update progress bar
    progress.update(1)
    return results

def process_parameters(parameters: dict, max_parameters: int = 6) -> dict:
    """
    Processes and sorts parameters based on their optionality and limits the number of parameters returned.

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters to process.
    max_parameters : int, optional
        The maximum number of parameters to return, default is 6.

    Returns
    -------
    dict
        The processed and sorted dictionary of parameters.
    """
    for param_data in parameters.values():
        param_data.pop('optional_value', None)
    sorted_parameters = sorted(parameters.items(), key=lambda x: (x[1]['optional'], x[0]))
    processed_parameters = {}
    required_count = 0
    optional_count = 0
    for param_name, param_data in sorted_parameters:
        if not param_data['optional']:
            if required_count < max_parameters:
                processed_parameters[param_name] = param_data
                required_count += 1
        else:
            if optional_count < max_parameters - required_count:
                processed_parameters[param_name] = param_data
                optional_count += 1
    return processed_parameters

def preprocess_retriever_data(OUTPUT_DIR: str, QUERY_FILE: str, QUERY_ANNOTATE_FILE: str, INDEX_FILE: str) -> None:
    """
    Preprocesses the data for a retriever by splitting into training, validation, and testing sets.

    Parameters
    ----------
    OUTPUT_DIR : str
        The directory where processed files will be saved.
    QUERY_FILE : str
        The file path to the original query data.
    QUERY_ANNOTATE_FILE : str
        The file path to the annotated query data.
    INDEX_FILE : str
        The file path to save indices of test and validation sets.

    """
    with open(QUERY_FILE, 'r') as f:
        query_data_ori = json.load(f)
    start_idx_for_test = max([i['query_id'] for i in query_data_ori])
    # code from toolbench retriever train.py
    with open(QUERY_ANNOTATE_FILE, 'r') as f:
        query_data = json.load(f)
    idx = len(query_data)
    ############# fixed split
    test_indices = [i['query_id'] for i in query_data if i['query_id']>start_idx_for_test]
    test_index_set = list(set(test_indices))
    val_index_set = []
    # Track the current consecutive duration and API calling
    current_duration = 1
    current_api_calling = query_data[0]['api_calling']
    # Iterate through the data
    for i in range(1, len(query_data_ori)):
        api_calling = query_data[i]['api_calling']
        if api_calling == current_api_calling:
            current_duration += 1
        else:
            if current_duration<10:
                print(api_calling, current_duration)
            # If the API calling changes, consider the first and last indices for validation
            if current_duration >= 3:
                val_index_set.append(i - current_duration )  # First index
                val_index_set.append(i-1)  # Last index
            else:
                val_index_set.append(i-1)
            current_api_calling = api_calling
            current_duration = 1
    # remaining 10
    if current_duration<10:
        print(api_calling, current_duration)
    if current_duration >= 3:
        print(len(query_data_ori) - current_duration, len(query_data_ori)-1)
        val_index_set.append(len(query_data_ori) - current_duration )  # First index
        val_index_set.append(len(query_data_ori)-1)  # Last index
    else:
        val_index_set.append(len(query_data_ori)-1)
    ### CHANGE 2: Update the logic to select every fifth data point for validation ###
    final_index_data = {'test':test_index_set, 'val':val_index_set}
    assert len(set(test_index_set).intersection(set(val_index_set))) == 0, f"Test and Val sets overlap.{set(test_index_set).intersection(set(val_index_set))}"
    #assert len(query_data) not in test_index_set, "Test set is empty or contains the last index."
    with open(INDEX_FILE, 'w') as f:
        json.dump(final_index_data, f, indent=4)
    query_train = [i for i in query_data if i['query_id'] not in test_index_set and i['query_id'] not in val_index_set]
    query_val = [i for i in query_data if i['query_id'] in val_index_set]
    query_test = [i for i in query_data if i['query_id'] in test_index_set]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/train.json", 'w') as f:
        json.dump(query_train, f, indent=4)
    with open(f"{OUTPUT_DIR}/val.json", 'w') as f:
        json.dump(query_val, f, indent=4)
    with open(f"{OUTPUT_DIR}/test.json", 'w') as f:
        json.dump(query_test, f, indent=4)
    ### For dataset preprocess ###
    documents = []
    doc_id_map = {}  # Create a mapping from doc to doc_id
    train_pairs = []
    test_pairs = []
    val_pairs = []
    def process_data(data, pairs):
        for doc in tqdm_normal(data):
            doc_content = {
                "api_calling": doc['api_calling'],
                "api_name": doc['api_calling'][0].split('(')[0],
                "api_description": doc["description"],
                "required_parameters": [{"name": param, "info": param_info} for param, param_info in doc["Parameters"].items() if not param_info["optional"]],
                "optional_parameters": [{"name": param, "info": param_info} for param, param_info in doc["Parameters"].items() if param_info["optional"]],
                "Returns": doc["Returns"],
            }
            #api = {key:doc[key] for key in doc if key not in ['query', 'query_id']}
            doc_id = doc_id_map.setdefault(json.dumps(doc_content), len(doc_id_map))
            pairs.append(([doc['query_id'], doc['query']], [doc['query_id'], 0, doc_id, 1]))
            documents.append((doc_id, json.dumps(doc_content)))
    process_data(query_train, train_pairs)
    process_data(query_test, test_pairs)
    process_data(query_val, val_pairs)
    # Shuffle the data using the shuffle function
    train_pairs = shuffle(train_pairs, random_state=42)
    test_pairs = shuffle(test_pairs, random_state=42)
    val_pairs = shuffle(val_pairs, random_state=42)
    # Split the shuffled data into queries and labels
    train_queries, train_labels = zip(*train_pairs)
    test_queries, test_labels = zip(*test_pairs)
    val_queries, val_labels = zip(*val_pairs)
    train_queries_df = pd.DataFrame(train_queries, columns=['qid', 'query_text'])
    train_labels_df = pd.DataFrame(train_labels, columns=['qid', 'useless', 'docid', 'label'])
    test_queries_df = pd.DataFrame(test_queries, columns=['qid', 'query_text'])
    test_labels_df = pd.DataFrame(test_labels, columns=['qid', 'useless', 'docid', 'label'])
    val_queries_df = pd.DataFrame(val_queries, columns=['qid', 'query_text'])
    val_labels_df = pd.DataFrame(val_labels, columns=['qid', 'useless', 'docid', 'label'])
    documents_df = pd.DataFrame(documents, columns=['docid', 'document_content'])
    # Save as .tsv and .txt files
    train_queries_df.to_csv(OUTPUT_DIR + '/train.query.txt', sep='\t', index=False, header=False)
    test_queries_df.to_csv(OUTPUT_DIR + '/test.query.txt', sep='\t', index=False, header=False)
    val_queries_df.to_csv(OUTPUT_DIR + '/val.query.txt', sep='\t', index=False, header=False)
    train_labels_df.to_csv(OUTPUT_DIR + '/qrels.train.tsv', sep='\t', index=False, header=False)
    test_labels_df.to_csv(OUTPUT_DIR + '/qrels.test.tsv', sep='\t', index=False, header=False)
    val_labels_df.to_csv(OUTPUT_DIR + '/qrels.val.tsv', sep='\t', index=False, header=False)
    documents_df.to_csv(OUTPUT_DIR + '/corpus.tsv', sep='\t', index=False)

def filter_and_update_query_id(query_data: list, api_list: list = []) -> list:
    """
    Filters and updates the query IDs based on a list of API names.

    Parameters
    ----------
    query_data : list
        A list of query data dictionaries.
    api_list : list, optional
        A list of API names to filter by.

    Returns
    -------
    list
        The filtered and updated list of query data.
    """
    filtered_queries = []
    for query in query_data:
        api_name = query['api_calling'][0].split('(')[0]
        if api_list:
            if api_name in api_list:
                filtered_queries.append(query)
        else:
            filtered_queries.append(query)
    for i, query in enumerate(filtered_queries):
        query['query_id'] = i
    return filtered_queries

def preprocess_retriever_data_shuffle(OUTPUT_DIR: str, QUERY_FILE: str, QUERY_ANNOTATE_FILE: str, INDEX_FILE: str, api_txt_path: Optional[str] = None) -> None:
    """
    Preprocesses retriever data with shuffling to ensure diverse training and testing sets.

    Parameters
    ----------
    OUTPUT_DIR : str
        The directory where processed files will be saved.
    QUERY_FILE : str
        The file path to the original query data.
    QUERY_ANNOTATE_FILE : str
        The file path to the annotated query data.
    INDEX_FILE : str
        The file path to save indices of test and validation sets.
    api_txt_path : Optional[str]
        The path to a text file containing API names, optional.
    """
    with open(QUERY_FILE, 'r') as f:
        query_data_ori = json.load(f)
    if api_txt_path:
        content_list = []
        try:
            with open(api_txt_path, 'r', encoding='latin') as file:
                content_list = file.readlines()
            api_list = parse_content_list(content_list)
        except FileNotFoundError:
            print(f"Error: File '{api_txt_path}' not found.")
        except Exception as e:
            print(f"Error: {e}")
    else:
        api_list = []
    print('previous query length:', len(query_data_ori))
    query_data_ori = filter_and_update_query_id(query_data_ori, api_list)
    print('filtered query length:', len(query_data_ori))
    #with open('data/standard_process/scanpy_subset/API_inquiry.json', 'w') as file:
    #    json.dump(query_data_ori, file, indent=4)
    start_idx_for_test = max([i['query_id'] for i in query_data_ori])
    assert start_idx_for_test==len(query_data_ori)-1, 'start_idx_for_test is not the last index of query_data_ori'
    # code from toolbench retriever train.py
    with open(QUERY_ANNOTATE_FILE, 'r') as f:
        query_data = json.load(f)
    print('previous query length:', len(query_data))
    query_data = filter_and_update_query_id(query_data, api_list)
    print('filtered query length:', len(query_data))
    #with open('data/standard_process/scanpy_subset/API_inquiry_annotate.json', 'w') as file:
    #    json.dump(query_data, file, indent=4)
    idx = len(query_data)
    ############# fixed split
    test_indices = [i['query_id'] for i in query_data if i['query_id']>start_idx_for_test]
    print('start_idx_for_test, idx: ', start_idx_for_test, idx)
    test_index_set = list(set(test_indices))
    val_index_set = []
    # Track the current consecutive duration and API calling
    current_duration = 1
    current_api_calling = query_data[0]['api_calling']
    # Iterate through the data
    for i in range(1, len(query_data_ori)):
        api_calling = query_data[i]['api_calling']
        if api_calling == current_api_calling:
            current_duration += 1
        else:
            if current_duration<10:
                print(api_calling, current_duration)
            # If the API calling changes, consider the first and last indices for validation
            if current_duration >= 3:
                val_indices = random.sample(list(range(i - current_duration, i)), 2)
                """if val_indices[0]>val_indices[1]:
                    val_index_set.append(val_indices[1])
                    val_index_set.append(val_indices[0])
                else:
                    val_index_set.append(val_indices[0])
                    val_index_set.append(val_indices[1])"""
                val_index_set.append(val_indices[0])
                val_index_set.append(val_indices[1])
                #val_index_set.append(i - current_duration)  # First index
                #val_index_set.append(i-1)  # Last index
            else:
                val_index_set.append(i-1)
            current_api_calling = api_calling
            current_duration = 1
    # remaining 10
    if current_duration<10:
        print(api_calling, current_duration)
    if current_duration >= 3:
        print(len(query_data_ori) - current_duration, len(query_data_ori)-1)
        val_indices = random.sample(list(range(len(query_data_ori) - current_duration, len(query_data_ori))), 2)
        """if val_indices[0]>val_indices[1]:
            val_index_set.append(val_indices[1])
            val_index_set.append(val_indices[0])
        else:
            val_index_set.append(val_indices[0])
            val_index_set.append(val_indices[1])"""
        val_index_set.append(val_indices[0])
        val_index_set.append(val_indices[1])
        #val_index_set.append(len(query_data_ori) - current_duration)  # First index
        #val_index_set.append(len(query_data_ori)-1)  # Last index
    else:
        val_index_set.append(len(query_data_ori)-1)
    ### CHANGE 2: Update the logic to select every fifth data point for validation ###
    final_index_data = {'test':test_index_set, 'val':val_index_set}
    assert len(set(test_index_set).intersection(set(val_index_set))) == 0, f"Test and Val sets overlap.{set(test_index_set).intersection(set(val_index_set))}"
    #assert len(query_data) not in test_index_set, "Test set is empty or contains the last index."
    with open(INDEX_FILE, 'w') as f:
        json.dump(final_index_data, f, indent=4)
    query_train = [i for i in query_data if i['query_id'] not in test_index_set and i['query_id'] not in val_index_set]
    query_val = [i for i in query_data if i['query_id'] in val_index_set]
    query_test = [i for i in query_data if i['query_id'] in test_index_set]
    print('length of query: ', len(query_train), len(query_val), len(query_test))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/train.json", 'w') as f:
        json.dump(query_train, f, indent=4)
    with open(f"{OUTPUT_DIR}/val.json", 'w') as f:
        json.dump(query_val, f, indent=4)
    with open(f"{OUTPUT_DIR}/test.json", 'w') as f:
        json.dump(query_test, f, indent=4)
    ### For dataset preprocess ###
    documents = []
    doc_id_map = {}  # Create a mapping from doc to doc_id
    train_pairs = []
    test_pairs = []
    val_pairs = []
    def process_data(data, pairs):
        for doc in tqdm_normal(data):
            doc_content = {
                "api_calling": doc['api_calling'],
                "api_name": doc['api_calling'][0].split('(')[0],
                "api_description": doc["description"],
                "required_parameters": [{"name": param, "info": param_info} for param, param_info in doc["Parameters"].items() if not param_info["optional"]],
                "optional_parameters": [{"name": param, "info": param_info} for param, param_info in doc["Parameters"].items() if param_info["optional"]],
                "Returns": doc["Returns"],
            }
            #api = {key:doc[key] for key in doc if key not in ['query', 'query_id']}
            doc_id = doc_id_map.setdefault(json.dumps(doc_content), len(doc_id_map))
            pairs.append(([doc['query_id'], doc['query']], [doc['query_id'], 0, doc_id, 1]))
            documents.append((doc_id, json.dumps(doc_content)))
    process_data(query_train, train_pairs)
    process_data(query_test, test_pairs)
    process_data(query_val, val_pairs)
    # Shuffle the data using the shuffle function
    train_pairs = shuffle(train_pairs, random_state=42)
    test_pairs = shuffle(test_pairs, random_state=42)
    val_pairs = shuffle(val_pairs, random_state=42)
    # Split the shuffled data into queries and labels
    print('length of train_pairs: ', len(train_pairs), len(val_pairs), len(test_pairs))
    train_queries, train_labels = zip(*train_pairs)
    test_queries, test_labels = zip(*test_pairs)
    val_queries, val_labels = zip(*val_pairs)
    train_queries_df = pd.DataFrame(train_queries, columns=['qid', 'query_text'])
    train_labels_df = pd.DataFrame(train_labels, columns=['qid', 'useless', 'docid', 'label'])
    test_queries_df = pd.DataFrame(test_queries, columns=['qid', 'query_text'])
    test_labels_df = pd.DataFrame(test_labels, columns=['qid', 'useless', 'docid', 'label'])
    val_queries_df = pd.DataFrame(val_queries, columns=['qid', 'query_text'])
    val_labels_df = pd.DataFrame(val_labels, columns=['qid', 'useless', 'docid', 'label'])
    documents_df = pd.DataFrame(documents, columns=['docid', 'document_content'])
    # Save as .tsv and .txt files
    train_queries_df.to_csv(OUTPUT_DIR + '/train.query.txt', sep='\t', index=False, header=False)
    test_queries_df.to_csv(OUTPUT_DIR + '/test.query.txt', sep='\t', index=False, header=False)
    val_queries_df.to_csv(OUTPUT_DIR + '/val.query.txt', sep='\t', index=False, header=False)
    train_labels_df.to_csv(OUTPUT_DIR + '/qrels.train.tsv', sep='\t', index=False, header=False)
    test_labels_df.to_csv(OUTPUT_DIR + '/qrels.test.tsv', sep='\t', index=False, header=False)
    val_labels_df.to_csv(OUTPUT_DIR + '/qrels.val.tsv', sep='\t', index=False, header=False)
    documents_df.to_csv(OUTPUT_DIR + '/corpus.tsv', sep='\t', index=False)

def get_all_path(lib_name: str) -> Tuple[str, str, str, str, str, str]:
    """
    Retrieves all necessary file paths based on the library name.

    Parameters
    ----------
    lib_name : str
        The name of the library for which paths are required.

    Returns
    -------
    Tuple[str, str, str, str, str, str]
        A tuple containing paths to API initialization, API composite, output directory, query file, annotated query file, and index file.
    """
    os.makedirs(f"data/standard_process/{lib_name}/retriever_train_data", exist_ok=True)
    API_composite = f'./data/standard_process/{lib_name}/API_composite.json'
    API_init = f'./data/standard_process/{lib_name}/API_init.json'
    OUTPUT_DIR = f"data/standard_process/{lib_name}/retriever_train_data"
    QUERY_FILE = f"data/standard_process/{lib_name}/API_inquiry.json"
    QUERY_ANNOTATE_FILE = f"data/standard_process/{lib_name}/API_inquiry_annotate.json"
    INDEX_FILE = f"data/standard_process/{lib_name}/API_instruction_testval_query_ids.json"
    return API_init, API_composite, OUTPUT_DIR, QUERY_FILE, QUERY_ANNOTATE_FILE, INDEX_FILE

def preprocess_fake_test_data(QUERY_FILE: str, QUERY_ANNOTATE_FILE: str) -> None:
    """
    Prepares fake test data by processing existing query files and ensuring a certain level of diversity.

    Parameters
    ----------
    QUERY_FILE : str
        The file path to the original query data.
    QUERY_ANNOTATE_FILE : str
        The file path where the annotated query data will be stored.
    """
    with open(QUERY_FILE, 'r') as f:
        query_data = json.load(f)
    seen_apis = set()
    final_data = []
    start_idx = len(query_data)
    for entry in query_data:
        new_entry = entry.copy()
        api_name = entry['api_calling'][0].split('(')[0]
        if api_name not in seen_apis:
            seen_apis.add(api_name)
            new_entry['query_id'] = start_idx
            start_idx+=1
            final_data.append(new_entry)
    datadata = query_data+final_data
    for entry in datadata:
        entry['api_name'] = entry['api_calling'][0].split('(')[0]
    with open(QUERY_ANNOTATE_FILE, "w") as file:
        json.dump(datadata, file, ensure_ascii=False, indent=4)

def create_corpus_from_json(API_init_json: dict, corpus_tsv_path: str) -> None:
    """
    Creates a corpus file from a JSON file detailing API information, saving it as a TSV.

    Parameters
    ----------
    API_init_json : dict
        The JSON containing API information.
    corpus_tsv_path : str
        The file path where the corpus will be saved as a TSV.
    """
    documents = []
    doc_id_map = {}
    #pairs = []
    for api_name, details in API_init_json.items():
        if details['api_type'] != 'class' and details['api_type']!='unknown':
            doc_content = {
                #"doc_id": api_name,
                "api_name": api_name,
                "api_description": str(details.get('description', '')),
                #"parameters": ', '.join([f"{param}: {info.get('description', '')}" for param, info in details.get('Parameters', {}).items()]),
                #"returns": details.get('Returns', {}).get('description', '')
            }
            #corpus_entries.append(doc_content)
            doc_id = doc_id_map.setdefault(json.dumps(doc_content), len(doc_id_map))
            #pairs.append(([details['api_name'], details['description']], [details['api_name'], 0, doc_id, 1]))
            documents.append((doc_id, json.dumps(doc_content)))
    documents_df = pd.DataFrame(documents, columns=["docid", "document_content"])
    documents_df.to_csv(corpus_tsv_path, sep='\t', index=False)

async def preprocess_instruction_d(desc_retriever: Any, API_init: dict, QUERY_FILE: str, LIB: str, GPT_model: str) -> None:
    """
    Asynchronously processes instruction data using a descriptor retriever and a language model.

    Parameters
    ----------
    desc_retriever : Any
        The descriptor retriever for pulling similar API descriptions.
    API_init : dict
        Initial API data loaded as a dictionary.
    QUERY_FILE : str
        The file path where the query data will be stored.
    LIB : str
        The library name being processed.
    GPT_model : str
        The specific GPT model version to use.
    """
    print('The length of api_data is: ',len(API_init))
    # filter and remove class API
    ori_data = {api: details for api, details in API_init.items() if details['api_type'] != 'class' and details['api_type']!='unknown'}
    print('The length of api_data after filtering class type API is: ',len(ori_data))
    # Convert the output data dict to a list of dicts
    llm, tokenizer = LLM_model()
    results = []
    #tasks = [process_api_async(api_name, ori_data[api_name], llm, tokenizer) for api_name in tqdm(ori_data)]
    all_tasks = []
    print('Num. of instruction generation Tasks is one times of the num. of APIs ...')
    progress = tqdm_asyncio(total=len(ori_data))
    idx = 0
    # get similar pairs
    merged_pairs, similar_api_same_desc, similar_api_same_funcname = get_ambiguous_pairs(f"./data/standard_process/{LIB}/API_init.json")
    for api_name in tqdm_asyncio(ori_data):
        async with semaphore:
            if True:
                tmp_doc = json_to_docstring(api_name, API_init[api_name]['description'].replace('\n',' '), process_parameters(API_init[api_name]['Parameters']))
                all_tasks.append(process_prompt_async(desc_retriever, API_init, api_name, ori_data[api_name], llm, tokenizer, tmp_doc, progress, similar_api_same_desc, similar_api_same_funcname, GPT_model)) # .split('\n')[0]
            else:
                pass
    # Run the tasks and collect results
    first_round_results = await asyncio.gather(*(all_tasks))
    # close progres bar
    progress.close()
    first_results = copy.deepcopy([item for sublist in first_round_results for item in sublist])
    
    async def process_api_async(api_name, api_data, api_ori_data, llm, tokenizer, max_retries = 3):
        for _ in range(max_retries):
            tmp_doc = json_to_docstring(api_name, api_data['description'].replace('\n', ' '), process_parameters(api_data['Parameters']))
            task_results = await process_prompt_async(desc_retriever, API_init, api_name, api_ori_data, llm, tokenizer, tmp_doc, progress, similar_api_same_desc, similar_api_same_funcname, GPT_model)
            if len(task_results) == 10:
                return task_results
        return task_results
    # filter api which inquiries < 10
    ########
    api_inquiries_count = {}
    for i, result in enumerate(first_results):
        api_name = result['api_calling'][0].split('(')[0]
        if api_name not in api_inquiries_count:
            api_inquiries_count[api_name] = 1
        else:
            api_inquiries_count[api_name] += 1
    insufficient_apis = [api_name for api_name, count in api_inquiries_count.items() if count < 10]
    print('insufficient_apis: ', insufficient_apis)
    # combine the first results and second results
    if len(insufficient_apis)>0:
        all_tasks = []
        progress = tqdm_asyncio(total=len(insufficient_apis))
        """for api_name in tqdm_asyncio(ori_data):
            async with semaphore:
                if api_name in insufficient_apis:
                    tmp_doc = json_to_docstring(api_name, API_init[api_name]['description'].replace('\n',' '), process_parameters(API_init[api_name]['Parameters']))
                    all_tasks.append(process_prompt_async(desc_retriever, API_init, api_name, ori_data[api_name], llm, tokenizer, tmp_doc, progress, similar_api_same_desc, similar_api_same_funcname, GPT_model)) # .split('\n')[0]
                else:
                    pass"""
        retry_tasks = [process_api_async(api_name, API_init[api_name], ori_data[api_name], llm, tokenizer, max_retries=3) for api_name in insufficient_apis]
        second_round_results = await asyncio.gather(*(retry_tasks))
        progress.close()
        retry_results = copy.deepcopy([item for sublist in second_round_results for item in sublist])
        # remove those inquiries from results
        print('first_results:', len(first_results))
        results = [res for i, res in enumerate(first_results) if res['api_calling'][0].split('(')[0] not in insufficient_apis]
        print('filter_results:', len(results))
        results.extend(retry_results)
        print('second_results:', len(results))
    else:
        results = first_results
    for i, result in enumerate(results):
        result['query_id'] = i
    # Process the ordered results
    with open(QUERY_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    retained_apis_from_results = set([entry['api_calling'][0].split('(')[0] for entry in results])
    retained_proportion = len(retained_apis_from_results) / len(ori_data)
    print(f'the retained proportion is {retained_proportion}')
    print(f'the retained proportion num of apis is {len(retained_apis_from_results)}')
    #print(results, 'results==>')

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--LIB', type=str, help='PyPI tool')
    parser.add_argument('--concurrency', type=int, default=80, help='adjust the maximum concurrency according to the rate limit of OpenAI API account')
    parser.add_argument('--GPT_model', type=str, default='gpt3.5', choices=['gpt4', 'gpt3.5'], help='GPT model version')
    parser.add_argument('--api_txt_path', type=str, default=None, help='Your self-defined api txt path')
    args = parser.parse_args()
    semaphore = asyncio.Semaphore(args.concurrency)
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-test')
    
    API_init, API_composite, OUTPUT_DIR, QUERY_FILE, QUERY_ANNOTATE_FILE, INDEX_FILE = get_all_path(args.LIB)
    with open(API_init, 'r') as f:
        API_init_json = json.load(f)
    # prepare desc_prompt corpus
    print('preparing desc_prompt corpus')
    os.makedirs(f"./data/standard_process/{args.LIB}/prompt_desc/", exist_ok=True)
    print('preparing API corpus')
    create_corpus_from_json(API_init_json,f"./data/standard_process/{args.LIB}/prompt_desc/corpus.tsv")
    # load pretrained bert model, prepare corpus
    desc_retriever = ToolRetriever(LIB = args.LIB, corpus_tsv_path=f"./data/standard_process/{args.LIB}/prompt_desc/corpus.tsv", model_path="all-MiniLM-L6-v2", add_base=False,shuffle_data=False, process_func=process_retrieval_desc)
    t1 = time.time()
    asyncio.run(preprocess_instruction_d(desc_retriever, API_init_json, QUERY_FILE, args.LIB, args.GPT_model))
    print('step1 cost:', time.time()-t1)
    t1 = time.time()
    preprocess_fake_test_data(QUERY_FILE, QUERY_ANNOTATE_FILE)
    print('step2 cost:', time.time()-t1)
    t1 = time.time()
    #preprocess_retriever_data(OUTPUT_DIR, QUERY_FILE, QUERY_ANNOTATE_FILE, INDEX_FILE)
    preprocess_retriever_data_shuffle(OUTPUT_DIR, QUERY_FILE, QUERY_ANNOTATE_FILE, INDEX_FILE, api_txt_path=args.api_txt_path)
    print('step3 cost:', time.time()-t1)
    # usage: python dataloader/preprocess_retriever_data.py --LIB scanpy_subset --api_txt_path ./data/standard_process/scanpy_subset/api_txt_path.txt

