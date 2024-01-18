# instruction_generation
# prepare for retriever data
import json, os, re, copy, ast, random, time, cProfile, pstats, argparse, asyncio
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-test')

from tqdm import tqdm as tqdm_normal
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
from sklearn.utils import shuffle
from models.model import LLM_response, LLM_model
#from configs.model_config import LIB
from prompt.instruction import Task_Description_of_Singletool_oneapi_Instructions_whole, Other_Requirements_singletool_oneapi_whole
from inference.utils import process_retrieval_document, compress_api_str_from_list
parser = argparse.ArgumentParser()
parser.add_argument('--LIB', type=str, help='PyPI tool')
parser.add_argument('--concurrency', type=int, default=80, help='adjust the maximum concurrency according to the rate limit of OpenAI API account')
args = parser.parse_args()

semaphore = asyncio.Semaphore(args.concurrency)

prompt_oneapi_whole = f"{Task_Description_of_Singletool_oneapi_Instructions_whole}\n{Other_Requirements_singletool_oneapi_whole}"

def unify_response_format(response):
    list_pattern = re.compile(r'\[\{.*?\}\]', re.DOTALL)
    matched_lists = list_pattern.findall(response)
    unified_response = []
    for single_response in matched_lists:
        response_list = ast.literal_eval(single_response)
        unified_response.extend(response_list)
    return unified_response

async def async_LLM_response(llm, tokenizer, prompt, history=[], kwargs={}):
    loop = asyncio.get_event_loop()
    response, history = await loop.run_in_executor(None, LLM_response, llm, tokenizer, prompt, history, kwargs)
    return response, history

async def process_prompt_async(api_name, api, llm, tokenizer, prompt_template, progress):
    prompt_ans = compress_api_str_from_list(api)
    prompt = f'\nGiven API information starts: \n{prompt_ans}\nGiven API information ends\n{prompt_template}'
    retry_count = 0
    MAX_trial = 3
    valid_response = None
    while retry_count < MAX_trial:
        #response, history = LLM_response(llm, tokenizer, prompt, history=[], kwargs={})
        response, _ = await async_LLM_response(llm, tokenizer, prompt)
        try:
            response_list = unify_response_format(response)
            if response_list and isinstance(response_list, list) and isinstance(response_list[0], dict) and all('Query' in response for response in response_list):
                valid_response = response_list
                break
        except:
            pass
        retry_count += 1
    #print('GPT response:', response)
    if not valid_response:
        return []
    results = []
    for response_dict in response_list:
        api_tmp = copy.deepcopy(api)
        if api_name not in response_dict['Query']:  # filter out the response which contains API
            query = response_dict['Query']
            api_tmp['query'] = query
            results.append(api_tmp)
        del api_tmp
    # update progress bar
    progress.update(1)
    return results

async def preprocess_instruction_generation(API_composite, QUERY_FILE):
    # step1: Instruction Generation, compress API, build prompt
    with open(API_composite, 'r') as f:
        ori_data = json.load(f)
    print('The length of api_data is: ',len(ori_data))
    # filter and remove class API
    ori_data = {api: details for api, details in ori_data.items() if details['type'] != 'class'}
    print('The length of api_data after filtering class type API is: ',len(ori_data))
    # Convert the output data dict to a list of dicts
    llm, tokenizer = LLM_model()
    results = []
    #tasks = [process_api_async(api_name, ori_data[api_name], llm, tokenizer) for api_name in tqdm(ori_data)]
    all_tasks = []
    print('Start instruction generation ...')
    print('Num. of Tasks is one times of the num. of APIs ...')
    progress = tqdm_asyncio(total=len(ori_data))
    for api_name in tqdm_asyncio(ori_data):
        async with semaphore:
            all_tasks.append(process_prompt_async(api_name, ori_data[api_name], llm, tokenizer, prompt_oneapi_whole, progress))
    # Run the tasks and collect results
    results_from_tasks = await asyncio.gather(*(all_tasks))
    # close progres bar
    progress.close()
    # Process the ordered results
    results = copy.deepcopy([item for sublist in results_from_tasks for item in sublist])
    for i in range(len(results)):
        results[i]['query_id'] = i
    retained_apis_from_results = set([entry['api_calling'][0].split('(')[0] for entry in results])
    retained_proportion = len(retained_apis_from_results) / len(ori_data)
    print(f'the retained proportion is {retained_proportion}')
    print(f'the retained proportion num of apis is {len(retained_apis_from_results)}')
    with open(QUERY_FILE, 'w') as f:
        json.dump(results, f, indent=4)

def preprocess_retriever_data(OUTPUT_DIR, QUERY_FILE, QUERY_ANNOTATE_FILE, INDEX_FILE):
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
            # If the API calling changes, consider the first and last indices for validation
            if current_duration >= 10:
                val_index_set.append(i - current_duration )  # First index
                val_index_set.append(i-1)  # Last index
            else:
                val_index_set.append(i-1)
            current_api_calling = api_calling
            current_duration = 1
    # remaining 10
    if current_duration >= 10:
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
    query_test = [i for i in query_data if i['query_id'] in test_index_set]
    query_val = [i for i in query_data if i['query_id'] in val_index_set]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/train.json", 'w') as f:
        json.dump(query_train, f, indent=4)
    with open(f"{OUTPUT_DIR}/test.json", 'w') as f:
        json.dump(query_test, f, indent=4)
    with open(f"{OUTPUT_DIR}/val.json", 'w') as f:
        json.dump(query_val, f, indent=4)
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

def get_all_path(lib_name):
    os.makedirs(f"data/standard_process/{lib_name}/retriever_train_data", exist_ok=True)
    API_composite = f'./data/standard_process/{lib_name}/API_composite.json'
    OUTPUT_DIR = f"data/standard_process/{lib_name}/retriever_train_data"
    QUERY_FILE = f"data/standard_process/{lib_name}/API_inquiry.json"
    QUERY_ANNOTATE_FILE = f"data/standard_process/{lib_name}/API_inquiry_annotate.json"
    INDEX_FILE = f"data/standard_process/{lib_name}/API_instruction_testval_query_ids.json"
    return API_composite, OUTPUT_DIR, QUERY_FILE, QUERY_ANNOTATE_FILE, INDEX_FILE

def preprocess_fake_test_data(QUERY_FILE, QUERY_ANNOTATE_FILE):
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

if __name__=='__main__':
    API_composite, OUTPUT_DIR, QUERY_FILE, QUERY_ANNOTATE_FILE, INDEX_FILE = get_all_path(args.LIB)
    t1 = time.time()
    asyncio.run(preprocess_instruction_generation(API_composite, QUERY_FILE))
    print('step1 cost:', time.time()-t1)
    t1 = time.time()
    preprocess_fake_test_data(QUERY_FILE, QUERY_ANNOTATE_FILE)
    print('step2 cost:', time.time()-t1)
    t1 = time.time()
    preprocess_retriever_data(OUTPUT_DIR, QUERY_FILE, QUERY_ANNOTATE_FILE, INDEX_FILE)
    print('step3 cost:', time.time()-t1)
