import random, json, re, os, hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from string import punctuation

def get_retrieved_prompt():
    prompt = """Task: Select the candidate that conveys the most information from the given instruction, and return it as a list in the format [function1]. """
    prompt += """Learn from the examples provided on how to choose candidates with maximum informative content. \nIncontext example: {similar_queries}"""
    prompt += """\n---\n"""
    prompt += """Now finish the selection task for the below instruction. Never mess up with the examples above. """
    prompt += """\nfunction candidates:\n{retrieved_apis}\nInstruction: {query}
Function: """
    return prompt

def get_nonretrieved_prompt():
    prompt = """Task: Select the candidate that conveys the most information from the given instruction, and return it as a list in the format [function1]. """
    prompt += """Learn from the examples provided on how to choose candidates with maximum informative content. \nIncontext example: {similar_queries}"""
    prompt += """\n---\n"""
    prompt += """Now finish the selection task for the below instruction. Never mess up with the examples above. """
    prompt += """\nInstruction: {query}
Function: """
    return prompt

def get_all_api_json(API_init_path):
    with open(API_init_path, 'r') as file:
        API_init = json.load(file)
    end_of_docstring_summary = re.compile(r'[{}\n]+'.format(re.escape(punctuation)))
    all_apis = {}
    for api_name in API_init:
        x = API_init[api_name]
        if x['api_type']=='class':
            continue
        if x['description']:
            #description = x['description'].split('\n')[0]
            description = x['description']
        else:
            description = end_of_docstring_summary.split(x['Docstring'])[0].strip()
        all_apis[api_name] = description
    all_apis = list(all_apis.items())
    all_apis_json = {i[0]:i[1] for i in all_apis}
    #print('len', len(all_apis), len(all_apis_json))
    return all_apis, all_apis_json

def find_similar_api_pairs(api_descriptions):
    descriptions = list(api_descriptions.values())
    api_names = list(api_descriptions.keys())
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    similar_pairs = []
    for i in range(len(api_names)):
        for j in range(i+1, len(api_names)):
            if cosine_similarities[i, j] >= 1:  # threshold can be adjusted
                similar_pairs.append((api_names[i], api_names[j]))
    return similar_pairs

def is_pair_in_merged_pairs(gold, pred, merged_pairs):
    # Check if the pair (gold, pred) or (pred, gold) exists in merged_pairs
    return (gold, pred) in merged_pairs or (pred, gold) in merged_pairs

def load_errors(fname):
    with open(fname, 'rt') as f:
        data = json.load(f)
    wrong = [ex for ex in data if not ex['correct']]
    random.Random(0).shuffle(wrong)
    return wrong

def correct_entries(res, lib_name):
    for entry in res:
        # Directly obtaining the substring starting from "scanpy"
        if 'pred' in entry and entry['pred'] is not None:
            if '(' in entry['pred'][0] and lib_name in entry['pred'][0]:
                entry['pred'] = [i.split('(')[0] for i in entry['pred']]
            if (entry['pred'][0].startswith('candidate') or entry['pred'][0].startswith('Candidate') or entry['pred'][0].startswith('Function')) and (lib_name in entry['pred'][0]):
                pred_substrings = [pred[pred.find(lib_name):] for pred in entry['pred'] if lib_name in pred]
                # Updating 'correct' based on whether modified 'gold' substring is in modified 'pred' substrings
                if len(pred_substrings)>0:
                    entry['correct'] = entry['gold'] == pred_substrings[0]
                else:
                    entry['correct'] = False
                entry['pred'] = pred_substrings
            else:
                pred_substrings = [pred[pred.find(lib_name):] for pred in entry['pred'] if lib_name in pred]
                #if (entry['gold'] in pred_substrings)!=entry['correct']:
                #    print(entry['correct'], entry['gold'], pred_substrings, entry['pred'])
                if len(pred_substrings)>0:
                    entry['correct'] = entry['gold'] == pred_substrings[0]
                else:
                    entry['correct'] = False
                #entry['pred'] = pred_substrings
    return res

def extract_specific_inquiries(data, desired_indices):
    """
    Extracts specific inquiries for each API based on given indices from a dataset.
    
    This function organizes inquiries by API names and selects specific inquiries
    based on provided indices. It requires that each API has at least 8 inquiries
    to select the specified inquiries (e.g., 2nd and 7th inquiries).
    """
    subset = []
    api_inquiries = {}
    for item in data:
        api_name = item['gold']
        if api_name not in api_inquiries:
            api_inquiries[api_name] = []
        api_inquiries[api_name].append(item)
    for inquiries in api_inquiries.values():
        if len(inquiries) >= 8:
            subset.append(inquiries[desired_indices[0]])
            subset.append(inquiries[desired_indices[1]])
    return subset

def extract_random_inquiries(data, desired_num=2):
    """
    Extracts a random set of inquiries for each API from a dataset, based on the desired number.
    
    This function organizes inquiries by API names and randomly selects a specified number
    of inquiries for each API, provided it has at least 8 inquiries. This is intended to create
    a diverse subset of inquiries for analysis or training purposes.
    """
    subset = []
    api_inquiries = {}
    for item in data:
        api_name = item['gold']
        if api_name not in api_inquiries:
            api_inquiries[api_name] = []
        api_inquiries[api_name].append(item)
    for inquiries in api_inquiries.values():
        if len(inquiries) >= 8:
            random_indices = random.sample(range(0, len(inquiries)), desired_num)
            for index in random_indices:
                subset.append(inquiries[index])
    return subset

def generate_seed(api_name):
    """
    Generates a seed value from an API name using MD5 hashing.

    Parameters:
        api_name (str): The name of the API.

    Returns:
        int: An integer seed value generated from the API name.
    """
    hash_value = hashlib.md5(api_name.encode()).hexdigest()
    return int(hash_value[:8], 16)

def get_sampled_shuffled(api_name, shuffled_list, num_samples=5):
    seed = generate_seed(api_name)
    random.seed(seed)
    sampled_shuffled = random.sample(shuffled_list, num_samples)
    return sampled_shuffled

def correct_pred(pred, lib_name):
    """
    Extracts the library name from the prediction string.
    """
    if '(' in pred and lib_name in pred:
        pred = pred.split('(')[0]
    #if (pred.startswith('candidate') or pred.startswith('Candidate') or pred.startswith('Function')) and (lib_name in pred):
    #    if lib_name in pred:
    #        ans = pred[pred.find(lib_name):]
    #else:
    #    if lib_name in pred:
    #        ans = pred[pred.find(lib_name):]
    if lib_name in pred:
        ans = pred[pred.find(lib_name):]
    else:
        ans = pred
    ans = ans.strip()
    return ans

def generate_custom_val_indices(api_ranges):
    """
    Generate a list of custom validation indices from a list of API ranges.
    """
    val_indices = []
    for start, end in api_ranges:
        val_indices.extend([start, end])
    return val_indices

def standardize_parameters(api):
    """
    Standardizes parameters of an API by extracting and sorting parameter names.
    """
    a =  {
        param_name: {
            "type": details["type"],
            #"default": details["default"],
            #"optional": details["optional"]
        }
        for param_name, details in sorted(api["Parameters"].items())
    }
    return sorted(a.keys())

def compare_apis(api1, api2):
    """
    example: 
    api1 = API_init["scanpy.pp.log1p"]
    api2 = API_init["scanpy.pp.log1p"]
    are_parameters_identical = compare_apis(api1, api2)
    """
    params1 = standardize_parameters(api1)
    params2 = standardize_parameters(api2)
    return params1 == params2

def find_matching_api_pairs(api_data, threshold=5):
    """
    Finds pairs of APIs in the given data that have matching standardized parameters.
    """
    # Standardize APIs
    standardized_apis = {api_name: standardize_parameters(api) for api_name, api in api_data.items()}
    # List to store matching pairs
    matching_pairs = []
    # Get list of API names
    api_names = list(standardized_apis.keys())
    # Iterate through API names and find matching pairs
    for i in range(len(api_names)):
        for j in range(i + 1, len(api_names)):
            # Check if both APIs have enough parameters for comparison
            if len(standardized_apis[api_names[i]]) > threshold and len(standardized_apis[api_names[j]]) > threshold:
                # Compare standardized parameters
                if standardized_apis[api_names[i]] == standardized_apis[api_names[j]]:
                    matching_pairs.append((api_names[i], api_names[j]))
    return matching_pairs
