import random, re, hashlib, json
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
import matplotlib.pyplot as plt
import numpy as np

def save_json(output_path: str, results: dict) -> None:
    """
    Save results to a JSON file with indentation of 4 spaces.

    Parameters
    ----------
    output_path : str
        Path to the output file.
    results : dict
        Dictionary containing the results to be saved.

    Returns
    -------
    None

    Examples
    --------
    >>> save_json('output.json', {'key': 'value'})
    """
    with open(output_path, 'w') as file:
        json.dump(results, file, indent=4)

def load_json(filename: str) -> dict:
    """
    Load JSON data from a specified file.

    Parameters
    ----------
    filename : str
        The path to the JSON file to be loaded.

    Returns
    -------
    dict
        The data loaded from the JSON file.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def get_generate_prompt():
    # added 240415
    prompt = """Task: choose only one most appropriate function from the PyPI {lib_name} library for the instruction and return the function name as a list in the format ["function1"] without any arguments."""
    prompt += """\n---\nHere are some answer examples you can refer."""
    prompt += """\nIncontext examples: {similar_queries}"""
    prompt += """\n---\n"""
    prompt += """Now answer the below Instruction with one most appropriate function from the lib {lib_name}. Never answer API from above examples. Use real function whose names start with the {lib_name}, and which is truly in this lib"""
    prompt += """\nInstruction: {query}
Function: 
    """
    return prompt

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

def get_first_sentence(text):
    sentence_end_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')
    sentences = sentence_end_pattern.split(text, 1)
    return sentences[0] if sentences else ''

def get_all_api_json(API_init_path, mode='full'):
    API_init = load_json(API_init_path)
    end_of_docstring_summary = re.compile(r'[{}\n]+'.format(re.escape(punctuation)))
    all_apis = {}
    for api_name in API_init:
        x = API_init[api_name]
        if x['api_type']=='class' or x['api_type']=='unknown':
            continue
        if mode == 'full':
            # design for generate instructions, as more information will help for accurate instruction
            if x['description']:
                #description = x['description'].split('\n')[0]
                description = x['description']
            else:
                # 240320: we filter out the API from the data prepare period, so we assume we won't include API for the evaluation
                #description = end_of_docstring_summary.split(x['Docstring'])[0].strip()
                description = ""
        else:
            # design for ambiguity check
            #description = end_of_docstring_summary.split(x['Docstring'])[0].strip()
            description = get_first_sentence(x['description'])
        # notice that sometimes we will obtain `Parameters` if author didn't write any description in docstring
        all_apis[api_name] = description
    all_apis = list(all_apis.items())
    all_apis_json = {i[0]:i[1] for i in all_apis}
    return all_apis, all_apis_json

def find_similar_api_pairs(api_descriptions):
    from sklearn.metrics.pairwise import linear_kernel
    # filter out apis with empty descriptions and not meaningful descriptions
    filtered_api_descriptions = {api: desc for api, desc in api_descriptions.items() if desc.strip() and desc.strip()!='Parameters'}
    print(len(api_descriptions), len(filtered_api_descriptions))
    if len(filtered_api_descriptions) <= 1:
        return []
    api_descriptions = filtered_api_descriptions
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

def get_ambiguous_pairs(API_init_path=f"./data/standard_process/scanpy/API_init.json"):
    # For accuracy without ambiguous pair
    from collections import defaultdict
    api_data = load_json(API_init_path)
    api_data = {key:api_data[key] for key in api_data if api_data[key]['api_type']!='class' and api_data[key]['api_type']!='unknown'}
    all_apis, all_apis_json = get_all_api_json(API_init_path, mode='full') # single
    similar_api_pairs = find_similar_api_pairs(all_apis_json)
    require_same_depth = False
    api_list = list(api_data.keys())
    groups = defaultdict(list)
    for api in api_list:
        parts = api.split('.')
        if require_same_depth:
            key = (parts[-1], len(parts))
        else:
            key = parts[-1]
        groups[key].append(api)
    similar_pairs = [group for group in groups.values() if len(group) > 1]
    list_1 = similar_api_pairs
    list_2 = similar_pairs
    pairs_from_list_2 = [(apis[i], apis[j]) for apis in list_2 for i in range(len(apis)) for j in range(i+1, len(apis))]
    print('information of the ambiguous pair:', len(list_1), len(pairs_from_list_2))
    merged_pairs = list(set(list_1 + pairs_from_list_2))
    return merged_pairs, similar_api_pairs, pairs_from_list_2

def find_similar_two_pairs(API_init_path=f"./data/standard_process/scanpy/API_init.json"):
    merged_pairs, _, _ = get_ambiguous_pairs(API_init_path)
    return merged_pairs

def is_pair_in_merged_pairs(gold, pred, merged_pairs):
    # Check if the pair (gold, pred) or (pred, gold) exists in merged_pairs
    return (gold, pred) in merged_pairs or (pred, gold) in merged_pairs

def load_errors(fname):
    data = load_json(fname)
    wrong = [ex for ex in data if not ex['correct']]
    random.Random(0).shuffle(wrong)
    return wrong

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
    if ans.endswith(']'):
        ans = ans.replace(']','')
    return ans

def correct_entries(res, lib_name, pred_key='pred'):
    # no worry for multiple answer as we only treat the first candidate as true answer!
    for entry in res:
        # Directly obtaining the substring starting from "scanpy"
        if pred_key in entry and entry[pred_key] is not None:
            if '(' in entry[pred_key][0] and lib_name in entry[pred_key][0]:
                entry[pred_key] = [i.split('(')[0] for i in entry[pred_key]]
            if (entry[pred_key][0].startswith('candidate') or entry[pred_key][0].startswith('Candidate') or entry[pred_key][0].startswith('Function')) and (lib_name in entry[pred_key][0]):
                pred_substrings = [pred[pred.find(lib_name):] for pred in entry[pred_key] if lib_name in pred]
                # Updating 'correct' based on whether modified 'gold' substring is in modified 'pred' substrings
                if len(pred_substrings)>0:
                    # we only consider the first candidate
                    entry['correct'] = entry['gold'] == pred_substrings[0]
                else:
                    entry['correct'] = False
                entry[pred_key] = pred_substrings
            else:
                pred_substrings = [pred[pred.find(lib_name):] for pred in entry[pred_key] if lib_name in pred]
                #if (entry['gold'] in pred_substrings)!=entry['correct']:
                #    print(entry['correct'], entry['gold'], pred_substrings, entry[pred_key])
                if len(pred_substrings)>0:
                    entry['correct'] = entry['gold'] == pred_substrings[0]
                else:
                    entry['correct'] = False
                #entry[pred_key] = pred_substrings
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

def plot_figure(cluster_data, LIB):
    categories = [
        "Multi-label\nclassification\nw/\nretriever",
        "GPT-3.5-turbo\ngeneration",
        "GPT-3.5-turbo\nclassification\nw/o\nretriever",
        "GPT-3.5-turbo\nclassification\nw/\nretriever",
        "GPT-4-turbo\nclassification\nretriever"
    ]
    legend_labels = [
        "Synthetic instruction",
        "Annotated instruction",
        "Synthetic instruction w/ ambiguity removal",
        "Annotated instruction w/ ambiguity removal"
    ]
    # Set up the bar chart
    num_categories = len(categories)
    num_conditions = len(legend_labels)
    bar_width = 0.15  # Reduce bar width to fit all bars
    index = np.arange(num_categories)
    fig, ax = plt.subplots(figsize=(14, 9))
    for i in range(num_conditions):
        bars = ax.bar(index + i * bar_width, cluster_data[i], bar_width, label=legend_labels[i])
        # Annotate the bars with the data values
        for rect in bars:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    # Set the labels and titles
    ax.set_title(f'Prediction Accuracy for {LIB}', fontsize=14)
    ax.set_xticks(index + bar_width * num_conditions / 2 - bar_width / 2)
    ax.set_xticklabels(categories, fontsize=10)
    # Place the legend outside of the plot
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=10)
    # Display the plot
    plt.xticks()  # Rotate the x-axis labels to show them more clearly
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the layout to fit everything except the legend
    plt.savefig(f'./gpt_{LIB}.pdf')
    plt.show()

def extract_and_print_adjusted(df):
    results = []
    # GPT-3.5 with top_k '-' (this corresponds to the first row requested)
    filtered_df_1 = df[(df['model_name'] == 'gpt-3.5') & (df['top_k'] == '-')]
    synthetic_val_acc_1 = filtered_df_1[filtered_df_1['test_val'] == 'synthetic_val']['accuracy'].iloc[0]
    synthetic_val_fil_acc_1 = filtered_df_1[filtered_df_1['test_val'] == 'synthetic_val']['filtered_accuracy'].iloc[0]
    human_annotate_acc_1 = filtered_df_1[filtered_df_1['test_val'] == 'human annotate']['accuracy'].iloc[0]
    human_annotate_fil_acc_1 = filtered_df_1[filtered_df_1['test_val'] == 'human annotate']['filtered_accuracy'].iloc[0]
    results.append([0, synthetic_val_acc_1, human_annotate_acc_1, synthetic_val_fil_acc_1, human_annotate_fil_acc_1])
    # GPT-3.5 with nonretrieved (this corresponds to the second row requested)
    filtered_df_2 = df[(df['model_name'] == 'gpt-3.5') & (df['retrieval_status'] == 'nonretrieved')]
    synthetic_val_acc_2 = filtered_df_2[filtered_df_2['test_val'] == 'synthetic_val']['accuracy'].iloc[0]
    synthetic_val_fil_acc_2 = filtered_df_2[filtered_df_2['test_val'] == 'synthetic_val']['filtered_accuracy'].iloc[0]
    human_annotate_acc_2 = filtered_df_2[filtered_df_2['test_val'] == 'human annotate']['accuracy'].iloc[0]
    human_annotate_fil_acc_2 = filtered_df_2[filtered_df_2['test_val'] == 'human annotate']['filtered_accuracy'].iloc[0]
    results.append([0, synthetic_val_acc_2, human_annotate_acc_2, synthetic_val_fil_acc_2, human_annotate_fil_acc_2])
    # GPT-3.5 with retrieved (this corresponds to the third row requested)
    filtered_df_3 = df[(df['model_name'] == 'gpt-3.5') & (df['retrieval_status'] == 'retrieved') & (df['top_k'] != '-')]
    synthetic_val_acc_3 = filtered_df_3[filtered_df_3['test_val'] == 'synthetic_val']['accuracy'].iloc[0]
    synthetic_val_fil_acc_3 = filtered_df_3[filtered_df_3['test_val'] == 'synthetic_val']['filtered_accuracy'].iloc[0]
    human_annotate_acc_3 = filtered_df_3[filtered_df_3['test_val'] == 'human annotate']['accuracy'].iloc[0]
    human_annotate_fil_acc_3 = filtered_df_3[filtered_df_3['test_val'] == 'human annotate']['filtered_accuracy'].iloc[0]
    results.append([0, synthetic_val_acc_3, human_annotate_acc_3, synthetic_val_fil_acc_3, human_annotate_fil_acc_3])
    # GPT-4 with retrieved (this corresponds to the fourth row requested)
    filtered_df_4 = df[(df['model_name'] == 'gpt-4') & (df['retrieval_status'] == 'retrieved')]
    synthetic_val_acc_4 = filtered_df_4[filtered_df_4['test_val'] == 'synthetic_val']['accuracy'].iloc[0]
    synthetic_val_fil_acc_4 = filtered_df_4[filtered_df_4['test_val'] == 'synthetic_val']['filtered_accuracy'].iloc[0]
    human_annotate_acc_4 = filtered_df_4[filtered_df_4['test_val'] == 'human annotate']['accuracy'].iloc[0]
    human_annotate_fil_acc_4 = filtered_df_4[filtered_df_4['test_val'] == 'human annotate']['filtered_accuracy'].iloc[0]
    results.append([0, synthetic_val_acc_4, human_annotate_acc_4, synthetic_val_fil_acc_4, human_annotate_fil_acc_4])
    # Print the results and convert the accuracy values to percentages rounded to two decimals
    transposed_results = list(zip(*results))  # Transpose the list of lists
    cluster_data = []
    for i in range(1, len(transposed_results)):
        new_row = [0] + [round(val * 100, 2) if isinstance(val, float) else val for val in transposed_results[i]]
        cluster_data.append(new_row)
    return np.array(cluster_data)

def extract_random_inquiries(data, desired_num=2):
    # extract 4th, 7th from 10 inquiries to obtain subset
    # usage: train_sub2 = extract_random_inquiries(train_remain, desired_num=2)
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

def extract_specific_inquiries(data, desired_indices):
    # deprecated
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

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == '__main__':
    #merged_pairs = find_similar_two_pairs()
    #print(len(merged_pairs))
    #all_apis, all_apis_json = get_all_api_json(, mode='single')
    #print(len(all_apis), len(all_apis_json))
    for lib in ['scanpy', 'squidpy', 'ehrapy', 'snapatac2']: #
        all_apis, all_apis_json = get_all_api_json(f"data/standard_process/{lib}/API_init.json", mode='single')
        #print(all_apis_json['ehrapy.tools.tsne'], all_apis_json['ehrapy.tools.test_kmf_logrank'])
        merged_pairs,a,b = get_ambiguous_pairs(f"data/standard_process/{lib}/API_init.json")
        print('-'*10)
        for item in a:
            print(item[0], ':',  all_apis_json[item[0]])
            print(item[1], ':',  all_apis_json[item[1]])
        """print('-'*10)
        print(b)"""
    