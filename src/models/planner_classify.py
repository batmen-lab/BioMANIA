"""
First created: 2024/09/12
Last modified: 2024/09/12
Main target:
    - Classify the API query based on the retrieved API ranks, using Gaussian distribution
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score

import argparse
from ..inference.retriever_finetune_inference import ToolRetriever
from ..gpt.utils import load_json, save_json
from ..inference.utils import find_differences
import json
from ..models.dialog_classifier import Dialog_Gaussian_classification
import random

np.random.seed(42)
random.seed(42)

def fit_gaussian_distribution(data):
    # compute mean and covariance matrix
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return multivariate_normal(mean=mean, cov=cov)

def train_gaussian_model(train_set_label0, train_set_label1):
    # fit label0's gaussian distribution
    gaussian_label0 = fit_gaussian_distribution(train_set_label0)
    # fit label1's gaussian distribution
    gaussian_label1 = fit_gaussian_distribution(train_set_label1)
    return gaussian_label0, gaussian_label1

# predict on validation set using the trained gaussian model
def predict_gaussian_model(val_set, gaussian_label0, gaussian_label1):
    probabilities_label0 = gaussian_label0.pdf(val_set)  # compute the probability of label0
    probabilities_label1 = gaussian_label1.pdf(val_set)  # compute the probability of label1
    predicted_labels = np.where(probabilities_label0 > probabilities_label1, 0, 1)  # compare the probabilities
    return predicted_labels

# evaluate the model on validation set
def evaluate_model(val_set, val_labels, gaussian_label0, gaussian_label1):
    predicted_labels = predict_gaussian_model(val_set, gaussian_label0, gaussian_label1)
    accuracy = accuracy_score(val_labels, predicted_labels)
    return accuracy

def calculate_acc_for_dimensions(train_set_label0, train_set_label1, val_set_label0, val_set_label1):
    for dims in range(1, 4):
        print(f"\nCalculating accuracy for {dims} dimensions:")
        # extract the specified dimensions of the training set
        train_set_label0_dims = train_set_label0[:, :dims]
        train_set_label1_dims = train_set_label1[:, :dims]
        # extract the specified dimensions of the validation set
        val_set_label0_dims = val_set_label0[:, :dims]
        val_set_label1_dims = val_set_label1[:, :dims]
        # combine the training set
        val_set = np.vstack((val_set_label0_dims, val_set_label1_dims))
        val_labels = np.hstack((np.zeros(val_set_label0_dims.shape[0]), np.ones(val_set_label1_dims.shape[0])))
        # fit the gaussian model
        gaussian_label0, gaussian_label1 = train_gaussian_model(train_set_label0_dims, train_set_label1_dims)
        # compute the accuracy on the training set
        train_set = np.vstack((train_set_label0_dims, train_set_label1_dims))
        train_labels = np.hstack((np.zeros(train_set_label0_dims.shape[0]), np.ones(train_set_label1_dims.shape[0])))
        train_acc = evaluate_model(train_set, train_labels, gaussian_label0, gaussian_label1)*100
        print(f"Train Accuracy for {dims} dimensions: {train_acc:.2f}")
        # compute the accuracy on the validation set
        val_acc = evaluate_model(val_set, val_labels, gaussian_label0, gaussian_label1)*100
        print(f"Validation Accuracy for {dims} dimensions: {val_acc:.2f}")

def generate_set1(api_queries):
    # same API 2 choose 1
    set1 = []
    for api, data in api_queries.items():
        if data['val'] and data['test']:
            selected_val_query = random.choice(data['val'])
            selected_test_query = random.choice(data['test'])
            set1.append({
                'api': api,
                'query': selected_val_query
            })
            set1.append({
                'api': api,
                'query': selected_test_query
            })
        else:
            raise ValueError(f"API {api} has no queries")
    return set1

def generate_set2(api_queries):
    # same API 5 choose 3
    set2 = []
    for api, data in api_queries.items():
        combined_queries = data['val'] + data['test']
        if len(combined_queries) >= 5:
            for _ in range(2):
                selected_queries = random.sample(combined_queries, 5)
                sample_three = ' '.join(random.sample(selected_queries, 3))
                set2.append({
                    'api': api,
                    'query': sample_three
                })
        else:
            raise ValueError(f"API {api} has less than 5 queries")
    return set2

def generate_set3(api_queries):
    # different API 5 choose 3
    set3 = []
    times = len(api_queries)*2
    for _ in range(times):
        selected_queries = []
        apis = list(api_queries.keys())
        selected_apis = random.sample(apis, min(3, len(apis)))
        for api in selected_apis:
            combined_queries = api_queries[api]['val'] + api_queries[api]['test']
            if combined_queries:
                selected_query = random.choice(combined_queries)
                selected_queries.append(selected_query)
            else:
                raise ValueError(f"No queries found for API {api}")
        if len(selected_queries) == 3:
            combined_query = ' '.join(selected_queries)
            set3.append({
                'api': '+'.join(selected_apis),
                'query': combined_query
            })
        else:
            raise ValueError(f"Selected queries: {selected_queries}")
    return set3

def combine_val_test(val_data, test_data):
    api_queries = {}
    for entry in val_data:
        for api in entry['api_calling']:
            if api not in api_queries:
                api_queries[api] = {'val': [], 'test': []}
            api_queries[api]['val'].append(entry['query'])
    for entry in test_data:
        for api in entry['api_calling']:
            if api not in api_queries:
                api_queries[api] = {'val': [], 'test': []}
            api_queries[api]['test'].append(entry['query'])
    return api_queries

def combine_train(train_data):
    api_queries = {}
    for entry in train_data:
        for api in entry['api_calling']:
            if api not in api_queries:
                api_queries[api] = {'val': [], 'test': []}
            api_queries[api]['val'].append(entry['query']) # we will combine val and test to sample set2 and set3, so it is ok to only append entry here
    return api_queries

def extract_ranks(data):
    ranks = []
    for i in range(1, 6):  
        rank_key = f'rank_{i}'
        if rank_key in data:
            ranks.append(data[rank_key])
    return np.array(ranks).T

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--LIB', type=str, required=True, help='Library name')
    parser.add_argument('--retrieved_api_nums', type=int, default=3, help='Number of retrieved APIs')
    parser.add_argument('--threshold', type=float, default=0.05, help='Threshold for p-value')
    args = parser.parse_args()
    threshold = args.threshold
    # Load and process tutorial summary data, and autoba and ours plan answer
    #json_path = f"examples_llama2_promptv5/{args.LIB}/analysis_results.json"
    #tutorial_summary_query = load_and_process_data(json_path)
    # Initialize retriever
    corpus_tsv_path = f"./data/standard_process/{args.LIB}/retriever_train_data/corpus.tsv"
    retrieval_model_path = f"./hugging_models/retriever_model_finetuned/{args.LIB}/assigned"
    retriever = ToolRetriever(LIB=args.LIB, corpus_tsv_path=corpus_tsv_path, model_path=retrieval_model_path)
    # get test data
    annotate_path = f'data/standard_process/{args.LIB}/API_inquiry_annotate.json'
    api_path = f'data/standard_process/{args.LIB}/API_inquiry.json'
    annotate_data = load_json(annotate_path)
    api_data = load_json(api_path)
    diff_data = find_differences(annotate_data, api_data)
    # get val and train data
    with open(f"data/standard_process/{args.LIB}/API_instruction_testval_query_ids.json", 'r') as f:
        index_data = json.load(f)
    test_ids = index_data['test']
    val_ids = index_data['val']
    train_data = [data for data in api_data if data['query_id'] not in test_ids and data['query_id'] not in val_ids]
    val_data = [data for data in api_data if data['query_id'] in val_ids]
    assert len(train_data)/8 == len(val_data)/2

    # control set, same API 5 choose 3, different API 5 choose 3
    total_query_train = combine_train(train_data)
    #set1 = generate_set1(total_query)
    #assert len(set1) ==2*len(total_query), len(set1)
    set2_train = generate_set2(total_query_train)
    assert len(set2_train) == 2*len(total_query_train), len(set2_train)
    set3_train = generate_set3(total_query_train)
    assert len(set3_train) == 2*len(total_query_train), len(set3_train)

    # control set, same API 5 choose 3, different API 5 choose 3
    total_query_val = combine_val_test(val_data, diff_data)
    #set1 = generate_set1(total_query)
    #assert len(set1) ==2*len(total_query), len(set1)
    set2_val = generate_set2(total_query_val)
    assert len(set2_val) == 2*len(total_query_val), len(set2_val)
    set3_val = generate_set3(total_query_val)
    assert len(set3_val) == 2*len(total_query_val), len(set3_val)

    classifer = Dialog_Gaussian_classification(LIB=args.LIB, threshold=threshold)
    data_source = "same_api_train"
    same_api_train, outliers = classifer.compute_accuracy_filter_compositeAPI(retriever, set2_train, args.retrieved_api_nums, name=data_source)
    data_source = "different_api_train"
    different_api_train, outliers = classifer.compute_accuracy_filter_compositeAPI(retriever, set3_train, args.retrieved_api_nums, name=data_source)
    data_source = "same_api_val"
    same_api_val, outliers = classifer.compute_accuracy_filter_compositeAPI(retriever, set2_val, args.retrieved_api_nums, name=data_source)
    data_source = "different_api_val"
    different_api_val, outliers = classifer.compute_accuracy_filter_compositeAPI(retriever, set3_val, args.retrieved_api_nums, name=data_source)

    val_set_label0 = extract_ranks(same_api_val)  # extract data from same_api_val
    val_set_label1 = extract_ranks(different_api_val)  # extract data from different_api_val
    train_set_label0 = extract_ranks(same_api_train)  # extract data from same_api_train
    train_set_label1 = extract_ranks(different_api_train)  # extract data from different_api_train

    # combine the training set
    val_set = np.vstack((val_set_label0, val_set_label1))
    val_labels = np.hstack((np.zeros(val_set_label0.shape[0]), np.ones(val_set_label1.shape[0])))
    print(val_set.shape, val_labels.shape)

    # fit the gaussian model
    gaussian_label0, gaussian_label1 = train_gaussian_model(train_set_label0, train_set_label1)
    print(gaussian_label0)

    # evaluate the model on the validation set
    """accuracy = evaluate_model(val_set, val_labels, gaussian_label0, gaussian_label1)
    print(f"Validation Accuracy: {accuracy:.2f}")
    """

    calculate_acc_for_dimensions(train_set_label0, train_set_label1, val_set_label0, val_set_label1)
