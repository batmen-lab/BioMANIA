"""
Author: Zhengyuan Dong
Date Created: May 06, 2024
Last Modified: May 21, 2024
Description: compare the tutorial summary query and single query retrieval results
"""

import argparse
import json

from ..models.dialog_classifier import Dialog_Gaussian_classification
from ..gpt.utils import load_json, save_json
from ..inference.retriever_finetune_inference import ToolRetriever
from ..inference.utils import find_differences

def load_and_process_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    processed_data = []
    for key, value in data.items():
        goal_description = value.get('goal_description')
        case_autoba = value.get('case_AutoBA')
        case_ours = value.get('case_Ours')
        # You can add more fields here if needed
        processed_data.append({
            'query_id': key,
            'query': goal_description,
            'case_autoba': case_autoba,
            'case_ours': case_ours
        })
    return processed_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--LIB', type=str, required=True, help='Library name')
    parser.add_argument('--retrieved_api_nums', type=int, default=3, help='Number of retrieved APIs')
    parser.add_argument('--threshold', type=float, default=0.05, help='Threshold for p-value')
    args = parser.parse_args()
    threshold = args.threshold
    
    # Load and process tutorial summary data, and autoba and ours plan answer
    json_path = f"examples_llama2_promptv5/{args.LIB}/analysis_results.json"
    tutorial_summary_query = load_and_process_data(json_path)
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
    print(len(train_data), len(val_data))
    
    
    
    classifer = Dialog_Gaussian_classification(LIB=args.LIB, threshold=threshold)
    
    data_source = "single_query_train"
    scores_train, outliers = classifer.compute_accuracy_filter_compositeAPI(retriever, train_data, args.retrieved_api_nums, name=data_source)
    
    data_source = "single_query_val"
    scores_val, outliers = classifer.compute_accuracy_filter_compositeAPI(retriever, val_data, args.retrieved_api_nums, name=data_source)

    data_source = "single_query_test"
    scores_test, outliers = classifer.compute_accuracy_filter_compositeAPI(retriever, diff_data, args.retrieved_api_nums, name=data_source)

    # Retrieve and calculate scores for each query
    data_source = "tutorial_summary_query"
    scores_tutorial, outliers = classifer.compute_accuracy_filter_compositeAPI(retriever, tutorial_summary_query, args.retrieved_api_nums, name=data_source)
    
    mean, std = classifer.fit_gaussian(scores_train['rank_1'])
    predictions_tutorial = classifer.classify(scores_tutorial['rank_1'])
    predictions_val = classifer.classify(scores_val['rank_1'])
    predictions_test = classifer.classify(scores_test['rank_1'])
    predictions_train = classifer.classify(scores_train['rank_1'])
    
    accuracy = classifer.compute_acc([0 for _ in predictions_val]+[0 for _ in predictions_test]+[1 for _ in predictions_tutorial], predictions_val+predictions_test+predictions_tutorial)
    print(f'{args.LIB} total accuracy is {accuracy}')
    val_accuracy = classifer.compute_acc([0 for _ in predictions_val], predictions_val)
    print(f'{args.LIB} val accuracy is {val_accuracy}')
    test_accuracy = classifer.compute_acc([0 for _ in predictions_test], predictions_test)
    print(f'{args.LIB} test accuracy is {test_accuracy}')
    tutorial_accuracy = classifer.compute_acc([1 for _ in predictions_tutorial], predictions_tutorial)
    print(f'{args.LIB} tutorial accuracy is {tutorial_accuracy}')
    train_accuracy = classifer.compute_acc([0 for _ in predictions_train], predictions_train)
    print(f'{args.LIB} train accuracy is {train_accuracy}')

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == "__main__":
    main()

