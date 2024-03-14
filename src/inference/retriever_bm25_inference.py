import json, ast

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--top_k', type=int, default=3, help='')
parser.add_argument('--LIB', type=str, help='')
args = parser.parse_args()

from inference.utils import find_similar_two_pairs

# step1: Instruction Generation, compress API, build prompt
with open(f'./data/standard_process/{args.LIB}/API_composite.json', 'r') as f:
    ori_data = json.load(f)
with open(f'./data/standard_process/{args.LIB}/API_inquiry_annotate.json', 'r') as f:
    results = json.load(f)
with open(f"./data/standard_process/{args.LIB}/API_instruction_testval_query_ids.json", "r") as f:
    index_file = json.load(f)
    test_ids = index_file['test']
    val_ids = index_file['val']

# Step2: LLM inference for checking
import json
import ast
from rank_bm25 import BM25Okapi
from retrievers import *
from models.model import *

# prepare corpus
def prepare_corpus(ori_data):
    corpus = [
        {
            'api_name': api_name,
            'description': api_info['description'],
            'Parameters': api_info['Parameters'],
            'returns': api_info['Returns'],
        }
        for api_name, api_info in ori_data.items()
    ]
    return corpus,[str(doc).split(" ") for doc in corpus]
def get_query_pairs(results):
    return [(entry['query'], entry['api_name']) for entry in results]

corpus, tokenized_corpus = prepare_corpus(ori_data)
# Separate query_pairs into train and test based on the test IDs
train_results = [entry for entry in results if entry['query_id'] not in test_ids and entry['query_id'] not in val_ids]
test_results = [entry for entry in results if entry['query_id'] in test_ids]
val_results = [entry for entry in results if entry['query_id'] in val_ids]

query_pairs = get_query_pairs(results)
train_query_pairs = get_query_pairs(train_results)
test_query_pairs = get_query_pairs(test_results)
val_query_pairs = get_query_pairs(val_results)

bm25 = BM25Okapi(tokenized_corpus)
retriever = BM25Retriever(index=bm25, corpus=corpus, query_kwargs={"similarity_top_k": args.top_k})

import ast
merged_pairs = find_similar_two_pairs(args.LIB)

def evaluate_query_pairs(query_pairs_set, name):
    retriever_correct_list = []
    ambiguous_correct_list = []
    retrieved_docs_list = []
    for query, target_api in query_pairs_set:
        retrieved_docs = retriever.get_relevant_documents(query)
        api_name_list = [ast.literal_eval(retrieved_doc.page_content).get('api_name') for retrieved_doc in retrieved_docs]
        retrieved_docs_list.append((target_api, api_name_list))
        if target_api in api_name_list:
            retriever_correct_list.append(target_api)
        else:
            for retrieved_api in api_name_list:
                if (target_api, retrieved_api) in merged_pairs or (retrieved_api, target_api) in merged_pairs:
                    ambiguous_correct_list.append(target_api)
                    break
    retrieve_rate = len(retriever_correct_list) / len(query_pairs_set)
    ambiguous_retrieve_rate = (len(retriever_correct_list) + len(ambiguous_correct_list)) / len(query_pairs_set)
    print(f"Totally tested {len(query_pairs_set)} {name} queries!")
    print(f"{name} retriever top-{args.top_k} accuracy rate: {retrieve_rate * 100:.2f}%")
    print(f"{name} retriever top-{args.top_k} ambiguous accuracy rate: {ambiguous_retrieve_rate * 100:.2f}%")

# Evaluate for train and test sets
evaluate_query_pairs(train_query_pairs, "train")
evaluate_query_pairs(val_query_pairs, "val")
evaluate_query_pairs(test_query_pairs, "test")
