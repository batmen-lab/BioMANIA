import json
import ast
import random
from transformers import LongformerTokenizer, LongformerForSequenceClassification
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--top_k', type=int, help='')
parser.add_argument('--LIB', type=str, help='')
args = parser.parse_args()

class Longformer_Reranker:
    def __init__(self, model_name='allenai/longformer-base-4096'):
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode

    def score(self, query, doc):
        inputs = self.tokenizer.encode_plus(query, doc, return_tensors="pt", max_length=4096, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits[0, 1].item()  # Assuming class 1 is the "relevant" class

    def rerank(self, query, docs, top_n=5):
        scores = [(doc, self.score(query, doc['description'])) for doc in docs]
        sorted_docs = sorted(scores, key=lambda x: x[1], reverse=True)
        return [doc[0]['api_name'] for doc in sorted_docs[:top_n]]  # Return the API names of the top N scoring documents

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
# ask LLM
using_reranker='' #longerformer or gpt

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
llm, tokenizer = LLM_model()
if using_reranker=='longerformer':
    reranker = Longformer_Reranker()

import ast
def evaluate_query_pairs(query_pairs_set, name):
    accuracy_list = []
    retriever_correct_list = []
    retrieved_docs_list = []
    reranker_list = []
    for query, target_api in query_pairs_set:
        retrieved_docs = retriever.get_relevant_documents(query)
        api_name_list = [ast.literal_eval(retrieved_doc.page_content).get('api_name') for retrieved_doc in retrieved_docs]
        retrieved_docs_list.append((target_api, api_name_list))
        if target_api in api_name_list:
            retriever_correct_list.append(target_api)
        else:
            #print('-' * 10)
            #print(f'This api is not retrieved successfully in {name} set!')
            #print('target_api:', target_api, 'retrieved_list:', api_name_list, 'query:', query)
            continue
        if using_reranker == 'longerformer':
            ori_ans = [ast.literal_eval(retrieved_doc.page_content) for retrieved_doc in retrieved_docs]
            answer_api = reranker.rerank(query, ori_ans)
            if answer_api == target_api:
                reranker_list.append(answer_api)
        elif using_reranker == 'gpt':
            prompt = """
Now I give you the candidate APIs and their descriptions, api_list.
candidate APIs starts:
"""
            for retrieved_doc in retrieved_docs:
                prompt += retrieved_doc.page_content + '\n'
            prompt += """
candidate APIs end.
You need to refer to their function descriptions and parameter descriptions, 
find the most relevant one based on the user's query, and return it to me. 
Restricted to Response format: 
{'answer_api':[top-1 api_name]}
query: 
"""
            prompt += query
            prompt += 'Now your answer starts:'
            result, _ = LLM_response(llm, tokenizer, prompt, history=[], kwargs={})
            val_result = ast.literal_eval(result)
            is_present = target_api in val_result['answer_api']
            accuracy_list.append(is_present)
        else:
            continue
    
    retrieve_rate = len(retriever_correct_list) / len(query_pairs_set)
    accuracy_rate = sum(accuracy_list) / len(query_pairs_set)
    reranker_rate = len(reranker_list) / len(query_pairs_set)
    print(f"Totally tested {len(query_pairs_set)} {name} queries! {name} retriever top-{args.top_k} accuracy rate: {retrieve_rate * 100:.2f}%")
    if accuracy_rate>0:
        print(f"{name} LLM Answer top-{args.top_k} accuracy rate: {accuracy_rate * 100:.2f}%")
    if reranker_rate>0:
        print(f"{name} reranker top-{args.top_k} accuracy rate: {reranker_rate * 100:.2f}%")

# Evaluate for train and test sets
evaluate_query_pairs(train_query_pairs, "train")
evaluate_query_pairs(test_query_pairs, "test")
evaluate_query_pairs(val_query_pairs, "val")
