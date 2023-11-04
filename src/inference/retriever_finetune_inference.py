import argparse
import os
import json
from xml.etree.ElementTree import QName
from tqdm import tqdm
import time
import pandas as pd
import re
from configs.model_config import HUGGINGPATH
from sentence_transformers import SentenceTransformer, util
from inference.utils import process_retrieval_document_query_version, compress_api_str_from_list_query_version
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print average scores for each rank
import matplotlib.pyplot as plt
import seaborn as sns


class ToolRetriever:
    def __init__(self, LIB, corpus_tsv_path = "", model_path=""):
        #self.model_path = os.path.join(model_path,f"{LIB}","assigned")
        self.model_path = model_path
        self.build_retrieval_corpus(corpus_tsv_path)
        self.shuffled_data = self.build_shuffle_data(LIB)
        self.shuffled_queries = [item['query'] for item in self.shuffled_data]
        self.shuffled_query_embeddings = self.embedder.encode(self.shuffled_queries, convert_to_tensor=True)
    def build_shuffle_data(self,LIB):
        import random
        with open(f'./data/standard_process/{LIB}/API_inquiry_annotate.json', 'r') as f:
            data = json.load(f)
        with open(f"./data/standard_process/{LIB}/API_instruction_testval_query_ids.json", 'r') as file:
            files_ids = json.load(file)
        shuffled = [dict(query=row['query'], gold=row['api_name']) for row in [i for i in data if i['query_id'] not in files_ids['val'] and i['query_id'] not in files_ids['test']]]
        random.Random(0).shuffle(shuffled)
        return shuffled
    def build_retrieval_corpus(self, corpus_tsv_path):
        self.corpus_tsv_path = corpus_tsv_path
        documents_df = pd.read_csv(self.corpus_tsv_path, sep='\t')
        corpus, self.corpus2tool = process_retrieval_document_query_version(documents_df)
        corpus_ids = list(corpus.keys())
        corpus = [corpus[cid] for cid in corpus_ids]
        self.corpus = corpus
        self.embedder = SentenceTransformer(self.model_path, device=device)
        self.corpus_embeddings = self.embedder.encode(self.corpus, convert_to_tensor=True)
    def retrieving(self, query, top_k):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k, score_function=util.cos_sim) #170*
        retrieved_apis = [self.corpus2tool[self.corpus[hit['corpus_id']]] for hit in hits[0]]
        return retrieved_apis[:top_k]
    def retrieve_similar_queries(self, query, shot_k=5):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.shuffled_query_embeddings, top_k=shot_k, score_function=util.cos_sim)
        #similar_queries = [shuffled_data[hit['corpus_id']] for hit in hits[0]]
        similar_queries = ["\nInstruction: " + self.shuffled_data[hit['corpus_id']]['query'] + "\nFunction: " + self.shuffled_data[hit['corpus_id']]['gold'] for hit in hits[0]]
        return ''.join(similar_queries)

def compute_accuracy(retriever, data, args, name='train'):
    correct_predictions = 0
    data_to_save = []
    scores_rank_1 = []
    scores_rank_2 = []
    scores_rank_3 = []
    scores_rank_4 = []
    scores_rank_5 = []
    for query_data in tqdm(data):
        retrieved_apis = retriever.retrieving(query_data['query'], top_k=args.retrieved_api_nums)
        true_api = query_data['api_name']
        if true_api in retrieved_apis:  # Checking for intersection between the two sets
            correct_predictions += 1
        else:
            data_to_save.append({
                "query": query_data['query'],
                "ground_truth": [true_api],
                "retrieved_apis": retrieved_apis
            })
        query_embedding = retriever.embedder.encode(query_data['query'], convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, retriever.corpus_embeddings, top_k=5, score_function=util.cos_sim)
        if len(hits[0]) > 0:
            scores_rank_1.append(hits[0][0]['score'])
        if len(hits[0]) > 1:
            scores_rank_2.append(hits[0][1]['score'])
        if len(hits[0]) > 2:
            scores_rank_3.append(hits[0][2]['score'])
        if len(hits[0]) > 3:
            scores_rank_4.append(hits[0][3]['score'])
        if len(hits[0]) > 4:
            scores_rank_5.append(hits[0][4]['score'])
    accuracy = correct_predictions / len(data) * 100
    with open(f'./plot/error_{name}.json', 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)
    # Compute average scores for each rank
    scores = {
        "rank_1": scores_rank_1,
        "rank_2": scores_rank_2,
        "rank_3": scores_rank_3,
        "rank_4": scores_rank_4,
        "rank_5": scores_rank_5
    }
    return accuracy, scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_tsv_path', type=str, required=True, help='')
    parser.add_argument('--retrieval_model_path', type=str, required=True, help='')
    parser.add_argument('--retrieved_api_nums', type=int, required=True, help='')
    parser.add_argument('--input_query_file', type=str, required=True, help='input path')
    parser.add_argument('--idx_file', type=str, required=True, help='idx path')
    parser.add_argument('--LIB', type=str, required=True, help='lib')
    args = parser.parse_args()

    # Step 1: Load API data from the JSON file
    with open(args.input_query_file, 'r') as file:
        api_data = json.load(file)
    with open(args.idx_file, 'r') as f:
        index_data = json.load(f)
    test_ids = index_data['test']
    val_ids = index_data['val']

    # Step 2: Create a ToolRetriever instance
    retriever = ToolRetriever(LIB = args.LIB, corpus_tsv_path=args.corpus_tsv_path, model_path=args.retrieval_model_path)
    print(retriever.corpus[0])

    total_queries = 0
    correct_predictions = 0
    # Step 3: Process each query and retrieve relevant APIs
    train_data = [data for data in api_data if data['query_id'] not in test_ids and data['query_id'] not in val_ids]
    val_data = [data for data in api_data if data['query_id'] in val_ids]
    test_data = [data for data in api_data if data['query_id'] in test_ids]
    print(len(train_data), len(val_data), len(test_data))

    os.makedirs("./plot",exist_ok=True)
    train_accuracy, train_avg_scores = compute_accuracy(retriever, train_data, args, 'train')
    val_accuracy, val_avg_scores = compute_accuracy(retriever, val_data, args, 'val')
    test_accuracy, test_avg_scores = compute_accuracy(retriever, test_data, args, 'test')
    print(f"Training Accuracy: {train_accuracy:.2f}%, #samples {len(train_data)}")
    print(f"val Accuracy: {val_accuracy:.2f}%, #samples {len(val_data)}")
    print(f"test Accuracy: {test_accuracy:.2f}%, #samples {len(test_data)}")

    def plot_boxplot(data, title):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data)
        plt.title(title)
        plt.xticks(ticks=range(5), labels=[f'Rank {i+1}' for i in range(5)])
        plt.ylabel('Score')
        plt.savefig(f'./plot/avg_retriever_{title}.pdf')

    train_scores = [train_avg_scores['rank_1'], train_avg_scores['rank_2'], train_avg_scores['rank_3'], train_avg_scores['rank_4'], train_avg_scores['rank_5']]
    val_scores = [val_avg_scores['rank_1'], val_avg_scores['rank_2'], val_avg_scores['rank_3'], val_avg_scores['rank_4'], val_avg_scores['rank_5']]
    test_scores = [test_avg_scores['rank_1'], test_avg_scores['rank_2'], test_avg_scores['rank_3'], test_avg_scores['rank_4'], test_avg_scores['rank_5']]

    plot_boxplot(train_scores, "Training")
    plot_boxplot(val_scores, "Validation")
    plot_boxplot(test_scores, "Test")
    