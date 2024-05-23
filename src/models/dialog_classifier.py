"""
Author: Zhengyuan Dong
Date Created: May 06, 2024
Last Modified: May 21, 2024
Description: Dialog classifier based on Gaussian distribution
"""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import os
from sentence_transformers import util
from tqdm import tqdm

from ..configs.model_config import get_all_variable_from_cheatsheet

from ..gpt.utils import load_json, save_json

class Dialog_Gaussian_classification:
    def __init__(self, LIB='scanpy', threshold=0.05):
        self.threshold = threshold
        self.LIB = LIB
        info_json = get_all_variable_from_cheatsheet(self.LIB)
        self.LIB_ALIAS = info_json['LIB_ALIAS']
        
    def fit_gaussian(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.save_mean_std()
        return self.mean, self.std

    def calculate_p_values(self, scores, mean, std):
        return [norm.cdf(score, mean, std) for score in scores]

    def classify_based_on_p(self, p_values, threshold=0.05):
        return [1 if p < threshold else 0 for p in p_values]
    
    def classify(self, rank_1_scores):
        p_values_val = self.calculate_p_values(rank_1_scores, self.mean, self.std)
        predictions_val = self.classify_based_on_p(p_values_val, threshold=self.threshold)
        return predictions_val

    def compute_acc(self, labels, predictions):
        return accuracy_score(labels, predictions)
        
    def plot_boxplot(self, data, title):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data)
        plt.title(title)
        plt.xticks(ticks=range(5), labels=[f'Rank {i+1}' for i in range(5)])
        plt.ylabel('Score')
        plt.savefig(f'./plot/{self.LIB}/avg_retriever_{title}.pdf')
    
    def compute_accuracy_filter_compositeAPI(self, retriever, data, retrieved_api_nums, name='train', verbose=False, filter_composite=True):
        # remove class type API, and composite API from the data
        API_composite = load_json(os.path.join(f"data/standard_process/{self.LIB}","API_composite.json"))
        data_to_save = []
        scores_rank_1 = []
        scores_rank_2 = []
        scores_rank_3 = []
        scores_rank_4 = []
        scores_rank_5 = []
        outliers = []
        total_api_non_composite = 0
        total_api_non_ambiguous = 0
        query_to_api = {}
        query_to_retrieved_api = {}
        query_to_all_scores = {}
        for query_data in tqdm(data):
            retrieved_apis = retriever.retrieving(query_data['query'], top_k=retrieved_api_nums+20)
            if filter_composite:
                retrieved_apis = [i for i in retrieved_apis if i.startswith(self.LIB_ALIAS) and API_composite[i]['api_type']!='class' and API_composite[i]['api_type']!='unknown']
            retrieved_apis = retrieved_apis[:retrieved_api_nums]
            assert len(retrieved_apis)==retrieved_api_nums
            query_to_retrieved_api[query_data['query']] = retrieved_apis
            try:
                query_to_api[query_data['query']] = query_data['api_calling'][0].split('(')[0]
            except:
                pass
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
            scores = [hit['score'] for hit in hits[0]] if hits[0] else []
            query_to_all_scores[query_data['query']] = scores
        # Compute average scores for each rank
        scores = {
            "rank_1": scores_rank_1,
            "rank_2": scores_rank_2,
            "rank_3": scores_rank_3,
            "rank_4": scores_rank_4,
            "rank_5": scores_rank_5
        }
        q1, q3 = np.percentile(scores_rank_1, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        for i, score in enumerate(scores_rank_1):
            if score < lower_bound or score > upper_bound:
                try:
                    outliers.append({'index': i, 'score': score, 'query': data[i]['query'], 'retrieved_apis': query_to_retrieved_api[data[i]['query']], 'query_api': query_to_api[data[i]['query']], 'all_scores': query_to_all_scores[data[i]['query']]})
                    if verbose:
                        print(f"{name} Outlier detected: Score = {score}, Query = {data[i]['query']}, retrieved_apis = {query_to_retrieved_api[data[i]['query']]}, query_api = {query_to_api[data[i]['query']]}, score = {query_to_all_scores[data[i]['query']]}")
                except:
                    pass
        return scores, outliers
    def single_prediction(self, query, retriever, top_k):
        #self.load_mean_std()
        query_embedding = retriever.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, retriever.corpus_embeddings, top_k=top_k, score_function=util.cos_sim)
        if len(hits[0]) > 0:
            score_rank1 = hits[0][0]['score']
        # TODO: need to load the threshold for the score_rank1 to distinguish whether it is a dialog
        pred_label = self.classify([score_rank1])
        if pred_label==1:
            pred_class = 'multiple'
        else:
            pred_class = 'single'
        return pred_class
    def save_mean_std(self,):
        data = {
            "mean": self.mean,
            "std": self.std,
        }
        directory = os.path.join(f"data/standard_process/{self.LIB}")
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, "dialog_classifier_data.json")
        save_json(data, file_path)
        print(f"Data saved to {file_path}")
    def load_mean_std(self, ):
        file_path = os.path.join(f"data/standard_process/{self.LIB}", "dialog_classifier_data.json")
        data = load_json(file_path)
        self.mean = data["mean"]
        self.std = data["std"]
        print(f"Data loaded from {file_path}")

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

