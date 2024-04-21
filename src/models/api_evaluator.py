from sklearn.metrics import ndcg_score
import numpy as np
import logging
import os
from typing import List, Dict, Set
from tqdm import trange
from tqdm import tqdm
from multiprocessing import Pool
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import cos_sim
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_file = "log_file.txt"
if os.path.exists(log_file):
    os.remove(log_file)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def compute_ndcg_for_query(query_tuple):
    query_itr, query_id, top_hits, relevant_docs, corpus_ids, k = query_tuple
    query_relevant_docs = relevant_docs[query_id]
    # Build the ground truth relevance scores and the model's predicted scores
    true_relevance = np.zeros(len(corpus_ids))
    predicted_scores = np.zeros(len(corpus_ids))
    for hit in top_hits[:k]: # Limit to top k results
        predicted_scores[corpus_ids.index(hit['corpus_id'])] = hit['score']
        if hit['corpus_id'] in query_relevant_docs:
            true_relevance[corpus_ids.index(hit['corpus_id'])] = 1
    return ndcg_score([true_relevance], [predicted_scores])

class APIEvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.
    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document.
    """
    def __init__(self,
             corpus_config: Dict[str, Dict[str, str]],
             corpus: Dict[str, str],  # cid => doc
             corpus_chunk_size: int = 5,
             show_progress_bar: bool = True,
             batch_size: int = 1,
             score_function=cos_sim,  # Score function, higher=more similar
             fig_path: str='./plot/retriever',
             optimize_top_k: int=3,
             ):
        self.train_queries_id = list(corpus_config['train']['queries'].keys())
        self.train_queries = [corpus_config['train']['queries'][qid] for qid in self.train_queries_id]
        self.train_relevant_docs = corpus_config['train']['relevant_docs']
        self.val_queries_id = list(corpus_config['val']['queries'].keys())
        self.val_queries = [corpus_config['val']['queries'][qid] for qid in self.val_queries_id]
        self.val_relevant_docs = corpus_config['val']['relevant_docs']
        self.test_queries_id = list(corpus_config['test']['queries'].keys())
        self.test_queries = [corpus_config['test']['queries'][qid] for qid in self.test_queries_id]
        self.test_relevant_docs = corpus_config['test']['relevant_docs']

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.score_function = score_function
        self.fig_path = fig_path
        self.csv_data = []
        self.optimize_top_k=optimize_top_k
    
    def plot_ndcg(self, csv_data, name_list, annotate=False):
        """
        Plot and save the NDCG scores.

        :param csv_data: List of tuples containing the data (epoch, avg_ndcg, name)
        :param name_list: List of dataset names (train/val/test)
        """
        colors = ['r', 'g', 'b']  # Colors for train, val, and test
        metrics = 3  # As per your description, there are three metrics
        plt.figure(figsize=(15, 15))
        for metric_idx in range(metrics):
            plt.subplot(3, 1, metric_idx + 1)
            for idx, name in enumerate(name_list):
                epoch_values = [i[0] for i in csv_data]
                ndcg_values = [i[2 + idx][metric_idx] for i in csv_data]
                plt.plot(epoch_values, ndcg_values, label=f'Average NDCG@{metric_idx*2 + 1} - {name}', color=colors[idx])
                # Add NDCG scores as text annotations
                if annotate:
                    for i in range(len(epoch_values)):
                        plt.annotate(f'{ndcg_values[i]:.2f}', (epoch_values[i], ndcg_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
            plt.xlabel('Epoch')
            plt.ylabel(f'NDCG@{metric_idx*2 + 1}')
            plt.legend()
            plt.title(f'NDCG@{metric_idx*2 + 1} scores')
            #plt.grid(True)
            plt.tight_layout()
        plt.savefig(os.path.join(self.fig_path, "ndcg_plot.pdf"))
        plt.close()

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"
        logger.info("Information Retrieval Evaluation" + out_txt)
        avg_ndcg_train = self.compute_metrices(model, self.train_queries, self.corpus, self.corpus_ids, self.train_relevant_docs, self.train_queries_id)
        avg_ndcg_val = self.compute_metrices(model, self.val_queries, self.corpus, self.corpus_ids, self.val_relevant_docs, self.val_queries_id)
        avg_ndcg_test = self.compute_metrices(model, self.test_queries, self.corpus, self.corpus_ids, self.test_relevant_docs, self.test_queries_id)
        
        self.csv_data.append([epoch, steps, avg_ndcg_train, avg_ndcg_val, avg_ndcg_test])
        self.plot_ndcg(self.csv_data, ['train', 'val']) # , 'test'
        jsond={'1':0,'3':1,'5':2}
        index = jsond[str(self.optimize_top_k)]
        #return avg_ndcg_val[index]
        return min(avg_ndcg_val)
    
    def compute_metrices(self, model, queries, corpus, corpus_ids, relevant_docs, queries_id) -> Dict[int, float]:
        # Compute embedding for the queries
        query_embeddings = model.encode(queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)
        queries_result_list = [[] for _ in range(len(query_embeddings))]
        # Iterate over chunks of the corpus
        for corpus_start_idx in trange(0, len(corpus), self.corpus_chunk_size, desc='Corpus Chunks', disable=not self.show_progress_bar):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))
            # Encode chunk of corpus
            sub_corpus_embeddings = model.encode(corpus[corpus_start_idx:corpus_end_idx], show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True)
            # Compute cosine similarites
            pair_scores = self.score_function(query_embeddings, sub_corpus_embeddings)
            # Convert scores to list
            pair_scores_list = pair_scores.cpu().tolist()
            for query_itr in range(len(query_embeddings)):
                for sub_corpus_id, score in enumerate(pair_scores_list[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                    queries_result_list[query_itr].append({'corpus_id': corpus_id, 'score': score})
        for query_itr in range(len(queries_result_list)):
            for doc_itr in range(len(queries_result_list[query_itr])):
                score, corpus_id = queries_result_list[query_itr][doc_itr]['score'], queries_result_list[query_itr][doc_itr]['corpus_id']
                queries_result_list[query_itr][doc_itr] = {'corpus_id': corpus_id, 'score': score}
        logger.info("Queries: {}".format(len(queries)))
        logger.info("Corpus: {}\n".format(len(corpus)))
        # Compute scores
        scores = self.compute_metrics(queries_result_list, relevant_docs, queries_id, corpus_ids)
        # Output
        logger.info("Average NDCG@1: {:.2f}".format(scores[0] * 100))
        logger.info("Average NDCG@3: {:.2f}".format(scores[1] * 100))
        logger.info("Average NDCG@5: {:.2f}".format(scores[2] * 100))
        return scores

    def compute_metrics(self, queries_result_list, relevant_docs, queries_id, corpus_ids):
        # Init score computation values
        ndcg_scores = []
        # Compute scores on results using a pool of workers
        k_list = [1, 3, 5]
        scores = []
        for k in k_list:
            # Build a list of tuples, each containing the data needed for one query
            query_tuples = []
            for query_itr in range(len(queries_result_list)):
                query_id = queries_id[query_itr]
                top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
                query_tuples.append((query_itr, query_id, top_hits, relevant_docs, corpus_ids, k))  # add 'k' to each tuple
            ndcg_scores.clear()  # clear the list for each 'k'
            with Pool() as p:
                max_ = len(query_tuples)
                with tqdm(total=max_) as pbar:
                    for i, _ in tqdm(enumerate(p.imap(compute_ndcg_for_query, query_tuples))):
                        pbar.update()
                        ndcg_scores.append(_)
            scores.append(np.mean(ndcg_scores))
        # Return the average NDCG@k of all queries for each 'k'
        return scores


import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))
