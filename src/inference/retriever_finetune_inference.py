import argparse, os, random
from tqdm import tqdm
import pandas as pd
from ..configs.model_config import get_all_variable_from_cheatsheet
from sentence_transformers import SentenceTransformer, util
from ..inference.utils import process_retrieval_document_query_version, is_pair_in_merged_pairs, find_similar_two_pairs
from ..gpt.utils import save_json, load_json
import torch
# Print average scores for each rank
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple

class ToolRetriever:
    def __init__(self, LIB, corpus_tsv_path = "", model_path="", base_corpus_tsv_path="./data/standard_process/base/retriever_train_data/corpus.tsv",add_base=False, shuffle_data=True, process_func=process_retrieval_document_query_version,max_seq_length=256):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.process_func=process_func
        self.max_seq_length = max_seq_length
        self.model_path = model_path
        self.corpus_tsv_path = corpus_tsv_path
        self.base_corpus_tsv_path = base_corpus_tsv_path
        self.build_and_merge_corpus(add_base=add_base)
        if shuffle_data:
            self.shuffled_data = self.build_shuffle_data(LIB, add_base=add_base)
            self.shuffled_queries = [item['query'] for item in self.shuffled_data]
            self.shuffled_query_embeddings = self.embedder.encode(self.shuffled_queries, convert_to_tensor=True)
    
    def build_shuffle_data(self,LIB, add_base=True):
        print('set add_base as :', add_base)
        # add API_base, fix 231227
        def process_data(path, files_ids):
            data = load_json(f'{path}/API_inquiry_annotate.json')
            return [dict(query=row['query'], gold=row['api_name']) for row in data if row['query_id'] not in files_ids['val'] and row['query_id'] not in files_ids['test']]
        lib_files_ids = load_json(f'./data/standard_process/{LIB}/API_instruction_testval_query_ids.json')
        lib_data = process_data(f'./data/standard_process/{LIB}', lib_files_ids)
        base_files_ids = load_json(f'./data/standard_process/base/API_instruction_testval_query_ids.json')
        base_data = process_data('./data/standard_process/base', base_files_ids)
        if add_base:
            lib_data = lib_data + base_data
        random.Random(0).shuffle(lib_data)
        return lib_data
    def build_and_merge_corpus(self, add_base=True):
        print('set add_base as :', add_base)
        # based on build_retrieval_corpus, add API_base.json, fix 231227
        original_corpus_df = pd.read_csv(self.corpus_tsv_path, sep='\t')
        if add_base:
            print('--------> add base!')
            additional_corpus_df = pd.read_csv(self.base_corpus_tsv_path, sep='\t')
            combined_corpus_df = pd.concat([original_corpus_df, additional_corpus_df], ignore_index=True)
            combined_corpus_df.reset_index(drop=True, inplace=True)
        else:
            print('--------> not add base!')
            combined_corpus_df = original_corpus_df
        corpus, self.corpus2tool = self.process_func(combined_corpus_df)
        corpus_ids = list(corpus.keys())
        corpus = [corpus[cid] for cid in corpus_ids]
        self.corpus = corpus
        if 'hugging_models' in self.model_path:
            print('running on pretrained model!!!')
            self.embedder = SentenceTransformer(self.model_path, device=self.device)
        elif self.model_path=='all-MiniLM-L6-v2' or self.model_path=='bert-base-uncased':
            print('running on unpretrained model!!!')
            self.embedder = SentenceTransformer(self.model_path, device=self.device)
        else:
            raise ValueError
        self.corpus_embeddings = self.embedder.encode(self.corpus, convert_to_tensor=True)
        print('the length of corpus is: ', len(self.corpus_embeddings))
    def retrieving(self, query, top_k):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k, score_function=util.cos_sim)
        retrieved_apis = [self.corpus2tool[hit['corpus_id']] for hit in hits[0]]
        return retrieved_apis[:top_k]
    def retrieve_similar_queries(self, query, shot_k=5):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.shuffled_query_embeddings, top_k=shot_k, score_function=util.cos_sim)
        #similar_queries = [shuffled_data[hit['corpus_id']] for hit in hits[0]]
        similar_queries = ["\nInstruction: " + self.shuffled_data[hit['corpus_id']]['query'] + "\nFunction: " + self.shuffled_data[hit['corpus_id']]['gold'] for hit in hits[0]]
        return ''.join(similar_queries)

def compute_accuracy(retriever: ToolRetriever, data: List[Dict[str, Any]], args: argparse.Namespace, name: str = 'train', LIB_ALIAS: str = 'scanpy') -> Tuple[float, Dict[str, List[float]], float, int]:
    """
    Computes the accuracy of the retrieval based on the given data and the retriever's responses.

    Parameters
    ----------
    retriever : ToolRetriever
        The retrieval tool to use for finding similar APIs.
    data : List[Dict[str, Any]]
        The dataset containing queries and their corresponding true APIs.
    args : argparse.Namespace
        The command line arguments passed to the script.
    name : str, optional
        The name of the dataset (e.g., 'train', 'test', or 'val'). Default is 'train'.
    LIB_ALIAS : str, optional
        The library alias to filter APIs by, default is 'scanpy'.

    Returns
    -------
    Tuple[float, Dict[str, List[float]], float, int]
        A tuple containing:
        - Accuracy as a percentage of correct predictions.
        - Dictionary with lists of scores for ranks 1 through 5.
        - Ambiguous accuracy as a percentage of correct predictions considering ambiguous matches.
        - The count of total non-ambiguous API matches.
    """
    merged_pairs = find_similar_two_pairs(f"./data/standard_process/{args.LIB}/API_init.json")
    correct_predictions = 0
    ambiguous_correct_predictions = 0  # Additional metric for ambiguous matches
    error_predictions = 0
    total_api_non_ambiguous = 0
    data_to_save = []
    scores_rank_1 = []
    scores_rank_2 = []
    scores_rank_3 = []
    scores_rank_4 = []
    scores_rank_5 = []
    for query_data in tqdm(data):
        retrieved_apis = retriever.retrieving(query_data['query'], top_k=args.retrieved_api_nums)
        true_api = query_data['api_name']
        # changed the acc count
        success = False
        ambiguous = False
        for pred_api in retrieved_apis:
            if true_api == pred_api:
                correct_predictions += 1
                success = True
                break
            elif is_pair_in_merged_pairs(true_api, pred_api, merged_pairs):
                ambiguous_correct_predictions += 1
                success = True
                ambiguous = True
                break
        if not ambiguous:
            total_api_non_ambiguous += 1
        if not success:
            error_predictions += 1
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
    assert total_api_non_ambiguous<len(data)
    accuracy = correct_predictions / len(data) * 100
    ambiguous_accuracy = (correct_predictions) / total_api_non_ambiguous * 100
    save_json(f'./plot/{args.LIB}/error_{name}_topk_{args.retrieved_api_nums}.json', data_to_save)
    # Compute average scores for each rank
    scores = {
        "rank_1": scores_rank_1,
        "rank_2": scores_rank_2,
        "rank_3": scores_rank_3,
        "rank_4": scores_rank_4,
        "rank_5": scores_rank_5
    }
    return accuracy, scores, ambiguous_accuracy, total_api_non_ambiguous

def compute_accuracy_filter_compositeAPI(retriever, data, args,name='train', LIB_ALIAS='scanpy'):
    # remove class type API, and composite API from the data
    API_composite = load_json(f"./data/standard_process/{args.LIB}/API_composite.json")
    merged_pairs = find_similar_two_pairs(f"./data/standard_process/{args.LIB}/API_init.json")
    correct_predictions = 0
    ambiguous_correct_predictions = 0  # Additional metric for ambiguous matches
    error_predictions = 0
    data_to_save = []
    scores_rank_1 = []
    scores_rank_2 = []
    scores_rank_3 = []
    scores_rank_4 = []
    scores_rank_5 = []
    #filtered_data = [item for item in data if not is_pair_in_merged_pairs(item['api_name'], item['pred'], merged_pairs)]
    #total_non_ambiguous_pairs = len(filtered_data)
    total_api_non_composite = 0
    total_api_non_ambiguous = 0
    for query_data in tqdm(data):
        retrieved_apis = retriever.retrieving(query_data['query'], top_k=args.retrieved_api_nums+20)
        true_api = query_data['api_name']
        if not true_api.startswith(LIB_ALIAS) or query_data['api_type']=='class' or query_data['api_type']=='unknown':
            # remove composite API, class API
            continue
        else:
            total_api_non_composite+=1
        retrieved_apis = [i for i in retrieved_apis if i.startswith(LIB_ALIAS) and API_composite[i]['api_type']!='class' and API_composite[i]['api_type']!='unknown']
        retrieved_apis = retrieved_apis[:args.retrieved_api_nums]
        assert len(retrieved_apis)==args.retrieved_api_nums
        # changed the acc count
        success = False
        ambiguous = False
        for pred_api in retrieved_apis:
            if true_api == pred_api:
                correct_predictions += 1
                success = True
                break
            elif is_pair_in_merged_pairs(true_api, pred_api, merged_pairs):
                #total_api_non_ambiguous+=1
                ambiguous_correct_predictions += 1
                success = True
                ambiguous = True
                break
        if not ambiguous:
            total_api_non_ambiguous += 1
        if not success:
            error_predictions += 1
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
    assert error_predictions + correct_predictions + ambiguous_correct_predictions == total_api_non_composite
    assert ambiguous_correct_predictions + total_api_non_ambiguous == total_api_non_composite
    accuracy = correct_predictions / total_api_non_composite * 100
    ambiguous_accuracy = (correct_predictions) / total_api_non_ambiguous * 100
    save_json(f'./plot/{args.LIB}/error_{name}_topk_{args.retrieved_api_nums}.json', data_to_save)
    # Compute average scores for each rank
    scores = {
        "rank_1": scores_rank_1,
        "rank_2": scores_rank_2,
        "rank_3": scores_rank_3,
        "rank_4": scores_rank_4,
        "rank_5": scores_rank_5
    }
    return accuracy, scores, ambiguous_accuracy, total_api_non_ambiguous

def compute_and_plot(data_set, set_name, retriever, args, compute_func, LIB_ALIAS):
    """Compute scores, visualize"""
    accuracy, avg_scores, ambiguous_accuracy, total_api_non_ambiguous = compute_func(retriever, data_set, args, set_name, LIB_ALIAS)
    print(f"{set_name.capitalize()} Accuracy: {accuracy:.2f}%, #samples {len(data_set)}")
    print(f"{set_name.capitalize()} ambiguous Accuracy: {ambiguous_accuracy:.2f}%, #samples {total_api_non_ambiguous}")
    scores = [avg_scores[f'rank_{i+1}'] for i in range(5)]
    plot_boxplot(scores, set_name.capitalize())

def plot_boxplot(data, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.title(title)
    plt.xticks(ticks=range(5), labels=[f'Rank {i+1}' for i in range(5)])
    plt.ylabel('Score')
    plt.savefig(f'./plot/{args.LIB}/avg_retriever_{title}.pdf')

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_tsv_path', type=str, required=True, help='')
    parser.add_argument('--retrieval_model_path', type=str, required=True, help='')
    parser.add_argument('--retrieved_api_nums', type=int, required=True, help='')
    parser.add_argument('--input_query_file', type=str, required=True, help='input path')
    parser.add_argument('--idx_file', type=str, required=True, help='idx path')
    parser.add_argument('--LIB', type=str, required=True, help='lib')
    parser.add_argument("--max_seq_length", default=256, type=int, required=True,help="Max sequence length.")
    parser.add_argument('--filter_composite', action='store_true', help='Use compute_accuracy_filter_compositeAPI instead of compute_accuracy')
    args = parser.parse_args()
    info_json = get_all_variable_from_cheatsheet(args.LIB)
    LIB_ALIAS, API_HTML, TUTORIAL_GITHUB, API_HTML_PATH = [info_json[key] for key in ['LIB_ALIAS', 'API_HTML', 'TUTORIAL_GITHUB','API_HTML_PATH']]

    # Step 1: Load API data from the JSON file
    api_data = load_json(args.input_query_file)
    index_data = load_json(args.idx_file)
    test_ids = index_data['test']
    val_ids = index_data['val']

    # Step 2: Create a ToolRetriever instance
    retriever = ToolRetriever(LIB = args.LIB, corpus_tsv_path=args.corpus_tsv_path, model_path=args.retrieval_model_path, add_base=False,max_seq_length=args.max_seq_length)

    total_queries = 0
    correct_predictions = 0
    # Step 3: Process each query and retrieve relevant APIs
    train_data = [data for data in api_data if data['query_id'] not in test_ids and data['query_id'] not in val_ids]
    val_data = [data for data in api_data if data['query_id'] in val_ids]
    test_data = [data for data in api_data if data['query_id'] in test_ids]
    print(len(train_data), len(val_data), len(test_data))

    os.makedirs("./plot",exist_ok=True)
    os.makedirs(f"./plot/{args.LIB}",exist_ok=True)
    compute_func = compute_accuracy_filter_compositeAPI if args.filter_composite else compute_accuracy

    for set_name, data_set in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        compute_and_plot(data_set, set_name, retriever, args, compute_func, LIB_ALIAS)
