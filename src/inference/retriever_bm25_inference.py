"""
Author: Zhengyuan Dong
Email: zydong122@gmail.com
Description: This script contains functions to prepare data, create a retriever, and evaluate its performance.
"""
import ast, os
from rank_bm25 import BM25Okapi
from ..retrievers import *
from ..models.model import *
from ..inference.utils import find_similar_two_pairs
from typing import Tuple, List, Dict, Any, Set
from ..gpt.utils import load_json
# prepare corpus
def prepare_corpus(ori_data: dict) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
    """
    Prepares a corpus from the original data.

    Parameters
    ----------
    ori_data : dict
        Original data mapping API names to their information.

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[List[str]]]
        A tuple containing:
        - A list of dictionaries, each representing an API with its description and parameters.
        - A list of tokenized representations of each API for BM25 processing.
    """
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

def get_query_pairs(results: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Extracts query pairs from the result data.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        The list of result entries containing query and API name.

    Returns
    -------
    List[Tuple[str, str]]
        A list of tuples where each tuple contains a query and the corresponding API name.
    """
    return [(entry['query'], entry['api_name']) for entry in results]

def evaluate_query_pairs(retriever: BM25Retriever, query_pairs_set: List[Tuple[str, str]], name: str, top_k: int, merged_pairs: Set[Tuple[str, str]]) -> None:
    """
    Evaluates the accuracy of the retriever on a set of query pairs.

    Parameters
    ----------
    retriever : BM25Retriever
        The retriever to use for getting relevant documents.
    query_pairs_set : List[Tuple[str, str]]
        A set of query pairs to evaluate.
    name : str
        A name for the dataset being evaluated (e.g., "train", "test").
    top_k : int
        The number of top documents to consider for accuracy calculation.
    merged_pairs : Set[Tuple[str, str]]
        A set of merged pairs for checking ambiguous correctness.
    """
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
    print(f"{name} retriever top-{top_k} accuracy rate: {retrieve_rate * 100:.2f}%")
    print(f"{name} retriever top-{top_k} ambiguous accuracy rate: {ambiguous_retrieve_rate * 100:.2f}%")

def main(lib_data_path, top_k: int) -> None:
    """
    Main function to prepare data, create a retriever, and evaluate its performance.

    Parameters
    ----------
    lib_data_path: str
        The path to the data directory.
    top_k : int
        The number of top documents to retrieve.
    """
    ori_data = load_json(os.path.join(lib_data_path, "API_composite.json"))
    results = load_json(os.path.join(lib_data_path, "API_inquiry_annotate.json"))
    index_file = load_json(os.path.join(lib_data_path, "API_instruction_testval_query_ids.json"))
    test_ids = index_file['test']
    val_ids = index_file['val']
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
    retriever = BM25Retriever(index=bm25, corpus=corpus, query_kwargs={"similarity_top_k": top_k})

    merged_pairs = find_similar_two_pairs(os.path.join(lib_data_path, "API_init.json"))

    # Evaluate for train and test sets
    evaluate_query_pairs(retriever, train_query_pairs, "train", top_k, merged_pairs)
    evaluate_query_pairs(retriever, val_query_pairs, "val", top_k, merged_pairs)
    evaluate_query_pairs(retriever, test_query_pairs, "test", top_k, merged_pairs)

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=3, help='')
    parser.add_argument('--LIB', type=str, help='')
    args = parser.parse_args()
    
    from ..configs.model_config import get_all_variable_from_cheatsheet
    info_json = get_all_variable_from_cheatsheet(args.LIB)
    LIB_DATA_PATH = info_json['LIB_DATA_PATH']
    
    main(LIB_DATA_PATH, args.top_k)
    