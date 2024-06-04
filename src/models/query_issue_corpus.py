"""
Author: Zhengyuan Dong
Date Created: May 29, 2024
Last Modified: May 29, 2024
Description: Query the issue corpus for the specified library
Usage: 
python -m src.models.query_issue_corpus --LIB scanpy --example_query "Traceback (most recent call last): \n File "/home/z6dong/BioChat/refer/src/2024_biomania_phase2/./examples/case2.1/output/3.sh.execute.py", line 13, in <module>\n     sc.tl.louvain(adata)\nFile "/home/z6dong/anaconda3/envs/biomania2/lib/python3.10/site-packages/scanpy/tools/_louvain.py", line 115, in louvain\n    adjacency = _choose_graph(adata, obsp, neighbors_key)\n  File "/home/z6dong/anaconda3/envs/biomania2/lib/python3.10/site-packages/scanpy/_utils/__init__.py", line 767, in _choose_graph\n   neighbors = NeighborsView(adata, neighbors_key)\n  File "/home/z6dong/anaconda3/envs/biomania2/lib/python3.10/site-packages/scanpy/_utils/__init__.py", line 711, in __init__\n    raise KeyError('No "neighbors" in .uns')\nKeyError: 'No "neighbors" in .uns'" --method sentencebert --field issue_description --top_k 3
"""

import os
import json
import argparse
from typing import Tuple, List, Dict, Any
from rank_bm25 import BM25Okapi
from ..retrievers import BM25Retriever
from ..gpt.utils import load_json
import ast
from sentence_transformers import SentenceTransformer, util
from ..dataloader.prepare_issue_corpus import ERROR_KEYWORDS, get_error_type

def prepare_corpus(queries: List[Dict[str, Any]], field: str) -> Dict[str, Tuple[List[Dict[str, Any]], List[str]]]:
    """
    Prepares a corpus from the query data, organized by error type.

    Parameters
    ----------
    queries : List[Dict[str, Any]]
        List of query dictionaries.
    field : str
        The field to compare (issue_title or issue_description).

    Returns
    -------
    Dict[str, Tuple[List[Dict[str, Any]], List[str]]]
        A dictionary where keys are error types and values are tuples containing:
        - A list of dictionaries, each representing an issue with its description and solution.
        - A list of strings representing each issue's specified field for BM25 and BERT processing.
    """
    corpus_dict = {}
    for query in queries:
        if query['solution'] is None:
            continue
        error_types = query.get('error_type', {'Other'})
        for error_type in error_types:
            if error_type not in corpus_dict:
                corpus_dict[error_type] = ([], [])
            doc_content = {
                'issue_title': query['issue_title'],
                "issue_description": query['issue_description'],
                "solution": query['solution']
            }
            corpus_dict[error_type][0].append(doc_content)
            corpus_dict[error_type][1].append(doc_content[field])
    return corpus_dict

def bm25_retriever(corpus_texts: List[str], query: str, top_k: int) -> List[int]:
    """
    Retrieves document indices using BM25.

    Parameters
    ----------
    corpus_texts : List[str]
        The corpus to search.
    query : str
        The query to search for.
    top_k : int
        The number of top documents to retrieve.

    Returns
    -------
    List[int]
        A list of indices of retrieved documents.
    """
    tokenized_corpus = [text.split(" ") for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return top_k_indices

def sentencebert_retriever(corpus_texts: List[str], query: str, top_k: int) -> List[int]:
    """
    Retrieves document indices using Sentence-BERT.

    Parameters
    ----------
    corpus_texts : List[str]
        The corpus to search.
    query : str
        The query to search for.
    top_k : int
        The number of top documents to retrieve.

    Returns
    -------
    List[int]
        A list of indices of retrieved documents.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    return [hit['corpus_id'] for hit in hits]

def retrieved_issue_solution(LIB: str, top_k: int, example_query: str, method: str, field: str) -> None:
    """
    Main function to prepare data, create a retriever, and evaluate its performance.

    Parameters
    ----------
    LIB: str
        The library name.
    top_k : int
        The number of top documents to retrieve.
    example_query: str
        An example query to test.
    method: str
        The retrieval method to use ('bm25' or 'sentencebert').
    field: str
        The field to compare (issue_title or issue_description).
    """
    lib_data_path = f'data/github_issues/{LIB}'
    results = load_json(os.path.join(lib_data_path, "API_inquiry_annotate.json"))
    corpus_dict = prepare_corpus(results, field)
    error_types = get_error_type(example_query)
    print('error_types:', error_types)
    combined_corpus = []
    combined_texts = []
    for error_type in error_types:
        if error_type in corpus_dict:
            combined_corpus.extend(corpus_dict[error_type][0])
            combined_texts.extend(corpus_dict[error_type][1])
    if not combined_corpus:
        print(f"No issues found for error types: {', '.join(error_types)}")
        return
    
    if method == 'bm25':
        top_k_indices = bm25_retriever(combined_texts, example_query, top_k)
    elif method == 'sentencebert':
        top_k_indices = sentencebert_retriever(combined_texts, example_query, top_k)
    else:
        raise ValueError("Unsupported method. Use 'bm25' or 'sentencebert'.")

    retrieved_docs = [combined_corpus[i] for i in top_k_indices]
    retrieved_titles = [doc['issue_title'] for doc in retrieved_docs]
    retrieved_issue_description = [doc['issue_description'] for doc in retrieved_docs]
    retrieved_solution = [doc['solution'] for doc in retrieved_docs]
    
    print(f"Query: {example_query}")
    print(f"Retrieved titles: {retrieved_titles}")
    print(f"Retrieved issue descriptions: {retrieved_issue_description}")
    print(f"Retrieved solutions: {retrieved_solution}")

def main():
    parser = argparse.ArgumentParser(description='Query the issue corpus for a library')
    parser.add_argument('--LIB', type=str, required=True, help='Library name')
    parser.add_argument('--example_query', type=str, required=True, help='Example query to test')
    parser.add_argument('--method', type=str, required=True, choices=['bm25', 'sentencebert'], help='Retrieval method to use')
    parser.add_argument('--field', type=str, required=True, choices=['issue_title', 'issue_description'], help='Field to compare')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top documents to retrieve')
    args = parser.parse_args()

    retrieved_issue_solution(args.LIB, args.top_k, args.example_query, args.method, args.field)

if __name__ == "__main__":
    main()
