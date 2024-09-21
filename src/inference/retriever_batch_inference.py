"""
Author: Zhengyuan Dong
Email: zydong122@gmail.com
Description: This script contains functions to retrieve documents from a corpus using a pre-trained sentence transformer model.
"""
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import os
from ..inference.utils import process_retrieval_document_query_version, compress_api_str_from_list_query_version
from ..gpt.utils import save_json

def load_query(dataset_type, data_path):
    queries_df = pd.read_csv(os.path.join(data_path, f'{dataset_type}.query.txt'), sep='\t', names=['qid', 'query'])
    labels_df = pd.read_csv(os.path.join(data_path, f'qrels.{dataset_type}.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])
    ir_queries = {row.qid: row.query for _, row in queries_df.iterrows()}
    return labels_df, ir_queries
def load_relevant_docs(labels_df):
    relevant_docs = {}
    for row in labels_df.itertuples():
        relevant_docs.setdefault(row.qid, set()).add(row.docid)
    return relevant_docs

def retrieve_inference(model, dataset_path, dataset_type, top_k, device):
    documents_df = pd.read_csv(os.path.join(dataset_path, 'corpus.tsv'), sep='\t')
    ir_corpus,_ = process_retrieval_document_query_version(documents_df)
    labels_df, ir_queries = load_query(dataset_type, dataset_path)
    ir_relevant_docs = load_relevant_docs(labels_df)

    query_embeddings = model.encode(list(ir_queries.values()), convert_to_tensor=True).to(device)
    corpus_embeddings = model.encode(list(map(' '.join, ir_corpus.values())), convert_to_tensor=True).to(device)
    cos_scores = util.pytorch_cos_sim(query_embeddings, corpus_embeddings)

    successful_match_count = 0
    top_results = {}
    for query_index, (query_id, query) in enumerate(ir_queries.items()):
        relevant_docs_indices = cos_scores[query_index].topk(top_k).indices
        relevant_docs_scores = cos_scores[query_index].topk(top_k).values
        relevant_docs = [(list(ir_corpus.keys())[index], list(ir_corpus.values())[index]) for index in relevant_docs_indices]
        relevant_docs_with_scores = {str((doc_id, tool_name_api_name)): {'score': float(score)} for (doc_id, tool_name_api_name), score in zip(relevant_docs, relevant_docs_scores)}
        # Count the number of successful matches
        matches = len(set([doc_id for doc_id, _ in relevant_docs]) & set(ir_relevant_docs[query_id]))
        if matches>0:
            successful_match_count+=1
        
        # Extract the desired key and split by the specified delimiters
        #print(true_label, predicted_label)
        top_results[query] = {
            'original_docs': [' '.join(ir_corpus[doc_id]) for doc_id in ir_relevant_docs[query_id]],
            'top_docs': relevant_docs_with_scores,
            'successful_matches': matches,
        }
    accuracy = successful_match_count / (query_index+1) * 100
    print(f"Accuracy for {dataset_type}: {accuracy:.2f}, total #{query_index+1}, accurate #{successful_match_count}")

    save_json(os.path.join(args.dataset_path,f"retrieved_{dataset_type}.json"), top_results)

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Your trained model path')
    parser.add_argument('--dataset_path', type=str, help='The processed dataset files path')
    parser.add_argument('--top_k', type=int, help='The output files path')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(args.model_path).to(device)
    retrieve_inference(model, args.dataset_path, 'train', args.top_k, device)
    retrieve_inference(model, args.dataset_path, 'test', args.top_k, device)
    retrieve_inference(model, args.dataset_path, 'val', args.top_k, device)