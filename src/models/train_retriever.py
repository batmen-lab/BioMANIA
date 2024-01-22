import logging, os, json
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#torch.cuda.set_per_process_memory_fraction(0.5)
from sentence_transformers import SentenceTransformer, models, InputExample, losses, LoggingHandler
from models.api_evaluator import APIEvaluator
import argparse
import os
from inference.utils import process_retrieval_document_query_version, compress_api_str_from_list_query_version

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

def get_data(data_path, process_corpus_df):
    documents_df = pd.read_csv(os.path.join(data_path, 'corpus.tsv'), sep='\t')
    ir_corpus, _ = process_corpus_df(documents_df)
    labels_df_train, ir_train_queries = load_query("train", data_path)
    labels_df_test, ir_test_queries = load_query("test", data_path)
    labels_df_val, ir_val_queries = load_query("val", data_path)
    train_samples = []
    for row in labels_df_train.itertuples():
        train_samples.append(InputExample(texts=[ir_train_queries[row.qid], ir_corpus[row.docid]], label=row.label))
    train_relevant_docs = load_relevant_docs(labels_df_train)
    test_relevant_docs = load_relevant_docs(labels_df_test)
    val_relevant_docs = load_relevant_docs(labels_df_val)
    corpus_config = {
        'train': {'queries': ir_train_queries, 'relevant_docs': train_relevant_docs},
        'val': {'queries': ir_val_queries, 'relevant_docs': val_relevant_docs},
        'test': {'queries': ir_test_queries, 'relevant_docs': test_relevant_docs},
    }
    return ir_corpus, train_samples, corpus_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, type=str, required=True,help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--model_name", default=None, type=str, required=True,help="The base model name.")
    parser.add_argument("--output_path", default=None, type=str, required=True,help="The base path where the model output will be saved.")
    parser.add_argument("--num_epochs", default=10, type=int, required=True,help="Train epochs.")
    parser.add_argument("--train_batch_size", default=32, type=int, required=True,help="Train batch size.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, required=True,help="Learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=float, required=True,help="Warmup steps.")
    parser.add_argument("--max_seq_length", default=256, type=int, required=True,help="Max sequence length.")
    parser.add_argument("--optimize_top_k", default=3, type=int, required=True,help="The metric which to save best model")
    parser.add_argument("--plot_dir", default="./plot/retriever/", type=str, required=True,help="plot dir for saving")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    args = parser.parse_args()

    torch.cuda.set_device(int(args.gpu))
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    model_save_path = os.path.join(args.output_path,'assigned')
    os.makedirs(model_save_path, exist_ok=True)

    # Model definition
    word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    ir_corpus, train_samples, corpus_config = get_data(args.data_path,process_retrieval_document_query_version)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size, pin_memory=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    evaluator = APIEvaluator(corpus_config, ir_corpus, fig_path=args.plot_dir,optimize_top_k=args.optimize_top_k)
    # You may need to modify the .fit() method to ensure all data is moved to the correct device during parallel computations
    #from tensorflow.keras.callbacks import EarlyStopping
    #early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(train_objectives=[(train_dataloader, train_loss)],
                    evaluator=evaluator,
                    epochs=args.num_epochs,
                    warmup_steps=args.warmup_steps,
                    optimizer_params={'lr': args.learning_rate},
                    output_path=model_save_path
                    )
if __name__=='__main__':
    main()


