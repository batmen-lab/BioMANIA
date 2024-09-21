"""
Author: Zhengyuan Dong
Date Created: May 29, 2024
Last Modified: May 29, 2024
Description: Prepare the issue corpus for the specified library
Usage: 
python -m src.dataloader.prepare_issue_corpus --LIB scanpy
Email: zydong122@gmail.com
"""

import os, re, json, argparse
import pandas as pd
from sklearn.utils import shuffle

ERROR_KEYWORDS = [
    'ArithmeticError',
    'AssertionError',
    'AttributeError',
    'BufferError',
    'EOFError',
    'FloatingPointError',
    'GeneratorExit',
    'ImportError',
    'ModuleNotFoundError',
    'IndexError',
    'KeyError',
    'KeyboardInterrupt',
    'LookupError',
    'MemoryError',
    'NameError',
    'NotImplementedError',
    'OSError',
    'OverflowError',
    'RecursionError',
    'ReferenceError',
    'RuntimeError',
    'StopIteration',
    'StopAsyncIteration',
    'SyntaxError',
    'IndentationError',
    'TabError',
    'SystemError',
    'SystemExit',
    'TypeError',
    'UnboundLocalError',
    'UnicodeError',
    'UnicodeEncodeError',
    'UnicodeDecodeError',
    'UnicodeTranslateError',
    'ValueError',
    'ZeroDivisionError'
]

def get_error_type(issue_description):
    original_description = issue_description
    if issue_description:
        issue_description = issue_description.lower()
    else:
        return {'Other'}
    issue_description = issue_description.replace('\n',' ')
    matched_errors = set()
    for error in ERROR_KEYWORDS:
        if re.search(r'\b' + error.lower() + r'\b', issue_description):
            matched_errors.add(error)
    matches = re.findall(r'(\w+Error):', original_description)
    matched_errors.update(matches)
    return matched_errors if matched_errors else {'Other'}

def main():
    parser = argparse.ArgumentParser(description='Prepare the issue corpus for a library')
    parser.add_argument('--LIB', type=str, required=True, help='Library name')
    args = parser.parse_args()

    LIB = args.LIB
    OUTPUT_DIR = f'data/github_issues/{LIB}'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(f'{OUTPUT_DIR}/github_issues.json', 'r') as f:
        issue_solution_pairs = json.load(f)

    corpus_data = []
    for idx, pair in enumerate(issue_solution_pairs):
        error_types = get_error_type(pair['issue_body'])
        doc_content = {
            "issue_title": pair['issue_title'],
            "issue_description": pair['issue_body'],
            "solution": pair['solution'],
            "error_type": list(error_types)
        }
        corpus_data.append((idx, json.dumps(doc_content)))

    corpus_df = pd.DataFrame(corpus_data, columns=["docid", "document_content"])
    corpus_df.to_csv(os.path.join(OUTPUT_DIR, 'corpus.tsv'), sep='\t', index=False)

    queries = []
    for idx, pair in enumerate(issue_solution_pairs):
        error_types = get_error_type(pair['issue_body'])
        queries.append({
            "query_id": idx,
            "issue_title": pair['issue_title'],
            "issue_description": pair['issue_body'],
            "solution": pair['solution'],
            "error_type": list(error_types)
        })

    with open(os.path.join(OUTPUT_DIR, 'API_inquiry.json'), 'w') as f:
        json.dump(queries, f, indent=4)

    annotated_queries = queries

    with open(os.path.join(OUTPUT_DIR, 'API_inquiry_annotate.json'), 'w') as f:
        json.dump(annotated_queries, f, indent=4)

    total_queries = len(queries)
    train_split = int(total_queries * 0.8)
    val_split = int(total_queries * 0.1)
    test_split = total_queries - train_split - val_split

    train_indices = list(range(train_split))
    val_indices = list(range(train_split, train_split + val_split))
    test_indices = list(range(train_split + val_split, total_queries))

    index_data = {
        "test": test_indices,
        "val": val_indices
    }

    with open(os.path.join(OUTPUT_DIR, 'API_instruction_testval_query_ids.json'), 'w') as f:
        json.dump(index_data, f, indent=4)

    def save_split(data, indices, file_prefix):
        split_data = [data[i] for i in indices]
        queries = [(item['query_id'], item['issue_title']) for item in split_data]
        labels = [(item['query_id'], 0, item['query_id'], 1) for item in split_data]

        queries_df = pd.DataFrame(queries, columns=['qid', 'query_text'])
        labels_df = pd.DataFrame(labels, columns=['qid', 'useless', 'docid', 'label'])

        queries_df.to_csv(os.path.join(OUTPUT_DIR, f'{file_prefix}.query.txt'), sep='\t', index=False, header=False)
        labels_df.to_csv(os.path.join(OUTPUT_DIR, f'{file_prefix}.qrels.tsv'), sep='\t', index=False, header=False)

    save_split(queries, train_indices, 'train')
    save_split(queries, val_indices, 'val')
    save_split(queries, test_indices, 'test')

if __name__ == "__main__":
    main()
