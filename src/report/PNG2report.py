from fpdf import FPDF
import os
from PIL import Image
import numpy as np
import subprocess
import json
import argparse
import matplotlib.pyplot as plt

# Parsing arguments for the JSON file path
parser = argparse.ArgumentParser(description="")
parser.add_argument("LIB", help="library")
args = parser.parse_args()
LIB = args.LIB


data_values = {}
###########
# main texts for reports
#############
section_texts = [
    "We assessed the effectiveness of BioMANIA in each stage of the ChatBot, starting from the user initiating the conversation to the end of the conversation. We first investigated BioMANIA's performance in distinguishing between user queries related to API-encoded data analysis and casual chit-chat. We randomly sampled an equal number of chit-chat samples from two existing chit-chat datasets, Topical-Chat and Qnamaker. The number of samples for each cluster is {chitchat_sample_size}. As shown in figure 1.,  the TF-IDF embeddings of chit-chat samples and API-encoded instructions distinctly separate in the t-SNE projection. The notable separation allows a simple nearest centroid classifier to achieve a {chitchat_acc}% classification accuracy on the test data.",
    "After initiating an API call, we examined BioMANIA's performance in identifying the top-3 candidate APIs by utilizing a API Retriever fine-tuned from the BERT-BASE model. We assessed the retrieval performance by comparing it to two baseline methods: the untuned BERT-BASE model and BM25, a commonly used deterministic relevance measure for document comparison. The fine-tuned API Retriever attains an accuracy of {retriever_synthetic}% for synthetic instructions in shortlisting the target API within the top 3 candidates, while the untuned BERT-BASE model reaches {retriever_wo_finetune_synthetic}%. In contrast, the BM25-based API Retriever, which lacks access to training data, achieves an accuracy of {bm25_synthetic}% for synthetic instructions, respectively. The sample size of the splitted training dataset and validation dataset are {retriever_wo_finetune_samples}.",
    "Lastly, we examined BioMANIA's performance in predicting the target API from the shortlist. We assessed the prediction performance from multiple settings, considering factors such as whether to use the LLM for prediction or not, whether to use the API Retriever or not, whether to formulate the problem as a prediction task or a sequence generation task (by generating the API from scratch), and whether to use GPT-3.5 or GPT-4. The performance of in-context API classification model based on GPT or synthetic instructions in predicting the target API from the top 3 candidates is presented as below. It's worth noting that a noticeable portion of misclassifications is caused by the presence of ambiguous APIs by design, which BioMANIA identifies during ChatBot generation. "
]
############
# run inference scripts and extract performance
############
import re
os.environ['HUGGINGPATH'] = './hugging_models'
command = [
    'python', 'inference/retriever_finetune_inference.py',
    '--retrieval_model_path', f'./hugging_models/retriever_model_finetuned/{LIB}/assigned',
    '--corpus_tsv_path', f'./data/standard_process/{LIB}/retriever_train_data/corpus.tsv',
    '--input_query_file', f'./data/standard_process/{LIB}/API_inquiry_annotate.json',
    '--idx_file', f'./data/standard_process/{LIB}/API_instruction_testval_query_ids.json',
    '--retrieved_api_nums', '3',
    '--LIB', LIB
]
process = subprocess.run(command, capture_output=True, text=True)
output = process.stdout
print('output:', output)
accuracy_regex = r"(Training|val|test) Accuracy: ([0-9.]+)%, #samples ([0-9]+)"
matches = re.findall(accuracy_regex, output)
accuracy_data = {match[0]: {"accuracy": float(match[1]), "samples": int(match[2])} for match in matches}
data_values["retriever_synthetic"]= accuracy_data['val']['accuracy']
data_values['retriever_human']=accuracy_data['test']['accuracy']
data_values['retriever_samples']=[accuracy_data['Training']['samples'],accuracy_data['val']['samples'],accuracy_data['test']['samples']]
command = [
    'python', 'inference/retriever_bm25_inference.py',
    '--top_k', '3',
    '--LIB', LIB
]
process = subprocess.run(command, capture_output=True, text=True)
output = process.stdout
print('output:', output)
accuracy_regex_bm25 = r"Totally tested (\d+) (\w+) queries! \w+ retriever top-3 accuracy rate: ([0-9.]+)%"
matches_bm25 = re.findall(accuracy_regex_bm25, output)
accuracy_data_bm25 = {match[1]: {"samples": int(match[0]), "accuracy": float(match[2])} for match in matches_bm25}
data_values["bm25_synthetic"]= accuracy_data_bm25['val']['accuracy']
data_values['bm25_human']=accuracy_data_bm25['test']['accuracy']
data_values['bm25_samples']=[accuracy_data_bm25['train']['samples'], accuracy_data_bm25['val']['samples'], accuracy_data_bm25['test']['samples']]
command = [
    'python', 'models/chitchat_classification.py',
    '--LIB', LIB
]
process = subprocess.run(command, capture_output=True, text=True)
terminal_output_data = process.stdout
print('output:', terminal_output_data)
sample_size_regex = r"length of train_data1, train_data2, train_data3:  (\d+)"
accuracy_regex = r"Accuracy on test data on 2 clusters: ([0-9.]+)"
sample_size_match = re.search(sample_size_regex, terminal_output_data)
sample_size = int(sample_size_match.group(1)) if sample_size_match else None
accuracy_match = re.search(accuracy_regex, terminal_output_data)
accuracy = float(accuracy_match.group(1)) if accuracy_match else None
data_values["chitchat_acc"]= accuracy
data_values["chitchat_sample_size"]= sample_size
command = [
    'python', 'inference/retriever_finetune_inference.py',
    '--retrieval_model_path', "bert-base-uncased",
    '--corpus_tsv_path', f'./data/standard_process/{LIB}/retriever_train_data/corpus.tsv',
    '--input_query_file', f'./data/standard_process/{LIB}/API_inquiry_annotate.json',
    '--idx_file', f'./data/standard_process/{LIB}/API_instruction_testval_query_ids.json',
    '--retrieved_api_nums', '3',
    '--LIB', LIB
]
process = subprocess.run(command, capture_output=True, text=True)
output = process.stdout
print('output:', output)
accuracy_regex = r"(Training|val|test) Accuracy: ([0-9.]+)%, #samples ([0-9]+)"
matches = re.findall(accuracy_regex, output)
accuracy_data = {match[0]: {"accuracy": float(match[1]), "samples": int(match[2])} for match in matches}
data_values["retriever_wo_finetune_synthetic"]= accuracy_data['val']['accuracy']
data_values['retriever_wo_finetune_human']=accuracy_data['test']['accuracy']
data_values['retriever_wo_finetune_samples']=str([accuracy_data['Training']['samples'],accuracy_data['val']['samples']]) # accuracy_data['test']['samples']]
############
# plotting
############
import matplotlib.pyplot as plt
keys = ['bm25_synthetic', 'retriever_wo_finetune_synthetic', 'retriever_synthetic']
values = [float(data_values[key]) for key in keys]
print('values:', values)
plt.figure(figsize=(10,10))
plt.bar(keys, values, color='blue')
plt.title('Retriever Performance')
plt.xlabel('Retrievers')
plt.ylabel('Accuracy')
plt.savefig('./report/retriever_performance.png')

############
# titles
############
filled_section_texts = [text.format(**data_values) for text in section_texts]
section_titles = [
    "Chitchat Classification Analysis",
    "API Retriever Training Insights",
    "GPT Model Performance Comparison"
]
captions = [
    "Chitchat classification model performance",
    "Retriever training performance",
]

# Generate demo images
image_paths = ['plot/scanpy/chitchat_train_tsne_modified.png', 'report/retriever_performance.png']

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)  
        self.cell(0, 10, 'Performance Report', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)  
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    def add_section(self, num, title, body):
        self.set_font('Helvetica', 'B', 10)  
        self.cell(0, 10, f'Section {num}: {title}', 0, 1)
        self.set_font('Helvetica', '', 8)  
        self.multi_cell(0, 10, body)
        self.ln(10)
    def add_image_row(self, image_paths, captions):
        if self.get_y() + 60 > (self.h - 20):
            self.add_page()
        num_images = len(image_paths)
        image_width = (self.w - 20) / num_images
        image_height = image_width * 0.75 
        x_position = (self.w - num_images * image_width) / 2
        y_position = self.get_y()
        self.set_font('Helvetica', '', 8)
        for idx, (image_path, caption) in enumerate(zip(image_paths, captions)):
            self.image(image_path, x=x_position, y=y_position, w=image_width, h=image_height)
            self.set_xy(x_position, y_position + image_height + 2)
            self.multi_cell(image_width, 10, caption, align='C')
            x_position += image_width
        self.ln(10)
    def table(self, df):
        self.set_font('Arial', '', 10)
        col_widths = []
        for col in df.columns:
            max_col_width = max(
                self.get_string_width(str(col)) + 2,  # title width
                max([self.get_string_width(str(x)) + 2 for x in df[col]])  # max content width
            )
            col_widths.append(max_col_width)
        page_width = self.w - 2 * self.l_margin
        if sum(col_widths) > page_width:
            scale_factor = page_width / sum(col_widths)
            col_widths = [w * scale_factor for w in col_widths]
        for i, col_name in enumerate(df.columns):
            self.cell(col_widths[i], 10, col_name, border=1)
        self.ln()
        for row in df.itertuples(index=False):
            for i, item in enumerate(row):
                self.cell(col_widths[i], 10, str(item), border=1)
            self.ln()

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.add_image_row(image_paths, captions)
for i, (title, body) in enumerate(zip(section_titles, filled_section_texts), start=1):
    pdf.add_section(i, title, body)

############
# run scripts for getting gpt api prediction performance
############
import json
from tqdm import auto as tqdm
import logging
logging.basicConfig(level=logging.CRITICAL)  # turn off logging
# load data
with open(f'./data/standard_process/{LIB}/API_inquiry_annotate.json', 'r') as f:
    data = json.load(f)
# all-apis
import re, os
from string import punctuation
end_of_docstring_summary = re.compile(r'[{}\n]+'.format(re.escape(punctuation)))
all_apis = {x['api_name']: end_of_docstring_summary.split(x['Docstring'])[0].strip() for x in data}
all_apis = list(all_apis.items())
all_apis_json = {i[0]:i[1] for i in all_apis}
# For accuracy without ambiguous pair
from collections import defaultdict
with open(f"./data/standard_process/{LIB}/API_composite.json", "r") as file:
    api_composite_data = json.load(file)
    
api_composite_data = {key:api_composite_data[key] for key in api_composite_data if api_composite_data[key]['api_type']!='class'}

# 1: description
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
def find_similar_api_pairs(api_descriptions):
    descriptions = list(api_descriptions.values())
    api_names = list(api_descriptions.keys())
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    similar_pairs = []
    for i in range(len(api_names)):
        for j in range(i+1, len(api_names)):
            if cosine_similarities[i, j] > 0.999:  # threshold can be adjusted
                similar_pairs.append((api_names[i], api_names[j]))
    return similar_pairs
similar_api_pairs = find_similar_api_pairs(all_apis_json)
# 2. 
require_same_depth=False
api_list = list(api_composite_data.keys())
groups = defaultdict(list)
for api in api_list:
    parts = api.split('.')
    if require_same_depth:
        key = (parts[-1], len(parts))
    else:
        key = parts[-1]
    groups[key].append(api)
similar_pairs = [group for group in groups.values() if len(group) > 1]# Filter out groups that only contain 1 API (no similar pairs).
#for pair in similar_pairs:
#    print(pair)

list_1 = similar_api_pairs
list_2 = similar_pairs
pairs_from_list_2 = [(apis[i], apis[j]) for apis in list_2 for i in range(len(apis)) for j in range(i+1, len(apis))]
merged_pairs = list(set(list_1 + pairs_from_list_2))

import glob
import pandas as pd
import os
import json
import re

results = []
def is_pair_in_merged_pairs(gold, pred, merged_pairs):
    # Check if the pair (gold, pred) or (pred, gold) exists in merged_pairs
    return (gold, pred) in merged_pairs or (pred, gold) in merged_pairs
all_apis_from_pairs = set(api for pair in merged_pairs for api in pair)
for fname in glob.glob('./gpt/*/*.json'):
    with open(fname) as f:
        res = json.load(f)
    original_correct = [ex['correct'] for ex in res]
    original_c = [i for i in original_correct if i]
    original_accuracy = sum(original_correct) / len(original_correct) if res else 0
    #filtered_res = [item for item in res if item['gold'] not in all_apis_from_pairs]
    filtered_res = [item for item in res if not is_pair_in_merged_pairs(item['gold'], item['pred'], merged_pairs)]
    
    filtered_correct = [ex['correct'] for ex in filtered_res]
    filtered_c = [i for i in filtered_correct if i]
    filtered_accuracy = sum(filtered_correct) / len(filtered_res) if filtered_res else 0
    parent_dir = os.path.dirname(fname)
    match = re.search('-topk-(\d+)', os.path.basename(fname))
    top_k = int(match.group(1)) if match else '-'
    if os.path.basename(fname).replace('.json', '').startswith('gpt-4'):
        model_name = "gpt-4"
    else:
        model_name = "gpt-3.5-turbo-16k"
    if os.path.basename(fname).replace('.json', '').endswith('trainsample'):
        test_val = 'synthetic'
    else:
        test_val = 'human annotate'
    results.append(dict(
        task=parent_dir,
        model_name=model_name,
        accuracy=original_accuracy,
        total=len(res),
        #total_c = len(original_c),
        filtered_accuracy=filtered_accuracy,
        #filtered_c = len(filtered_c),
        filter_total=len(filtered_res),
        top_k=top_k,
        test_val=test_val,
    ))
results = pd.DataFrame(results)
results = results.sort_values(by=['task', 'model_name', 'top_k','test_val'])
results = results.applymap(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
print(results)
############
# reports
############
pdf.table(results)
pdf_output_path = 'report/performance_report.pdf'
pdf.output(pdf_output_path)

