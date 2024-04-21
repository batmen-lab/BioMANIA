
import sys
from pathlib import Path
import os
import random

import lightning as L
import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from models.lit_llama.model import LLaMA, LLaMAConfig
from models.lit_llama.tokenizer import Tokenizer
from inference.retriever_finetune_inference import ToolRetriever
#from inference.utils import process_retrieval_document, compress_api_str_from_list

def compress_api_str_from_list(api):
    api_name = api['api_calling'][0].split('(')[0]
    api_desc_truncated = api['description'].split('\n')[0]
    return_schema = json.dumps(api['Returns'])
    compressed_str = f"{api_name}, {api_desc_truncated}, return_schema: {return_schema}"
    return compressed_str

# Hyperparameters
max_seq_length = 4000  # see scripts/prepare_alpaca.py
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataPreprocess():
    def __init__(self, llama_model, retriever, tokenizer, similarity_metric="cosine", top_k=3, max_seq_length=800, debug_mode='1'):
        super(DataPreprocess, self).__init__()
        self.llama = llama_model
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.similarity_metric = similarity_metric
        self.top_k = top_k
        self.linear = nn.Linear(top_k, top_k)
        self.max_seq_length = max_seq_length
        self.debug_mode = debug_mode
    
    def process_data(self, query, API_composite, retrieved_names):
        # 单个数据sample infer用
        print('start processing data using llama model!')
        str_list = [compress_api_str_from_list(API_composite[tool]) for tool in retrieved_names][:self.top_k]
        encoded_input_query = self.tokenizer.encode(query, bos=True, eos=False, device=device)
        encoded_input_retrieved = [self.tokenizer.encode(info, bos=False, eos=False, device=device) for info in str_list]
        # Combine query with retrieved
        combined_inputs = [encoded_input_query] + encoded_input_retrieved

        max_length = max([len(e) for e in combined_inputs])
        combined_padded = torch.stack([torch.cat([e, torch.zeros(max_length - len(e), dtype=torch.long, device=device)]) for e in combined_inputs])
        with torch.no_grad():
            combined_output = self.llama(combined_padded)
        mask = (combined_padded != 0).float()
        sum_mask = mask.sum(dim=1, keepdim=True)
        combined_output_mean = (combined_output * mask.unsqueeze(-1)).sum(dim=1) / sum_mask
        llama_output_query_mean = combined_output_mean[0].unsqueeze(0)
        llama_output_retrieved_mean = combined_output_mean[1:]
        assert llama_output_query_mean.shape[-1]==4096
        print('end processing data!')
        return llama_output_query_mean, llama_output_retrieved_mean
    def retrieve_names(self,query,actual_api):
        retrieved_names = self.retriever.retrieving(query, top_k=self.top_k)
        if self.debug_mode == '1':
            if actual_api not in retrieved_names:
                retrieved_names = [actual_api] + retrieved_names[:-1]
            assert actual_api in retrieved_names
        random.shuffle(retrieved_names)
        return retrieved_names
    def generate_label(self, api_name, api_name_list):
        if api_name in api_name_list:
            return torch.tensor(api_name_list.index(api_name), dtype=torch.long)
        if self.debug_mode == '1':
            raise ValueError
        return torch.tensor(-1, dtype=torch.long)
    def _process_retrieved_names_and_labels(self,data_point):
        actual_api = data_point['api_name']
        retrieved_names = self.retrieve_names(data_point['query'],actual_api)
        if self.debug_mode == '1':
            if actual_api not in retrieved_names:
                print('correct api manually!')
                retrieved_names = [actual_api] + retrieved_names[:-1]
            assert actual_api in retrieved_names
        #print('retrieved time cost:', time.time()-t1)
        random.shuffle(retrieved_names)
        data_point["retrieved_names"] = retrieved_names
        data_point["label"] = self.generate_label(data_point["api_calling"][0].split('(')[0], data_point["retrieved_names"])
        return data_point
    def create_batches(self, data, batch_size):
        return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    def _process_batch_data_point(self,tmp_data,API_composite,batch_size):
        print('start processing batch data!')
        batches = self.create_batches(tmp_data, batch_size)
        for batch_data_points in tqdm(batches):
            batch_queries = [dp['query'] for dp in batch_data_points]
            batch_retrieved_names = [dp['retrieved_names'] for dp in batch_data_points]
            batch_llama_output_query_mean, batch_llama_output_retrieved_mean = self.process_batch_data(batch_queries, API_composite, batch_retrieved_names)
            for i, data_point in enumerate(batch_data_points):
                data_point["llama_output"] = batch_llama_output_query_mean[i]
                data_point["retrieved_embeddings"] = batch_llama_output_retrieved_mean[i]
        return tmp_data
        
    def update_and_save_dataset(self, data_dir, save_path, idx_file, API_composite,retrieved_path,batch_size=64):
        with open(data_dir, 'r') as f:
            dataset = json.load(f)
        with open(idx_file, "r") as f:
            idx_data = json.load(f)
        
        test_tmp_data = [data for data in dataset if data['query_id'] in idx_data['test']]
        val_tmp_data = [data for data in dataset if data['query_id'] in idx_data['val']]
        #train_tmp_data = [data for data in dataset if data['query_id'] in idx_data['train']]
        train_tmp_data = [data for data in dataset if data['query_id'] not in idx_data['test'] and data['query_id'] not in idx_data['val']]
        print('length of dataset: ', len(dataset), len(train_tmp_data), len(val_tmp_data), len(test_tmp_data))

        # retrieved names and labels
        train_data = [self._process_retrieved_names_and_labels(data_point) for data_point in tqdm(train_tmp_data)]
        val_data = [self._process_retrieved_names_and_labels(data_point) for data_point in tqdm(val_tmp_data)]
        test_data = [self._process_retrieved_names_and_labels(data_point) for data_point in tqdm(test_tmp_data)]

        train_tmp_data = self._process_batch_data_point(train_data,API_composite,batch_size)
        test_tmp_data = self._process_batch_data_point(test_data,API_composite,batch_size)
        val_tmp_data = self._process_batch_data_point(val_data,API_composite,batch_size)
        
        torch.save(train_tmp_data, os.path.join(save_path, "train.pt"))
        torch.save(test_tmp_data, os.path.join(save_path, "test.pt"))
        torch.save(val_tmp_data, os.path.join(save_path, "val.pt"))
    def process_batch_data(self, batch_queries, API_composite, batch_retrieved_names):
        # 1. Prepare input data
        batch_encoded_input_queries = [self.tokenizer.encode(query, bos=True, eos=False, device=device) for query in batch_queries]
        batch_encoded_input_retrieved = [[self.tokenizer.encode(compress_api_str_from_list(API_composite[api_name]), bos=False, eos=False, device=device) for api_name in batch_retrieved_names[i]] for i in range(len(batch_queries))]
        # 2. Combine and pad
        max_length = max([len(query) for query in batch_encoded_input_queries] + [len(retrieved) for sublist in batch_encoded_input_retrieved for retrieved in sublist])
        combined_padded = []
        for i in range(len(batch_queries)):
            query_padded = torch.cat([batch_encoded_input_queries[i], torch.zeros(max_length - len(batch_encoded_input_queries[i]), dtype=torch.long, device=device)])
            retrieved_padded = [torch.cat([encoded, torch.zeros(max_length - len(encoded), dtype=torch.long, device=device)]) for encoded in batch_encoded_input_retrieved[i]]
            combined_padded.extend([query_padded] + retrieved_padded)
        combined_padded_tensor = torch.stack(combined_padded)
        # 3. Pass through LLaMA
        with torch.no_grad():
            combined_output = self.llama(combined_padded_tensor)
        # 4. Separate and average
        mask = (combined_padded_tensor != 0).float()
        sum_mask = mask.sum(dim=1, keepdim=True)
        combined_output_mean = (combined_output * mask.unsqueeze(-1)).sum(dim=1) / sum_mask
        for length in sum_mask[::self.top_k + 1]:
            if length.item() > 100:
                #print(f"Query length exceeds limit: {length.item()} !")
                pass
        
        batch_llama_output_query_mean = []
        batch_llama_output_retrieved_mean = []
        for i in range(0, len(combined_output_mean), self.top_k + 1):
            batch_llama_output_query_mean.append(combined_output_mean[i].unsqueeze(0))
            batch_llama_output_retrieved_mean.append(combined_output_mean[i+1:i+1+self.top_k])
        return torch.stack(batch_llama_output_query_mean), torch.stack(batch_llama_output_retrieved_mean)

class CustomDataset(Dataset):
    def __init__(self, data, API_composite):
        self.data = data
        self.API_composite = API_composite
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.API_composite
def main(
    data_dir: str = "data/alpaca", 
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    tokenizer_path: str = "checkpoints/lit-llama/tokenizer.model",
    out_dir: str = "out/lora/alpaca",
    plot_dir: str = "./plot/llama",
    device_count: int=8,
    corpus_tsv_path: str='',
    retriever_path: str='',
    top_k: int=3,
    debug_mode: str="1",
    save_path: str="./data/classification_train",
    idx_file: str='',
    API_composite_dir:str='',
    batch_size:int = 64,
    retrieved_path:str="./tmp",
    LIB:str="scanpy",
):
    """
    Main function to preprocess data using the specified settings and models.

    Parameters
    ----------
    data_dir : str
        Directory containing the data to process.
    pretrained_path : str
        Path to the pretrained model file.
    tokenizer_path : str
        Path to the tokenizer model file.
    out_dir : str
        Output directory for storing results.
    plot_dir : str
        Directory for saving plots.
    device_count : int
        Number of devices to use.
    corpus_tsv_path : str
        Path to the corpus TSV file used by the retriever.
    retriever_path : str
        Path to the trained retriever model.
    top_k : int
        Number of top-k retrievals to consider for each query.
    debug_mode : str
        Specifies the debug mode (default is "1").
    save_path : str
        Path to save the processed dataset.
    idx_file : str
        Path to the index file containing test/validation indices.
    API_composite_dir : str
        Directory containing API composite data in JSON format.
    batch_size : int
        Batch size for processing data.
    retrieved_path : str
        Path where retrieved results are temporarily stored.
    LIB : str
        Library identifier used in retriever initialization.
    """
    os.makedirs(plot_dir, exist_ok=True)
    with open(API_composite_dir, 'r') as json_file:
        data = json.load(json_file)
    API_composite = data
    fabric = L.Fabric(accelerator="cuda", devices=device_count, precision="bf16") # bf16-true
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)
    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    # model
    config = LLaMAConfig.from_name("7B")
    config.block_size = max_seq_length
    checkpoint = torch.load(pretrained_path)
    with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        llama_model = LLaMA(config, return_after_wte=True)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        llama_model.load_state_dict(checkpoint, strict=False)
    print(llama_model)
    tokenizer = Tokenizer(tokenizer_path)
    retriever = ToolRetriever(LIB=LIB, corpus_tsv_path=corpus_tsv_path, model_path=retriever_path)
    model = DataPreprocess(llama_model, retriever, tokenizer, max_seq_length=max_seq_length, top_k=top_k,debug_mode=debug_mode)
    os.makedirs(save_path, exist_ok=True)
    model.update_and_save_dataset(data_dir, save_path,idx_file,API_composite,retrieved_path,batch_size=batch_size)

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    from jsonargparse.cli import CLI
    CLI(main)
