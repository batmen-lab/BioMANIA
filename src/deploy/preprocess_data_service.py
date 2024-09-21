"""
Author: Zhengyuan Dong
Email: zydong122@gmail.com
"""
from flask import Flask, request, jsonify
from ..models.lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict

app = Flask(__name__)
from ..models.data_classification import DataPreprocess
#from inference.retriever_finetune_inference import ToolRetriever
from ..models.lit_llama.model import LLaMA, LLaMAConfig
from ..models.lit_llama.tokenizer import Tokenizer
import torch
import lightning as L
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PreprocessDataService:
    def __init__(self):
        self.get_args()
        self.load_models_and_data()
    
    def get_args(self):
        # Define the arguments here...
        parser = argparse.ArgumentParser(description="Inference Pipeline")
        parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model")
        parser.add_argument("--pretrained_path", type=str, required=True, help="Path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
        parser.add_argument("--top_k", type=int, default=3, help="Top K value for the retrieval")
        parser.add_argument("--lora_r", type=int, default=8, help="LoRA parameter r")
        parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA parameter alpha")
        parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
        parser.add_argument("--device_count", type=int, default=1, help="Device count for parallel computation")
        parser.add_argument("--debug_mode", type=str, default='1', help="If choose api deterministically")
        self.args = parser.parse_args()
    
    def load_models_and_data(self):
        # Load models and data here
        self.load_tokenizer()
        self.load_llama_model()

    def load_tokenizer(self):
        self.tokenizer = Tokenizer(self.args.tokenizer_path)

    def load_llama_model(self):
        # ... (loading llama model, similar to your existing code)
        if self.args.device_count>0:
            fabric = L.Fabric(accelerator="cuda", devices=self.args.device_count, precision="bf16-true")
        else:
            fabric = L.Fabric(accelerator="cpu", devices=1, precision="bf16-true")
        fabric.launch()
        fabric.seed_everything(1337 + fabric.global_rank)
        # model
        config = LLaMAConfig.from_name("7B")
        config.block_size = self.args.max_seq_length
        if self.args.device_count>0:
            checkpoint = torch.load(self.args.pretrained_path)
        else:
            checkpoint = torch.load(self.args.pretrained_path, map_location=torch.device('cpu'))
        with fabric.init_module(), lora(r=self.args.lora_r, alpha=self.args.lora_alpha, dropout=self.args.lora_dropout, enabled=True):
            self.llama_model = LLaMA(config, return_after_wte=True)
            # strict=False because missing keys due to LoRA weights not contained in checkpoint state
            self.llama_model.load_state_dict(checkpoint, strict=False)
            self.preprocessdata_model = DataPreprocess(self.llama_model, "retriever_placeholder", self.tokenizer, max_seq_length=self.args.max_seq_length, top_k=self.args.top_k,debug_mode="")

preprocess_model = PreprocessDataService()

@app.route('/infer', methods=['GET', 'POST'])
def infer():
    data = request.json
    user_query = data['user_query']
    API_composite = data['API_composite']
    retrieved_names = data['retrieved_names']
    llama_output_query, llama_output_retrieved = preprocess_model.preprocessdata_model.process_data(user_query, API_composite,retrieved_names)
    llama_output_list = llama_output_query.tolist()
    retrieved_embeddings_list = [i.tolist() for i in llama_output_retrieved]
    return jsonify({
        'llama_output': llama_output_list,
        'retrieved_embeddings': retrieved_embeddings_list,
    })

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)