"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
"""
Borrow from https://github.com/Lightning-AI/lit-llama.git
"""
import sys
from pathlib import Path
import os, json
import time
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from inference.utils import process_retrieval_document_query_version, compress_api_str_from_list_query_version, is_pair_in_merged_pairs, find_similar_two_pairs

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.generate import generate
from models.lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from models.lit_llama.model import LLaMA, LLaMAConfig
from models.lit_llama.tokenizer import Tokenizer
from dataloader.utils.generate_prompt import generate_prompt
from inference.retriever_finetune_inference import ToolRetriever

# Hyperparameters
learning_rate = 1e-4

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.embedding_network = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )
        self.embedding_network.apply(self.init_weights)
    def forward(self, x):
        return self.embedding_network(x)
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class CombinedModel(nn.Module):
    def __init__(self, embedding_dim=4096, device='cuda'):
        super(CombinedModel, self).__init__()
        self.device=device
        self.siamese_network = SiameseNetwork(embedding_dim)
    def forward(self, query_embedding, retrieved_embeddings):
        query_embedding_transformed = self.siamese_network(query_embedding)
        scores = []
        for tool_embedding in retrieved_embeddings:
            tool_embedding_transformed = self.siamese_network(tool_embedding)
            score = F.cosine_similarity(query_embedding_transformed, tool_embedding_transformed, dim=-1)
            scores.append(score)
        scores = torch.cat(scores, dim=0).unsqueeze(0)
        softmax_scores = F.softmax(scores, dim=1)
        return softmax_scores

def print_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

def evaluate_model(model, loader, criterion, mode='Validation', LIB=''):
    merged_pairs = find_similar_two_pairs(f"./data/standard_process/{LIB}/API_init.json")
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    non_ambiguous_total = 0
    error_json = []
    correct_logits = []
    wrong_logits = []
    with torch.no_grad():
        for llama_output, retrieved_embeddings, labels, queries, retrieved_names in loader:
            llama_output, retrieved_embeddings, labels = llama_output.to('cuda'), retrieved_embeddings.to('cuda'), labels.to('cuda')
            outputs = model(llama_output, retrieved_embeddings)
            loss = criterion(outputs, labels)
            #print(labels)
            total_loss += loss.item()
            max_logits, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Check and print wrong predictions
            correct_idx = (predicted == labels).nonzero(as_tuple=True)[0]
            correct_logits.extend(max_logits[correct_idx].cpu().numpy().tolist())
            wrong_idx = (predicted != labels).nonzero(as_tuple=True)[0]
            wrong_logits.extend(max_logits[wrong_idx].cpu().numpy().tolist())
            for i in correct_idx:
                error_j = {'Query':queries[i],'gold':retrieved_names[i][labels[i].item()],'pred':retrieved_names[i][predicted[i].item()],'Logits': max_logits[i].item()}
                error_json.append(error_j)
                if not is_pair_in_merged_pairs(retrieved_names[i][labels[i].item()], retrieved_names[i][predicted[i].item()], merged_pairs):
                    non_ambiguous_total += 1
            for i in wrong_idx:
                error_j = {'Query':queries[i],'gold':retrieved_names[i][labels[i].item()],'pred':retrieved_names[i][predicted[i].item()],'Logits': max_logits[i].item()}
                error_json.append(error_j)
                if not is_pair_in_merged_pairs(retrieved_names[i][labels[i].item()], retrieved_names[i][predicted[i].item()], merged_pairs):
                    non_ambiguous_total += 1
    with open(f'./plot/{LIB}/API_error_{mode}.json', 'w') as f:
        json.dump(error_json, f, indent=4)
    print(f'{mode} Loss: {total_loss / len(loader):.4f}')
    print(f'{mode} Accuracy: {100 * correct / total:.2f}%')
    print(f'{mode} ambiguous Accuracy: {100 * correct / non_ambiguous_total:.2f}%')
    return total_loss, correct, total, correct_logits, wrong_logits

def plot_boxplot(correct_logits, wrong_logits, mode):
    data = [correct_logits, wrong_logits]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data)
    plt.xticks([1, 2], ['Correct', 'Wrong'])
    plt.title(f'Boxplot of Logits for {mode} Data')
    plt.ylabel('Logit Value')
    plt.savefig(f'./plot/logit_{mode}.png')

def main(
    data_dir: str = "data/alpaca", 
    batch_size: int=4,
    checkpoint_dir: str = "out/lora/alpaca",
    LIB: str = "",
):
    # model
    model = CombinedModel().to('cuda')
    print_parameters(model)
    # dataset
    train_dataset = CustomDataset(data_dir=data_dir, name='train')
    val_dataset = CustomDataset(data_dir=data_dir, name='val')
    test_dataset = CustomDataset(data_dir=data_dir, name='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # optimize
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss()
    checkpoint = {
        "linear_weights": model.state_dict(),
    }
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint["linear_weights"])
    
    os.makedirs("./plot",exist_ok=True)
    total_train_loss, correct_train, total, train_correct_logits, train_wrong_logits = evaluate_model(model, train_loader, criterion, mode='Train', LIB=LIB)
    plot_boxplot(train_correct_logits, train_wrong_logits, 'Train')
    total_val_loss, correct_val, total, val_correct_logits, val_wrong_logits = evaluate_model(model, val_loader, criterion, mode='Validation', LIB=LIB)
    plot_boxplot(val_correct_logits, val_wrong_logits, 'Validation')
    total_test_loss, correct_test, total, test_correct_logits, test_wrong_logits = evaluate_model(model, test_loader, criterion, mode='Test', LIB=LIB)
    plot_boxplot(test_correct_logits, test_wrong_logits, 'Test')
    
class CustomDataset(Dataset):
    def __init__(self, data_dir, name):
        assert name in ['train', 'val', 'test']
        self.data = torch.load(os.path.join(data_dir, f"{name}.pt"))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        llama_output = item["llama_output"]
        retrieved_embeddings = item["retrieved_embeddings"]
        label = item['label']
        query = item['query']
        retrieved_names = item['retrieved_names']
        return llama_output, retrieved_embeddings, label, query, retrieved_names
        
def collate_fn(batch):
    llama_output, retrieved_embeddings, labels, queries, retrieved_names = zip(*batch)
    llama_output = torch.stack(llama_output).to(dtype=torch.float32).squeeze(1)  # Convert to Float32
    retrieved_embeddings = torch.stack(retrieved_embeddings).to(dtype=torch.float32).to('cuda')
    labels = torch.stack(labels).to('cuda')  # Ensure labels are on the right device
    """print(llama_output.shape)
    print(retrieved_embeddings.shape)
    print(labels.shape)"""
    return llama_output, retrieved_embeddings, labels, queries, retrieved_names

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    from jsonargparse.cli import CLI
    CLI(main)
