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
import os
import time

import lightning as L
import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.generate import generate
from models.lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from models.lit_llama.model import LLaMA, LLaMAConfig
from models.lit_llama.tokenizer import Tokenizer
from dataloader.utils.generate_prompt import generate_prompt
from inference.retriever_finetune_inference import ToolRetriever

instruction_tuning = True
eval_interval = 1
save_interval = 10
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 1e-4
micro_batch_size = 4
weight_decay = 0.0
max_seq_length = 4000  # see scripts/prepare_alpaca.py
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_iters = 100

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        batch_size = query_embedding.size(0)
        query_embedding_transformed = self.siamese_network(query_embedding)
        scores = []
        for i in range(batch_size):
            individual_scores = []
            for tool_embedding in retrieved_embeddings[i]:
                tool_embedding_transformed = self.siamese_network(tool_embedding.unsqueeze(0))
                score = F.cosine_similarity(query_embedding_transformed[i].unsqueeze(0), tool_embedding_transformed, dim=-1)
                individual_scores.append(score)
            scores.append(torch.stack(individual_scores))
        scores = torch.cat([s.unsqueeze(0) for s in scores], dim=0)
        softmax_scores = F.softmax(scores, dim=1)
        return softmax_scores

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def print_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

def plot_metrics(train_iter, train_data, val_iter, val_data, test_iter, test_data, ylabel, title, save_path, annotate=False):
    plt.figure(figsize=(10, 6))
    # Plot data
    plt.plot(train_iter, train_data, label="Training", color="blue")
    plt.plot(val_iter, val_data, label="Validation", color="green")
    #plt.plot(test_iter, test_data, label="Test", color="red")
    # Annotate data
    if annotate:
        for j, txt in enumerate(val_data):
            plt.annotate(f"{txt:.2f}", (val_iter[j], val_data[j]), fontsize=8, ha='right')
        for j, txt in enumerate(test_data):
            plt.annotate(f"{txt:.2f}", (test_iter[j], test_data[j]), fontsize=8, ha='right')
        #for j, txt in enumerate(train_data):
        #    plt.annotate(f"{txt:.2f}", (train_iter[j], train_data[j]), fontsize=8, ha='right')
    # Set labels, title, and legend
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # Save and close the plot
    plt.savefig(save_path)
    plt.close()

def plot_results(train_iter, train_losses, train_accuracies, val_iter, val_losses, val_accuracies, test_iter, test_losses, test_accuracies, plot_dir):
    plot_metrics(train_iter, train_losses, val_iter, val_losses, test_iter, test_losses, "Loss", "Losses over Iterations", os.path.join(plot_dir, 'loss_plot.png'), annotate=False)
    plot_metrics(train_iter, train_accuracies, val_iter, val_accuracies, test_iter, test_accuracies, "Accuracy (%)", "Accuracies over Iterations", os.path.join(plot_dir, 'accuracy_plot.png'), annotate=False)

def evaluate_model(model, loader, criterion, mode='Validation'):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for llama_output, retrieved_embeddings, labels, queries, retrieved_names in loader:
            llama_output, retrieved_embeddings, labels = llama_output.to('cuda'), retrieved_embeddings.to('cuda'), labels.to('cuda')
            outputs = model(llama_output, retrieved_embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Check and print wrong predictions
            wrong_idx = (predicted != labels).nonzero(as_tuple=True)[0]
            """for i in wrong_idx:
                print(f"Query: {queries[i]}")
                print(f"True Label: {retrieved_names[i][labels[i].item()]}")
                print(f"Predicted Label: {retrieved_names[i][predicted[i].item()]}\n")"""
    print(f'{mode} Loss: {total_loss / len(loader):.4f}')
    print(f'{mode} Accuracy: {100 * correct / total:.2f}%\n')
    return total_loss, correct, total

def main(
    data_dir: str = "data/alpaca", 
    out_dir: str = "out/lora/alpaca",
    plot_dir: str = "./plot/llama",
    max_iters: int=50,
    batch_size: int=4,
) -> None:
    """
    Main function for executing the training loop of a Siamese network model using the Alpaca dataset.
    This includes data loading, model training, validation, and saving the trained model.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the data files are stored.
    out_dir : str
        Directory where the trained model and other outputs will be saved.
    plot_dir : str
        Directory where plots will be saved.
    max_iters : int
        Maximum number of training iterations.
    batch_size : int
        Number of samples per batch during training.

    Returns
    -------
    None
    """
    os.makedirs(plot_dir, exist_ok=True)
    # model
    model = CombinedModel().to('cuda')
    print_parameters(model)
    # dataset
    train_dataset = CustomDataset(data_dir=data_dir, name='train')
    val_dataset = CustomDataset(data_dir=data_dir, name='val')
    test_dataset = CustomDataset(data_dir=data_dir, name='test')
    train_label_distribution = train_dataset.label_distribution
    val_label_distribution = val_dataset.label_distribution
    test_label_distribution = test_dataset.label_distribution
    print('label distribution for train, val, test: ', train_label_distribution, val_label_distribution, test_label_distribution)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print('len of dataset, train, val, test:', len(train_dataset), len(val_dataset), len(test_dataset))
    # optimize
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()
    train_losses, train_accuracies, train_iter = [], [], []
    val_losses, val_accuracies, val_iter = [], [], []
    test_losses, test_accuracies, test_iter = [], [], []
    for epoch in range(max_iters):
        model.train()  # set model to training mode
        total_loss = 0
        total = 0
        correct_train = 0
        for i, (llama_output, retrieved_embeddings, labels, _, _) in enumerate(train_loader):
            llama_output, retrieved_embeddings, labels = llama_output.to('cuda'), retrieved_embeddings.to('cuda'), labels.to('cuda')
            outputs = model(llama_output, retrieved_embeddings)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)  # changed 1 to 0 for dimension
            total += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{max_iters}], Loss: {total_loss / (i + 1):.4f}')
        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(100 * correct_train / total)
        train_iter.append(epoch)
        if epoch%eval_interval==0:
            # Validation step
            total_val_loss, correct_val, total = evaluate_model(model, val_loader, criterion, mode='Validation')
            val_losses.append(total_val_loss / len(val_loader))
            val_accuracies.append(100 * correct_val / total)
            val_iter.append(epoch)
            total_test_loss, correct_test, total = evaluate_model(model, test_loader, criterion, mode='Test')
            test_losses.append(total_test_loss / len(test_loader))
            test_accuracies.append(100 * correct_test / total)
            test_iter.append(epoch)
            plot_results(train_iter, train_losses, train_accuracies, val_iter, val_losses, val_accuracies, test_iter, test_losses, test_accuracies, plot_dir)
        checkpoint = {"linear_weights": model.state_dict(),}
        torch.save(checkpoint, os.path.join(out_dir, "combined_model_checkpoint.pth"))
    total_train_loss, correct_train, total = evaluate_model(model, train_loader, criterion, mode='Train')
    total_val_loss, correct_val, total = evaluate_model(model, val_loader, criterion, mode='Validation')
    total_test_loss, correct_test, total = evaluate_model(model, test_loader, criterion, mode='Test')
    print('Training finished.')
    checkpoint = {"linear_weights": model.state_dict(),}
    torch.save(checkpoint, os.path.join(out_dir, "combined_model_checkpoint.pth"))
from collections import Counter
class CustomDataset(Dataset):
    def __init__(self, data_dir, name):
        assert name in ['train', 'val', 'test']
        self.data = torch.load(os.path.join(data_dir, f"{name}.pt"))
        self.label_distribution = Counter([item['label'].item() for item in self.data])
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
    labels = torch.stack(labels).unsqueeze(-1).to('cuda')  # Ensure labels are on the right device
    """print(llama_output.shape)
    print(retrieved_embeddings.shape)
    print(labels.shape)"""
    return llama_output, retrieved_embeddings, labels, queries, retrieved_names

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    from jsonargparse.cli import CLI
    CLI(main)
