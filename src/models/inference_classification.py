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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from inference.utils import is_pair_in_merged_pairs, find_similar_two_pairs
from gpt.utils import save_json
from typing import List, Tuple

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# Hyperparameters
learning_rate = 1e-4

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        """
        Initialize the Siamese Network with sequential layers to transform embeddings.

        Parameters
        ----------
        embedding_dim : int
            Dimension of the input and output embeddings.

        Returns
        -------
        None
        """
        super(SiameseNetwork, self).__init__()
        self.embedding_network = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )
        self.embedding_network.apply(self.init_weights)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Siamese Network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be transformed.

        Returns
        -------
        torch.Tensor
            Transformed tensor after passing through the network.
        """
        return self.embedding_network(x)
    def init_weights(self, m: nn.Module) -> None:
        """
        Initialize weights for the Linear layers using Xavier uniform distribution.

        Parameters
        ----------
        m : nn.Module
            Module to initialize.

        Returns
        -------
        None
        """
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class CombinedModel(nn.Module):
    def __init__(self, embedding_dim: int = 4096, device: str = 'cuda') -> None:
        """
        Initialize the Combined Model containing the Siamese Network.

        Parameters
        ----------
        embedding_dim : int, optional
            Dimension of embeddings processed by the network, by default 4096.
        device : str, optional
            Device to deploy the model, by default 'cuda'.

        Returns
        -------
        None
        """
        super(CombinedModel, self).__init__()
        self.device=device
        self.siamese_network = SiameseNetwork(embedding_dim)
    def forward(self, query_embedding: torch.Tensor, retrieved_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the softmax scores for query embeddings against a set of retrieved embeddings.

        Parameters
        ----------
        query_embedding : torch.Tensor
            The embedding of the query.
        retrieved_embeddings : List[torch.Tensor]
            List of embeddings for the retrieved documents or tools.

        Returns
        -------
        torch.Tensor
            Softmax scores comparing the query embedding with each retrieved embedding.
        """
        query_embedding_transformed = self.siamese_network(query_embedding)
        scores = []
        for tool_embedding in retrieved_embeddings:
            tool_embedding_transformed = self.siamese_network(tool_embedding)
            score = F.cosine_similarity(query_embedding_transformed, tool_embedding_transformed, dim=-1)
            scores.append(score)
        scores = torch.cat(scores, dim=0).unsqueeze(0)
        softmax_scores = F.softmax(scores, dim=1)
        return softmax_scores

def print_parameters(model: nn.Module) -> None:
    """
    Prints the names and shapes of the trainable parameters in a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        The model from which to print parameters.

    Returns
    -------
    None
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, mode: str = 'Validation', LIB: str = '') -> Tuple[float, int, int, List[float], List[float]]:
    """
    Evaluate a model using a given data loader and criterion.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    loader : DataLoader
        The DataLoader providing the dataset.
    criterion : nn.Module
        The loss criterion used for evaluation.
    mode : str, optional
        Mode of evaluation (e.g., 'Validation', 'Test'), by default 'Validation'.
    LIB : str, optional
        Library identifier used in model evaluation to handle specific cases or custom functionality, by default ''.

    Returns
    -------
    Tuple[float, int, int, List[float], List[float]]
        Returns the total loss, number of correct predictions, total number of predictions,
        list of logits for correct predictions, and list of logits for incorrect predictions.
    """
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
    save_json(f'./plot/{LIB}/API_error_{mode}.json', error_json)
    print(f'{mode} Loss: {total_loss / len(loader):.4f}')
    print(f'{mode} Accuracy: {100 * correct / total:.2f}%')
    print(f'{mode} ambiguous Accuracy: {100 * correct / non_ambiguous_total:.2f}%')
    return total_loss, correct, total, correct_logits, wrong_logits

def plot_boxplot(correct_logits: List[float], wrong_logits: List[float], mode: str) -> None:
    """
    Plot a boxplot comparing logits distributions for correct and incorrect model predictions.

    Parameters
    ----------
    correct_logits : List[float]
        Logits for correct predictions.
    wrong_logits : List[float]
        Logits for incorrect predictions.
    mode : str
        Describes the dataset context (e.g., 'Train', 'Validation', 'Test') for the plot title.

    Returns
    -------
    None
    """
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
) -> None:
    """
    Main execution function to load a trained model, evaluate it, and plot logits distributions for correct and incorrect predictions.

    Parameters
    ----------
    data_dir : str
        The directory where the dataset is stored.
    batch_size : int
        The number of samples per batch.
    checkpoint_dir : str
        The directory where the trained model checkpoint is stored.
    LIB : str
        Library identifier used in model evaluation to handle specific cases or custom functionality.

    Returns
    -------
    None
    """
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

import inspect
__all__ = list(set([name for name, obj in locals().items() if not name.startswith('_') and (inspect.isfunction(obj) or (inspect.isclass(obj) and name != '__init__') or (inspect.ismethod(obj) and not name.startswith('_')))]))

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    from jsonargparse.cli import CLI
    CLI(main)
