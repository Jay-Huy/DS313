from transformers import AutoTokenizer
from src.asr_model import ASRModel
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from evaluate import load
from utils import train
import torch
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train ASR Model")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
args = parser.parse_args()

cer = load("cer")

def get_scheduler(optimizer, num_warmup_steps):
    """
    Create a learning rate scheduler with a warm-up strategy.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to apply the scheduler to.
        num_warmup_steps (int): Number of warm-up steps.

    Returns:
        LambdaLR: A learning rate scheduler.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)

# Initialize dataset and dataloaders
train_dataloader = 
val_dataloader = 
# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASRModel(model_dim=768, mode='A').to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# Initialize Training Parameters
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Assuming 0 is the padding index
optimizer = Adam(model.parameters())
num_warmup_steps = 12000
scheduler = get_scheduler(optimizer, num_warmup_steps)

# Training and validation dataloaders
train_metrics_list, val_metrics_list = train(model = model,
                                             tokenizer = tokenizer, 
                                             train_dataloader = train_dataloader, 
                                             val_dataloader = val_dataloader, 
                                             optimizer = optimizer, 
                                             criterion = criterion, 
                                             scheduler = scheduler, 
                                             epochs = args.epochs,
                                             cer = load("cer"))

save_path = "checkpoint.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': args.epochs,
    'train_metrics': train_metrics_list,
    'val_metrics': val_metrics_list
}, save_path)

# criterion = ...  # Define your loss function here
# scheduler = ...  # Initialize your learning rate scheduler here
# train_dataloader = ...  # Load your training DataLoader here
# val_dataloader = ...  # Load your validation DataLoader here
# epochs = 10  # Define the number of epochs
# model_dir = "path/to/save/model"  # Define the directory to save the model

