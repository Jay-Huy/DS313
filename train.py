import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from asr_model import ASRModel
from evaluate import load
from utils import train
from dataset import AISHELL1Dataset, PadCollate, PretrainedVGGExtractor
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train ASR Model")
parser.add_argument("--transcript_path", type=str, required=True, help="Transcript path")
parser.add_argument("--wav_path", type=str, required=True, help="Wav path")
parser.add_argument("--structure", type=str, default='A', choices=['A', 'B', 'C'], help="Model structure")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
parser.add_argument("--batch_size", type=int, default=10, help="Number of epochs for training")
parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for DataLoader")
parser.add_argument("--save_path", type=str, default='checkpoint.pth', help="Path to save the model checkpoint")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a trained checkpoint for continuous training")  # New argument
args = parser.parse_args()

# --- Dataset and Dataloader Configuration ---
AISHELL_TRANSCRIPT_PATH = args.transcript_path
AISHELL_WAV_ROOT = args.wav_path
if not os.path.exists(AISHELL_TRANSCRIPT_PATH):
    exit(f"Không tìm thấy file transcript: {AISHELL_TRANSCRIPT_PATH}")
if not os.path.exists(AISHELL_WAV_ROOT):
    exit(f"Không tìm thấy thư mục wav gốc: {AISHELL_WAV_ROOT}")
available_splits = [d for d in ['train', 'dev', 'test'] if os.path.isdir(os.path.join(AISHELL_WAV_ROOT, d))]
if not available_splits:
    exit(f"Không tìm thấy thư mục con 'train', 'dev', hoặc 'test' trong {AISHELL_WAV_ROOT}")

# Configuration
TOKENIZER_NAME = "bert-base-chinese"
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
RESHAPE_VGG_OUTPUT = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    PAD_IDX = tokenizer.pad_token_id
    if PAD_IDX is None:
        exit("Vui lòng cấu hình pad token cho tokenizer.")
except Exception as e:
    exit(f"Không thể tải tokenizer '{TOKENIZER_NAME}'. Lỗi: {e}")

# Initialize VGG Feature Extractor
vgg_model = PretrainedVGGExtractor(freeze_features=True)

# Initialize PadCollate
pad_collate_instance = PadCollate(
    pad_idx=PAD_IDX,
    vgg_model=vgg_model,
    tokenizer=tokenizer,
    device=device,
    reshape_features=RESHAPE_VGG_OUTPUT
)

# Create Datasets and Dataloaders
datasets = {}
dataloaders = {}
for split in available_splits:
    datasets[split] = AISHELL1Dataset(
        AISHELL_TRANSCRIPT_PATH, AISHELL_WAV_ROOT, split=split
    )
    shuffle_data = (split == 'train')
    dataloaders[split] = DataLoader(
        datasets[split],
        batch_size=BATCH_SIZE,
        shuffle=shuffle_data,
        collate_fn=pad_collate_instance,
        num_workers=NUM_WORKERS
    )

train_dataloader = dataloaders['train']  # Use 'train' split for training
val_dataloader = dataloaders['dev']  # Use 'dev' split for validation

# --- Training Logic ---
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

# Initialize Model
model = ASRModel(model_dim=768, mode=args.structure).to(device)

# Load Checkpoint for Continuous Training
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load a checkpoint into the model, optimizer, and scheduler.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the checkpoint into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.
        scheduler (torch.optim.lr_scheduler, optional): The scheduler to load the state into.

    Returns:
        int: The epoch to resume training from.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint.get('epoch', 0)

# Initialize Training Parameters
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = Adam(model.parameters())
num_warmup_steps = 12000
scheduler = get_scheduler(optimizer, num_warmup_steps)

# Load checkpoint if provided
start_epoch = 0
if args.checkpoint_path:
    if os.path.exists(args.checkpoint_path):
        start_epoch = load_checkpoint(args.checkpoint_path, model, optimizer, scheduler)
    else:
        print(f"Checkpoint path {args.checkpoint_path} does not exist. Starting training from scratch.")

# Train the Model
train_metrics_list, val_metrics_list = train(
    model=model,
    tokenizer=tokenizer,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    epochs=args.epochs,
    cer=cer,
    start_epoch=start_epoch  # Pass the starting epoch
)

# Save the Model
save_path = args.save_path
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': args.epochs,
    'train_metrics': train_metrics_list,
    'val_metrics': val_metrics_list
}, save_path)
