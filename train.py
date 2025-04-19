import os
import torch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Tokenizer
from asr_model import ASRModel
from evaluate import load
from utils import train
from dataset import AISHELL1Dataset, PadCollate, PretrainedVGGExtractor
import argparse
import torch.multiprocessing as mp

# Ensure compatibility with multiprocessing
mp.set_start_method("spawn", force=True)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train ASR Model")
    parser.add_argument("--structure", type=str, default='A', choices=['A', 'B', 'C'], help="Model structure")
    parser.add_argument("--layer_selection_mode", type=str, required=True, choices=['last6', 'first3_last3'], help="Model structure")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--transcript_path", type=str, required=True, help="Transcript path")
    parser.add_argument("--wav_path", type=str, required=True, help="Wav path")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for DataLoader")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--subset", type=int, choices=[0, 1, 2], default = 0, required=True, help="Subset of train_dataloader to train on (0 or 1 or 2)")
    parser.add_argument("--save_path", type=str, default='checkpoint.pth', help="Path to save the model checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a trained checkpoint for continuous training")
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
    # TOKENIZER_NAME = "Langboat/mengzi-t5-base"
    TOKENIZER_NAME = "uer/t5-base-chinese-cluecorpussmall"
    
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    RESHAPE_VGG_OUTPUT = True
    APPLY_SPEC_AUGMENT = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Tokenizer
    try: 
        # tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_NAME)
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
        
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
        reshape_features=RESHAPE_VGG_OUTPUT,
        apply_spec_augment=APPLY_SPEC_AUGMENT
    )

    torch.manual_seed(42)
    # Create Datasets and Dataloaders
    datasets = {}
    dataloaders = {}
    for split in available_splits:
        if split == 'train':
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

    if args.subset == 0:
        train_dataloader = dataloaders['train']
        print(f'Subset mode: {args.subset} - Full Train Dataloader Length: {len(train_dataloader)}')
    else:
        subset_size = len(train_dataloader.dataset) // 2  # Divide the dataloader into two subsets
        if args.subset == 1:
            train_dataloader = torch.utils.data.Subset(train_dataloader.dataset, range(0, subset_size))
        elif args.subset == 2:
            train_dataloader = torch.utils.data.Subset(train_dataloader.dataset, range(subset_size, len(train_dataloader.dataset)))

        train_dataloader = DataLoader(
            train_dataloader,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=pad_collate_instance,
            num_workers=NUM_WORKERS
        )

        print(f'Subset mode: {args.subset} - Half Train Dataloader Length: {len(train_dataloader)}')

    # --- Training Logic ---
    cer = load("cer")

    def get_scheduler(optimizer, num_warmup_steps, start_step=0):
        """
        Create a learning rate scheduler with a warm-up strategy.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to apply the scheduler to.
            num_warmup_steps (int): Number of warm-up steps.
            start_step (int): The step to resume from in continuous training.

        Returns:
            LambdaLR: A learning rate scheduler.
        """
        def lr_lambda(current_step):
            total_step = current_step + start_step
            if total_step < num_warmup_steps:
                return float(total_step) / float(max(1, num_warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda)

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
        return checkpoint.get('trained_epoch', 0)

    # Initialize Model
    model = ASRModel(model_dim=768, mode=args.structure, layer_selection_mode=args.layer_selection_mode).to(device)
    if torch.cuda.device_count() > 1: 
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        
    # Initialize Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = AdamW(model.parameters())

    # Load Scheduler and Checkpoint
    start_epoch = 0
    # Load Schheduler
    num_warmup_steps = 12000
    steps_per_epoch = len(train_dataloader)

    start_step = start_epoch * steps_per_epoch
    scheduler = get_scheduler(optimizer, num_warmup_steps, start_step=start_step)

    # Load checkpoint if provided
    if args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            start_epoch = load_checkpoint(args.checkpoint_path, model, optimizer, scheduler)
            print(f'The model has been trained on {start_epoch} epochs')
        else:
            print(f"Checkpoint path {args.checkpoint_path} does not exist. Starting training from scratch.")
    
    # Train the Model
    train_metrics_list = train(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=args.epochs,
        cer=cer,
    )

    # Save the Model
    save_path = f"structure_{args.structure}_{args.layer_selection_mode}_epochs_{args.epochs + start_epoch}_subset_{args.subset}.pth"
    if args.subset == 1:
        trained_epoch = start_epoch
    else:
        trained_epoch = args.epochs + start_epoch

    print(f"Saving model checkpoint to {save_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': args.epochs,
        'trained_epoch': trained_epoch,
        'train_metrics': train_metrics_list,
    }, save_path)

if __name__ == '__main__':
    main()
