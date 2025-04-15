import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from asr_model import ASRModel
from evaluate import load
from utils import inference
from dataset import AISHELL1Dataset, PadCollate, PretrainedVGGExtractor
import argparse
import json  # Add this import

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate ASR Model")
    parser.add_argument("--structure", type=str, default='A', choices=['A', 'B', 'C'], help="Model structure")
    parser.add_argument("--transcript_path", type=str, required=True, help="Transcript path")
    parser.add_argument("--wav_path", type=str, required=True, help="Wav path")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for DataLoader")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--test_or_val", type=str, default="test", choices=["val", "test"], help="Split to evaluate on (val or test)")
    parser.add_argument("--output_path", type=str, default="eval_results.json", help="Path to save evaluation results")
    args = parser.parse_args()

    # --- Dataset and Dataloader Configuration ---
    AISHELL_TRANSCRIPT_PATH = args.transcript_path
    AISHELL_WAV_ROOT = args.wav_path
    if not os.path.exists(AISHELL_TRANSCRIPT_PATH):
        exit(f"Không tìm thấy file transcript: {AISHELL_TRANSCRIPT_PATH}")
    if not os.path.exists(AISHELL_WAV_ROOT):
        exit(f"Không tìm thấy thư mục wav gốc: {AISHELL_WAV_ROOT}")
    if not os.path.isdir(os.path.join(AISHELL_WAV_ROOT, args.test_or_val)):
        exit(f"Không tìm thấy thư mục con '{args.test_or_val}' trong {AISHELL_WAV_ROOT}")

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
        reshape_features=RESHAPE_VGG_OUTPUT
    )

    # Create Dataset and Dataloader for the specified split
    eval_dataset = AISHELL1Dataset(
        AISHELL_TRANSCRIPT_PATH, AISHELL_WAV_ROOT, split=args.test_or_val
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate_instance,
        num_workers=NUM_WORKERS
    )
    print(f'{args.test_or_val.capitalize()} Dataloader Length: {len(eval_dataloader)}')

    # --- Evaluation Logic ---
    cer = load("cer")

    # Initialize Model
    model = ASRModel(model_dim=768, mode=args.structure).to(device)
    if torch.cuda.device_count() > 1: 
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Load Checkpoint
    if not os.path.exists(args.checkpoint_path):
        exit(f"Checkpoint path {args.checkpoint_path} does not exist.")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded from {args.checkpoint_path}")

    # Perform Inference
    cer_score, predictions, references = inference(model, tokenizer, eval_dataloader, cer)

    # Save Results to JSON
    results = {
        "cer_score": cer_score,
        "predictions": predictions,
        "references": references,
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # Output Results
    print(f"Average CER Score: {cer_score:.4f}")
    print(f"Results saved to {args.output_path}")
    print("Sample Predictions:")
    for pred, ref in zip(predictions[:5], references[:5]):
        print(f"Prediction: {pred}")
        print(f"Reference: {ref}")
        print("---")

if __name__ == '__main__':
    main()