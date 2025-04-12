import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import torchvision.models as models
import tqdm
from transformers import AutoTokenizer
try:
    from torchvision.models import VGG16_Weights
except ImportError:
    VGG16_Weights = None

class PretrainedVGGExtractor(nn.Module):
    """
    Sử dụng các lớp đầu của VGG16 pretrained để downsample feature theo chiều thời gian.
    - Lớp Conv đầu tiên được điều chỉnh cho đầu vào 1 kênh.
    - Các lớp MaxPool2d được sửa để chỉ downsample theo chiều thời gian (kernel=(2,1), stride=(2,1)).
    Input: (Batch, Channel=1, Time_padded, Freq) -> Tensor batch đầu vào
    Output: (Batch, OutputChannel, Time_padded/4, Freq) -> Tensor batch đầu ra sau VGG
    """
    def __init__(self, freeze_features=True):
        super().__init__()
        # Tải VGG16 pretrained
        if VGG16_Weights:
              vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
              vgg16 = models.vgg16(pretrained=True)
        vgg_features = vgg16.features

        # Chỉnh sửa lớp Conv đầu tiên cho đầu vào 1 kênh (thay vì 3 kênh RGB)
        original_first_layer = vgg_features[0]
        new_first_layer = nn.Conv2d(1, original_first_layer.out_channels,
                                    kernel_size=original_first_layer.kernel_size,
                                    stride=original_first_layer.stride,
                                    padding=original_first_layer.padding)
        new_first_layer.weight.data = torch.mean(original_first_layer.weight.data, dim=1, keepdim=True)
        if original_first_layer.bias is not None:
            new_first_layer.bias.data = original_first_layer.bias.data

        modified_layers = [new_first_layer]
        num_pools = 0
        # Giữ lại một phần các lớp đầu của VGG và chỉnh sửa lớp MaxPool
        for i in range(1, 10):
            layer = vgg_features[i]
            if isinstance(layer, nn.MaxPool2d):
                # Chỉ pooling theo chiều thời gian
                modified_layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
                num_pools += 1
                if num_pools >= 2:
                    break
            else:
                modified_layers.append(layer)

        self.features = nn.Sequential(*modified_layers)
        self.output_channels = None
        for layer in reversed(modified_layers):
            if hasattr(layer, 'out_channels'):
                self.output_channels = layer.out_channels
                break
        if self.output_channels is None:
            raise RuntimeError("Không thể tự động xác định output_channels từ các lớp đã chọn trong VGG.")
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: (Batch, 1, Time_padded, Freq)
        return self.features(x) # Output: (Batch, C_out, Time_padded/4, Freq)

class AISHELL1Dataset(Dataset):
    def __init__(self, transcript_path, wav_root_dir, tokenizer, split='train', sample_rate=16000, n_mels=80, frame_length=25, frame_shift=10):
        self.wav_root_dir = wav_root_dir
        self.split = split.lower()
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer # Lưu tokenizer được truyền vào
        self.pad_idx = self.tokenizer.pad_token_id # Lấy PAD ID từ tokenizer

        self.fbank_params = { # Tham số trích xuất FBank
            "sample_frequency": sample_rate, "num_mel_bins": n_mels,
            "frame_length": frame_length, "frame_shift": frame_shift,
            "use_energy": False
        }

        self.data = []
        self.transcripts = self._load_transcripts(transcript_path)
        if not self.transcripts:
              raise ValueError(f"Không thể tải transcript từ {transcript_path} hoặc file rỗng.")

        self._find_wav_files_for_split()

    def _load_transcripts(self, transcript_path):
        transcripts = {}
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        utterance_id = parts[0]
                        text = ' '.join(parts[1].split())
                        if text: transcripts[utterance_id] = text
        except FileNotFoundError:
            print(f"Không tìm thấy file transcript tại {transcript_path}")
        return transcripts

    def _find_wav_files_for_split(self):
        # Tìm các file wav tương ứng với transcript trong split hiện tại
        split_folder_path = os.path.join(self.wav_root_dir, self.split)
        if not os.path.isdir(split_folder_path):
            print(f"Không tìm thấy thư mục con '{self.split}' trong {self.wav_root_dir}")
            return
        found_count = 0
        for root, _, files in os.walk(split_folder_path):
            for file in files:
                if file.endswith('.wav'):
                    utterance_id = file[:-4]
                    if utterance_id in self.transcripts:
                        wav_path = os.path.join(root, file)
                        self.data.append({
                            'id': utterance_id,
                            'wav_path': wav_path,
                            'transcript': self.transcripts[utterance_id]
                        })
                        found_count += 1
        print(f"  [{self.split.upper()}] Tìm thấy {found_count} file WAV có transcript.")
        if found_count == 0:
             print(f"  [{self.split.upper()}] Không tìm thấy file WAV nào khớp!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        utterance_id = item['id']
        wav_path = item['wav_path']
        transcript_text = item['transcript']

        try:
            waveform, sr = torchaudio.load(wav_path)
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8) # Chuẩn hóa waveform

            # --- Trích xuất FBank ---
            fbank_features = kaldi.fbank(waveform, **self.fbank_params) # Shape: (Time, Freq)
            # Chuẩn hóa FBank (CMVN từng utterance)
            fbank_features = (fbank_features - fbank_features.mean(dim=0, keepdim=True)) / (fbank_features.std(dim=0, keepdim=True) + 1e-8)
            token_ids = self.tokenizer.encode(
                transcript_text,
                add_special_tokens=True,
                truncation=False
            )
            transcript_tensor = torch.LongTensor(token_ids)

            # Trả về FBank thô, transcript đã tokenize và metadata
            return fbank_features, transcript_tensor, transcript_text, wav_path, utterance_id

        except Exception as e:
            print(f"Lỗi xử lý file {wav_path} (ID: {utterance_id}) trong split '{self.split}': {e}")
            return None, None, None, None, None # Trả về None để lọc trong collate_fn
class PadCollate:
    """
    Xử lý padding, trích xuất đặc trưng VGG và định dạng batch.
    Args:
        pad_idx (int): Index để padding token transcript.
        vgg_model (nn.Module): Model PretrainedVGGExtractor đã khởi tạo.
        device (torch.device): Thiết bị để chạy VGG ('cuda' hoặc 'cpu').
        reshape_features (bool): True nếu muốn reshape output VGG thành (B, T', C*F).
    """
    def __init__(self, pad_idx, vgg_model, device, reshape_features=True):
        self.pad_idx = pad_idx
        self.vgg_model = vgg_model.to(device)
        self.vgg_model.eval()
        self.device = device
        self.reshape_features = reshape_features

    def __call__(self, batch):
        # Lọc bỏ các sample bị lỗi
        batch = [item for item in batch if item[0] is not None and item[1] is not None]
        if not batch:
            return None

        # Tách các thành phần
        fbank_features = [item[0] for item in batch]
        transcripts = [item[1] for item in batch]
        original_texts = [item[2] for item in batch]
        wav_paths = [item[3] for item in batch]
        ids = [item[4] for item in batch]
        feature_lengths = torch.tensor([f.shape[0] for f in fbank_features], dtype=torch.long) # Độ dài FBank (chiều Time)
        transcript_lengths = torch.tensor([t.shape[0] for t in transcripts], dtype=torch.long) # Độ dài transcript

        try:
            # Pad FBank features (padding chiều Time)
            padded_fbanks = pad_sequence(fbank_features, batch_first=True, padding_value=0.0) # Shape: (B, T_max, F)

            # Pad transcripts
            padded_transcripts = pad_sequence(transcripts, batch_first=True, padding_value=self.pad_idx) # Shape: (B, L_max)
            # Reshape FBank cho VGG: (B, T_max, F) -> (B, 1, T_max, F)
            vgg_input = padded_fbanks.unsqueeze(1).float().to(self.device)

            with torch.no_grad():
                downsampled_features = self.vgg_model(vgg_input) # Shape: (B, C_vgg, T_max/4, F)
            output_feature_lengths = torch.div(feature_lengths, 4, rounding_mode='floor')

            # --- Reshape output VGG ---
            if self.reshape_features:
                # Từ (B, C_vgg, T_prime, F) sang (B, T_prime, C_vgg * F)
                B, C_vgg, T_prime_padded, F_vgg = downsampled_features.shape
                downsampled_features = downsampled_features.permute(0, 2, 1, 3) # (B, T', C, F)
                downsampled_features = downsampled_features.reshape(B, T_prime_padded, C_vgg * F_vgg) # (B, T', C*F)

            # --- Trả về dữ liệu dạng dictionary ---
            return {
                'ids': ids,                                     # List[str]
                'wav_paths': wav_paths,                         # List[str]
                'original_transcripts': original_texts,         # List[str]
                'features': downsampled_features,               # Tensor (B, T', C*F) hoặc (B, C, T', F) trên device
                'feature_lengths': output_feature_lengths,      # Tensor (B,) - độ dài SAU VGG
                'target_tokens': padded_transcripts,            # Tensor (B, L_max) trên CPU
                'target_lengths': transcript_lengths            # Tensor (B,) trên CPU
            }

        except Exception as e:
            print(f"Lỗi trong quá trình collate: {e}")
            import traceback
            traceback.print_exc()
            problem_ids = ids if 'ids' in locals() else "Batch không xác định"
            print(f"Lỗi xảy ra có thể trong batch chứa ID: {problem_ids}")
            return None

if __name__ == "__main__":
    AISHELL_TRANSCRIPT_PATH = "C:\\Users\\Tuong\\Downloads\\Compressed\\data_aishell\\data_aishell\\data_aishell\\transcript\\aishell_transcript_v0.8.txt"
    AISHELL_WAV_ROOT = "C:\\Users\\Tuong\\Downloads\\Compressed\\data_aishell\\data_aishell\\data_aishell\\wav"
    if not os.path.exists(AISHELL_TRANSCRIPT_PATH): exit(f"Không tìm thấy file transcript: {AISHELL_TRANSCRIPT_PATH}")
    if not os.path.exists(AISHELL_WAV_ROOT): exit(f"Không tìm thấy thư mục wav gốc: {AISHELL_WAV_ROOT}")
    available_splits = [d for d in ['train', 'dev', 'test'] if os.path.isdir(os.path.join(AISHELL_WAV_ROOT, d))]
    if not available_splits: exit(f"Không tìm thấy thư mục con 'train', 'dev', hoặc 'test' trong {AISHELL_WAV_ROOT}")
    # print(f"Các split dữ liệu tìm thấy: {available_splits}")

    # --- Cấu hình ---
    TOKENIZER_NAME = "bert-base-chinese" # Tên tokenizer sẽ sử dụng
    SAMPLE_RATE = 16000
    N_MELS = 80 # Số lượng FBank mel bins
    FRAME_LENGTH = 25 # ms
    FRAME_SHIFT = 10  # ms
    BATCH_SIZE = 32   # Chỉnh sửa dựa trên bộ nhớ GPU
    NUM_WORKERS = 4   # Bắt đầu với 0 để dễ debug
    RESHAPE_VGG_OUTPUT = True # True để reshape output VGG thành (B, T', C*F)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Sử dụng thiết bị: {device}")

    # Tải Tokenizer
    # print(f"Đang tải Tokenizer: {TOKENIZER_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        PAD_IDX = tokenizer.pad_token_id
        if PAD_IDX is None:
             # print("Cảnh báo: Tokenizer thiếu pad_token_id mặc định!")
             exit("Vui lòng cấu hình pad token cho tokenizer.")
        # print(f"Tokenizer đã tải xong. PAD ID: {PAD_IDX}")
    except Exception as e:
        exit(f"Không thể tải tokenizer '{TOKENIZER_NAME}'. Lỗi: {e}")
    # print("Đang khởi tạo VGG Feature Extractor...")
    vgg_model = PretrainedVGGExtractor(freeze_features=True)
    # print(f"VGG đã khởi tạo. Kênh output: {vgg_model.output_channels}")

    # --- Tạo Datasets và DataLoaders ---
    datasets = {}
    dataloaders = {}

    # Khởi tạo Hàm Collate (Chứa VGG model và device)
    pad_collate_instance = PadCollate(
        pad_idx=PAD_IDX,
        vgg_model=vgg_model,
        device=device,
        reshape_features=RESHAPE_VGG_OUTPUT
    )

    for split in available_splits:
        # print(f"\n--- Chuẩn bị Split: {split} ---")
        try:
            datasets[split] = AISHELL1Dataset(
                AISHELL_TRANSCRIPT_PATH, AISHELL_WAV_ROOT,
                tokenizer=tokenizer, # Truyền tokenizer vào dataset
                split=split,
                sample_rate=SAMPLE_RATE, n_mels=N_MELS,
                frame_length=FRAME_LENGTH, frame_shift=FRAME_SHIFT
            )

            if len(datasets[split]) == 0:
                print(f"!!! Dataset cho '{split}' rỗng. Bỏ qua.")
                continue

            shuffle_data = (split == 'train') # Chỉ xáo trộn dữ liệu train
            pin_memory_flag = (device == torch.device("cuda") and NUM_WORKERS == 0)

            dataloaders[split] = DataLoader(
                datasets[split],
                batch_size=BATCH_SIZE,
                shuffle=shuffle_data,
                collate_fn=pad_collate_instance, # Sử dụng hàm collate đã chỉnh sửa
                num_workers=NUM_WORKERS,
                pin_memory=pin_memory_flag
            )
            print(f"DataLoader cho '{split}'. Số batch: {len(dataloaders[split])}")

        except Exception as e:
            print(f"Lỗi khi tạo dataset/dataloader cho split '{split}': {e}")
            import traceback
            traceback.print_exc()

    if not dataloaders:
        exit("Không thể tạo bất kỳ DataLoader nào.")

    # --- Kiểm tra thử một batch ---
    test_split = next((s for s in ['dev', 'test', 'train'] if s in dataloaders), None)
    if test_split:
        # print(f"\n--- Kiểm tra một batch từ DataLoader '{test_split}' ---")
        try:
            # Lấy một batch
            batch_dict = next(iter(dataloaders[test_split]))

            if batch_dict is None:
                 exit(f"Batch đầu tiên từ DataLoader '{test_split}' là None (có thể do lỗi).")

            print("Các keys trong Batch Dictionary:", batch_dict.keys())

            # Kiểm tra shapes và devices
            features = batch_dict['features']
            feature_lengths = batch_dict['feature_lengths']
            target_tokens = batch_dict['target_tokens']
            target_lengths = batch_dict['target_lengths']
            ids = batch_dict['ids']

            print(f"\n[Features (Output VGG)] Shape: {features.shape}")
            print(f"[Features (Output VGG)] Device: {features.device}")
            print(f"[Feature Lengths (Sau VGG)] Shape: {feature_lengths.shape}")
            print(f"[Feature Lengths (Sau VGG)] Giá trị: {feature_lengths.tolist()}")
            print(f"\n[Target Tokens] Shape: {target_tokens.shape}")
            print(f"[Target Tokens] Device: {target_tokens.device}")
            print(f"[Target Tokens] dtype: {target_tokens.dtype}")
            print(f"[Target Lengths] Shape: {target_lengths.shape}")
            print(f"[Target Lengths] Giá trị: {target_lengths.tolist()}")
            print(f"\n[IDs] Số lượng: {len(ids)}")
            print(f"   Input features shape: {features.shape} (trên {features.device})")
            print(f"   Target tokens shape: {target_tokens.shape} (trên {target_tokens.device})")

        except StopIteration:
            print(f"DataLoader '{test_split}' không trả về batch nào.")
        except Exception as e:
            print(f"\n!!! Lỗi trong quá trình kiểm tra batch '{test_split}':")
            print(e)
            import traceback
            traceback.print_exc()
    else:
        print("\nKhông có DataLoader nào để chạy kiểm tra batch.")