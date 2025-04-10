import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

# --- Mạng VGG Downsampler ---
class VGGDownsampler(nn.Module):
    """
    Mạng CNN kiểu VGG 4 lớp để downsample feature theo chiều thời gian.
    Input: (Batch, Channel=1, Time, Freq)
    Output: (Batch, OutputChannel, Time/4, Freq)
    """
    def __init__(self, input_channels=1, output_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) # Time/2

        self.conv3 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(output_channels)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) # Time/4

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        return x

# --- Dataset AISHELL-1 ---
class AISHELL1Dataset(Dataset):
    def __init__(self, transcript_path, wav_root_dir, split='train', sample_rate=16000, n_mels=80, frame_length=25, frame_shift=10):
        self.wav_root_dir = wav_root_dir
        self.split = split.lower()
        if self.split not in ['train', 'dev', 'test']:
            raise ValueError("split phải là một trong 'train', 'dev', 'test'")

        self.sample_rate = sample_rate
        print(f"Initializing Dataset for SPLIT='{self.split}' with FBank params: n_mels={n_mels}, frame_length={frame_length}ms, frame_shift={frame_shift}ms")
        self.fbank_params = {
            "sample_frequency": sample_rate, "num_mel_bins": n_mels,
            "frame_length": frame_length, "frame_shift": frame_shift,
            "use_energy": False
        }

        self.data = []
        self.transcripts = self._load_transcripts(transcript_path)
        self._find_wav_files_for_split()

    def _load_transcripts(self, transcript_path):
        transcripts = {}
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        utterance_id = parts[0]
                        text = parts[1]
                        transcripts[utterance_id] = text
        except FileNotFoundError:
            print(f"Không tìm thấy file transcript tại {transcript_path}")
        return transcripts
    def _find_wav_files_for_split(self):
        split_folder_path = os.path.join(self.wav_root_dir, self.split)

        print(f"Đang tìm kiếm file wav trong thư mục của split '{self.split}': {split_folder_path}")

        if not os.path.isdir(split_folder_path):
            print(f"Không tìm thấy thư mục con '{self.split}' tại {split_folder_path}")
            return

        found_count = 0
        missing_transcript_count = 0
        for root, dirs, files in os.walk(split_folder_path):
            for file in files:
                if file.endswith('.wav'):
                    utterance_id = file[:-4]
                    if utterance_id in self.transcripts:
                        wav_path = os.path.join(root, file)
                        self.data.append({
                            'id': utterance_id,
                            'wav_path': wav_path,
                        })
                        found_count += 1
                    else:
                        missing_transcript_count += 1

        print(f"[{self.split.upper()}] Đã tìm thấy {found_count} file wav có transcript.")
        if missing_transcript_count > 0:
             print(f"[{self.split.upper()}] Cảnh báo: Không tìm thấy transcript cho {missing_transcript_count} file wav.")
        if found_count == 0:
             print(f"[{self.split.upper()}] Lỗi: Không tìm thấy file wav nào khớp với transcript trong thư mục này!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        utterance_id = item['id']
        wav_path = item['wav_path']
        try:
            waveform, sr = torchaudio.load(wav_path)
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            fbank_features = kaldi.fbank(waveform, **self.fbank_params)
            return fbank_features, utterance_id
        except Exception as e:
            print(f"Lỗi khi xử lý file {wav_path} trong split '{self.split}': {e}")
            return None, None

# --- Collate Function cho DataLoader ---
def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch: return None, None
    features = [item[0] for item in batch]
    ids = [item[1] for item in batch]
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    return padded_features, ids

# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    AISHELL_TRANSCRIPT_PATH = "C:\\Users\\Tuong\\Downloads\\Compressed\\data_aishell\\data_aishell\\data_aishell\\transcript\\aishell_transcript_v0.8.txt"
    AISHELL_WAV_ROOT = "C:\\Users\\Tuong\\Downloads\\Compressed\\data_aishell\\data_aishell\\data_aishell\\wav" # Thư mục cha chứa train/dev/test
    if not os.path.exists(AISHELL_TRANSCRIPT_PATH): exit(f"Không tìm thấy file transcript tại: {AISHELL_TRANSCRIPT_PATH}")
    if not os.path.exists(AISHELL_WAV_ROOT): exit(f"Không tìm thấy thư mục wav gốc tại: {AISHELL_WAV_ROOT}")
    if not all(os.path.isdir(os.path.join(AISHELL_WAV_ROOT, d)) for d in ['train', 'dev', 'test']):
         print(f"Thiếu một hoặc nhiều thư mục con 'train', 'dev', 'test' bên trong: {AISHELL_WAV_ROOT}")
    SAMPLE_RATE = 16000; N_MELS = 80; FRAME_LENGTH = 25; FRAME_SHIFT = 10
    VGG_OUT_CHANNELS = 128; BATCH_SIZE = 4; NUM_WORKERS = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    print("\n--- Khởi tạo Datasets và DataLoaders cho Train, Dev, Test ---")
    # Tạo Dataset và DataLoader
    datasets = {}
    dataloaders = {}
    for split in ['train', 'dev', 'test']:
        print(f"\n=> Đang tạo Dataset cho split: {split}")
        datasets[split] = AISHELL1Dataset(
            AISHELL_TRANSCRIPT_PATH, AISHELL_WAV_ROOT, split=split,
            sample_rate=SAMPLE_RATE, n_mels=N_MELS,
            frame_length=FRAME_LENGTH, frame_shift=FRAME_SHIFT
        )
        if len(datasets[split]) == 0:
            print(f"!!! Dataset cho split '{split}' bị rỗng. Kiểm tra lại đường dẫn và file.")
            continue

        print(f"=> Đang tạo DataLoader cho split: {split}")
        # Shuffle=True cho tập train, False cho dev và test
        shuffle_data = (split == 'train')
        dataloaders[split] = DataLoader(
            datasets[split], batch_size=BATCH_SIZE,
            shuffle=shuffle_data, collate_fn=collate_fn,
            num_workers=NUM_WORKERS
        )
        print(f"Số lượng mẫu trong dataset '{split}': {len(datasets[split])}")
        print(f"Số lượng batch trong dataloader '{split}': {len(dataloaders[split])}")

    if not dataloaders:
        exit("Không thể tạo bất kỳ DataLoader nào")
    print("\nĐang khởi tạo Model VGG Downsampler...")
    vgg_model = VGGDownsampler(input_channels=1, output_channels=VGG_OUT_CHANNELS).to(device)
    vgg_model.eval()

    # --- Chạy thử 1 batch  ---
    test_split = 'dev' if 'dev' in dataloaders else ('test' if 'test' in dataloaders else 'train')

    if test_split not in dataloaders:
        exit(f"Không có DataLoader nào hợp lệ")

    try:
        test_dataloader = dataloaders[test_split]
        batch_data = next(iter(test_dataloader))

        if batch_data[0] is None: exit("Batch đầu tiên không hợp lệ.")

        features_batch, ids_batch = batch_data
        print(f"\n[DataLoader: {test_split}]")
        print(f"[Trước VGG] Kích thước FBank features batch (đã pad): {features_batch.shape}")

        vgg_input = features_batch.unsqueeze(1).to(device)
        print(f"[Trước VGG] Kích thước input sau khi thêm channel dim: {vgg_input.shape}")

        with torch.no_grad():
            downsampled_features = vgg_model(vgg_input)
        print(f"\n[Sau VGG] Kích thước Output Features đã downsample: {downsampled_features.shape}")

        # --- Chuẩn bị input cho Transformer ---
        B, C_vgg, T_prime_max, F = downsampled_features.shape
        permuted_features = downsampled_features.permute(0, 2, 1, 3).contiguous()
        print(f"[Transformer Input Prep] Shape sau khi permute: {permuted_features.shape}")
        transformer_input_dim = C_vgg * F
        transformer_input_features = permuted_features.reshape(B, T_prime_max, transformer_input_dim)
        print(f"[Transformer Input Prep] Embedding Dimension (C_vgg * F): {transformer_input_dim}")
        print(f"[Transformer Input Prep] Shape cuối cùng sẵn sàng cho Transformer: {transformer_input_features.shape}")

    except StopIteration:
        print(f"Lỗi: DataLoader '{test_split}' không trả về batch nào.")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi trong quá trình chạy thử nghiệm với dataloader '{test_split}': {e}")
        import traceback
        traceback.print_exc()