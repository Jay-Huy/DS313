import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import torchvision.models as models
from tqdm.auto import tqdm
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
    def __init__(self, transcript_path, wav_root_dir, split='train'):
        self.wav_root_dir = wav_root_dir
        self.split = split.lower()
        self.transcripts = self._load_transcripts(transcript_path)
        if not self.transcripts:
            raise ValueError(f"Không thể tải transcript từ {transcript_path} hoặc file rỗng.")
        self.data = []
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
                        if text:
                            transcripts[utterance_id] = text
        except FileNotFoundError:
            print(f"Không tìm thấy file transcript tại {transcript_path}")
        return transcripts

    def _find_wav_files_for_split(self):
        split_folder_path = os.path.join(self.wav_root_dir, self.split)
        if not os.path.isdir(split_folder_path):
            print(f"Không tìm thấy thư mục con '{self.split}' trong {self.wav_root_dir}")
            return
        for root, _, files in tqdm(os.walk(split_folder_path)):
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


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['wav_path'], item['transcript'], item['id']


class PadCollate:
    def __init__(self, pad_idx, vgg_model, tokenizer, reshape_features=True, apply_spec_augment=True): # Thêm apply_spec_augment
        self.pad_idx = pad_idx
        self.vgg_model = vgg_model
        self.vgg_model.eval()
        self.tokenizer = tokenizer
        self.reshape_features = reshape_features

        self.SAMPLE_RATE = 16000
        self.N_MELS = 80
        self.FRAME_LENGTH = 25
        self.FRAME_SHIFT = 10

        self.apply_spec_augment = apply_spec_augment
        if self.apply_spec_augment:
            # --- Khởi tạo SpecAugment ---
            freq_mask_param = int(0.5 * self.N_MELS)
            T_MAX_EXPECTED = 2000
            time_mask_param = int(0.2 * T_MAX_EXPECTED)
            self.spec_augment = nn.Sequential(
                T.FrequencyMasking(freq_mask_param=freq_mask_param),
                T.TimeMasking(time_mask_param=time_mask_param)
            )
            # --- Kết thúc SpecAugment ---

    def __call__(self, batch):
        wav_paths, transcripts, ids = zip(*batch)

        # Extract FBank features
        fbank_features = []
        feature_lengths = []
        for wav_path in wav_paths:
            # --- Bỏ qua file lỗi hoặc quá ngắn ---
            try:
                waveform, sr = torchaudio.load(wav_path)
                if sr != self.SAMPLE_RATE:
                     resampler = T.Resample(sr, self.SAMPLE_RATE)
                     waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                     waveform = torch.mean(waveform, dim=0, keepdim=True)
                if waveform.shape[1] < self.FRAME_LENGTH * self.SAMPLE_RATE / 1000:
                    continue

                waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
                fbank = kaldi.fbank(waveform, sample_frequency=self.SAMPLE_RATE, num_mel_bins=self.N_MELS, frame_length=self.FRAME_LENGTH, frame_shift=self.FRAME_SHIFT, use_energy=False)
                if fbank.shape[0] == 0:
                    continue

                fbank = (fbank - fbank.mean(dim=0, keepdim=True)) / (fbank.std(dim=0, keepdim=True) + 1e-8)
                fbank_features.append(fbank)
                feature_lengths.append(fbank.shape[0])

            except Exception as e:
                return None # Trả về None nếu có lỗi khi tải/xử lý audio

        if len(fbank_features) != len(batch):
             print(f"Warning: Batch size reduced from {len(batch)} to {len(fbank_features)} due to errors/skips.")
             if not fbank_features: return None
             pass


        # Pad FBank features
        feature_lengths = torch.tensor(feature_lengths, dtype=torch.long)
        padded_fbanks = pad_sequence(fbank_features, batch_first=True, padding_value=0.0)

        # --- Áp dụng SpecAugment ---
        if self.apply_spec_augment:
            padded_fbanks_t = padded_fbanks.transpose(1, 2) # (B, F, T)
            augmented_fbanks_t = self.spec_augment(padded_fbanks_t)
            augmented_fbanks = augmented_fbanks_t.transpose(1, 2) # (B, T, F)
        else:
            augmented_fbanks = padded_fbanks # Không áp dụng augment
        # --- Kết thúc SpecAugment ---

        vgg_input = augmented_fbanks.unsqueeze(1).float()

        # Downsample features using VGG
        with torch.no_grad():
            vgg_input_device = vgg_input.to(next(self.vgg_model.parameters()).device)
            downsampled_features = self.vgg_model(vgg_input_device)
            downsampled_features = downsampled_features.cpu()


        if self.reshape_features:
            B, C_vgg, T_prime_padded, F_vgg = downsampled_features.shape
            downsampled_features = downsampled_features.permute(0, 2, 1, 3).reshape(B, T_prime_padded, C_vgg * F_vgg)

        # Tokenize transcripts
        transcripts_list = list(transcripts)
        tokenized = self.tokenizer(transcripts_list, padding=True, truncation=True, return_tensors="pt", max_length = 256)
        transcript_ids = tokenized['input_ids']
        transcript_attention_mask = tokenized['attention_mask']
        output_feature_lengths = torch.div(feature_lengths, 4, rounding_mode='floor')


        # Return batch dictionary (on CPU)
        return {
            'downsampled_features': downsampled_features.cpu(),  # Tensor (B, T', C*F) or (B, C, T', F)
            'feature_lengths': output_feature_lengths.cpu(), # Tensor (B,) - độ dài SAU VGG
            'original_transcript': transcripts_list,             # List[str]
            'transcript_ids': transcript_ids.cpu(),              # Tensor (B, L_max)
            'transcript_attention_mask': transcript_attention_mask.cpu(),# Tensor (B, L_max)
            'id': list(ids)                                      # List[str]
        }