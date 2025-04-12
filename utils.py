import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from tqdm.auto import tqdm

class MultiHeadedDotAttention(nn.Module):
    def __init__(self, num_heads, features_size, casual = False, dropout=0.1):
        super(MultiHeadedDotAttention, self).__init__()
        self.d_model = features_size
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads

        # Create linear projections
        self.query_linear = nn.Linear(features_size, features_size)
        self.key_linear = nn.Linear(features_size, features_size)
        self.value_linear = nn.Linear(features_size, features_size)

        self.output_linear = nn.Linear(features_size, features_size)

        self.dropout = nn.Dropout(dropout)
        self.casual = casual

    def attention(self, query, key, value, dropout, att_mask = None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1).unsqueeze(-1)  # Shape: (batch_size, 1, seq_length, 1)
            scores = scores.masked_fill(att_mask == 0, float('-inf'))  # Mask rows

        # Apply causal mask if enabled
        if self.casual:
            seq_len = scores.size(-1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device)).bool()
            scores = scores.masked_fill(~causal_mask, float('-inf'))
            
        p_attn = F.softmax(scores, dim=-1)
        p_attn = dropout(p_attn)

        p_attn = torch.nan_to_num(p_attn, nan=0.0) # Avoid NaN values
        return torch.matmul(p_attn, value)

    def forward(self, query, key, value, use_aoa = False, att_mask = None):
        batch_size = query.size(0)

        query_ = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key_ = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value_ = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attended = self.attention(query_, key_, value_, self.dropout, att_mask)
        # Concat using view
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, -1, self.d_model)

        return self.output_linear(attended)
    
class FFN(nn.Module):
    def __init__(self, features_size, dim_feedforward, dropout=0.1):
        super(FFN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(features_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, features_size)
        )
        self.norm = nn.LayerNorm(features_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        ffn_out = self.layer(x)
        return self.dropout(self.norm(ffn_out + x))
    
def loss_fn(outputs, labels, cer_score, criterion, gamma=1.0, ignore_index=0):
    """
    Compute the weighted loss for the Decoder using alpha weights based on CER.

    Args:
        outputs (torch.Tensor): Model predictions of shape (batch_size, seq_length, vocab_size).
        labels (torch.Tensor): Ground truth labels of shape (batch_size, seq_length).
        cer_score (float): Character Error Rate (CER) for the Decoder.
        gamma (float): Hyperparameter to control the influence of CER in alpha computation.
        ignore_index (int): The padding value in the labels to be ignored in the loss calculation.

    Returns:
        torch.Tensor: The computed weighted loss.
    """
    # Ensure cer_score is a tensor
    cer_score = torch.tensor(cer_score, dtype=torch.float32, device=outputs.device)

    # Compute alpha weight based on CER
    alpha = -(cer_score ** gamma) * torch.log(1 - cer_score + 1e-8)  # Add epsilon to avoid log(0)

    # Compute the base loss
    base_loss = criterion(outputs.mT, labels)  # Flatten outputs and labels
    if base_loss < 1:
        try: 
            alpha = -(cer_score ** gamma) * torch.log(1 - cer_score + 1e-8)  # Add epsilon to avoid log(0)
            base_loss = base_loss * alpha  # Apply alpha weight to the loss
        except:
            base_loss = base_loss # Avoid cer_score is bigger than 1

    return base_loss

def step(model, tokenizer, data_loader, optimizer, criterion, device, cer, train=True):
    model.train() if train else model.eval()

    total_loss = 0
    total_cer = 0
    num_batches = len(data_loader)

    for i, batch in tqdm(enumerate(data_loader)):
        # Move batch to device
        if i == 50: break
        for k in batch:
            if batch[k] == torch.Tensor:
                batch[k] = batch[k].to(device=device, non_blocking=True)

        input_ids = batch['transcript_ids']  # Batch_size, seq_length
        downsampled_features = batch['downsampled_features']  # Batch_size, seq_length, feature_dim
        attention_mask = batch['transcript_attention_mask']  # Batch_size, seq_length

        # Shifted left input_ids for loss calculation
        shifted_left_outputs = torch.cat([input_ids[:, 1:], torch.full((input_ids.size(0), 1), tokenizer.pad_token_id, dtype=torch.long, device=device)], dim=1)

        # Forward pass
        outputs = model(input_ids, attention_mask, downsampled_features)  # Batch_size, seq_length, vocab_size

        # Compute CER and loss
        ids_prediction = outputs.argmax(dim=-1)  # Batch_size, seq_length
        predictions = tokenizer.batch_decode(ids_prediction, skip_special_tokens=True)  # Batch_size
        references = tokenizer.batch_decode(shifted_left_outputs, skip_special_tokens=True)  # Batch_size
        cer_score = cer.compute(predictions=predictions, references=references)

        loss = loss_fn(outputs, shifted_left_outputs, cer_score, criterion)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_cer += cer_score

    mean_loss = total_loss / num_batches
    mean_cer = total_cer / num_batches

    return {
        'mean_loss': mean_loss,
        'mean_cer': mean_cer,
    }

def train(model, tokenizer, train_dataloader, val_dataloader, optimizer, criterion, scheduler, epochs, cer, model_dir=None):
    train_metrics_list = []
    val_metrics_list = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        print("Training")
        train_metrics = step(model=model, tokenizer=tokenizer, data_loader=train_dataloader, 
                             optimizer=optimizer, criterion=criterion, device = device, cer= cer, train=True)
        print("Validation")
        val_metrics = step(model=model, tokenizer=tokenizer, data_loader=val_dataloader, 
                           optimizer=optimizer, criterion=criterion, device = device, cer= cer, train=False)

        scheduler.step()

        print(f"Train - Loss: {train_metrics['mean_loss']:.4f}, CER: {train_metrics['mean_cer']:.4f}")
        print(f"Val   - Loss: {val_metrics['mean_loss']:.4f}, CER: {val_metrics['mean_cer']:.4f}")

        # Save metrics to lists
        train_metrics_list.append(train_metrics)
        val_metrics_list.append(val_metrics)

        # Early stopping (optional)
        # early_stopping(val_metrics['mean_loss'], model, epoch, optimizer, scheduler, model_dir)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    return train_metrics_list, val_metrics_list

def inference(model, tokenizer, test_dataloader, cer):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            for k in batch:
                if batch[k] == torch.Tensor:
                    batch[k] = batch[k].to(device=device, non_blocking=True)

            input_ids = batch['transcript_ids']
            audio_features = batch['downsampled_features']
            attention_mask = batch['transcript_attention_mask']
            
            # Shifted left input_ids for loss calculation
            shifted_left_outputs = torch.cat([input_ids[:, 1:], torch.full((input_ids.size(0), 1), tokenizer.pad_token_id, dtype=torch.long, device=device)], dim=1)
        
            # Forward pass
            outputs = model(input_ids, attention_mask, audio_features)
            ids_prediction = outputs.argmax(dim=-1)

            # Decode predictions and references
            decoded_predictions = tokenizer.batch_decode(ids_prediction, skip_special_tokens=True)
            decoded_references = tokenizer.batch_decode(shifted_left_outputs, skip_special_tokens=True)

            predictions.extend(decoded_predictions)
            references.extend(decoded_references)

    cer_score = cer.compute(predictions=predictions, references=references)
    return cer_score, predictions, references
