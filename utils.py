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
    
def _shift_right(input_ids, decoder_start_token_id, pad_token_id):
    """
    Shifts the input_ids to the right and prepends the decoder_start_token_id.

    Args:
        input_ids (torch.Tensor): Tensor of shape (batch_size, seq_length) containing input token IDs.
        decoder_start_token_id (int): The token ID to prepend at the start of each sequence.
        pad_token_id (int): The token ID used for padding.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, seq_length) with shifted input IDs.
    """
    if decoder_start_token_id is None:
        raise ValueError(
            "decoder_start_token_id must be defined. In T5, it is usually set to the pad_token_id."
        )

    if pad_token_id is None:
        raise ValueError("pad_token_id must be defined.")

    # Create a tensor for the shifted input IDs
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)

    # Shift inputs to the right
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()

    # Replace the first token with the decoder_start_token_id
    shifted_input_ids[..., 0] = decoder_start_token_id

    # Replace possible -100 values in labels with `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def _shift_left(input_ids, pad_token_id):
    """
    Shifts the input_ids to the left and appends the pad_token_id at the end.

    Args:
        input_ids (torch.Tensor): Tensor of shape (batch_size, seq_length) containing input token IDs.
        pad_token_id (int): The token ID used for padding.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, seq_length) with left-shifted input IDs.
    """
    # Create a tensor for the shifted input IDs
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)

    # Shift inputs to the left
    shifted_input_ids[..., :-1] = input_ids[..., 1:].clone()

    # Replace the last token with the pad_token_id
    shifted_input_ids[..., -1] = pad_token_id

    return shifted_input_ids

def ignore_padding(outputs, labels, padding_values = 1):

  mask = labels != padding_values

  new_outputs, new_labels = [], []

  for i, each in enumerate(mask):
    ignore_outputs = outputs[i][each]
    ignore_labels = labels[i][each]

    new_outputs.append(ignore_outputs), new_labels.append(ignore_labels)

  return new_outputs, new_labels

def convert_ids_to_string(ids, tokenizer):

  new_ids = ids.copy()
  return [tokenizer.decode(id) for id in new_ids]
    
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
    # if base_loss < 1:
    #     try: 
    #         alpha = -(cer_score ** gamma) * torch.log(1 - cer_score + 1e-8)  # Add epsilon to avoid log(0)
    #         base_loss = base_loss * alpha  # Apply alpha weight to the loss
    #     except:
    #         base_loss = base_loss # Avoid cer_score is bigger than 1

    return base_loss

def step(model, tokenizer, data_loader, optimizer, criterion, device, cer, train=True):
    model.train()

    total_loss = 0
    total_cer = 0
    total_ignore_padding_cer = 0
    num_batches = len(data_loader)
    
    if isinstance(model, torch.nn.DataParallel):
        decoder_start_token_id = model.module.decoder.config.decoder_start_token_id
    else:
        decoder_start_token_id = model.decoder.config.decoder_start_token_id

    for i, batch in tqdm(enumerate(data_loader)):
        # Move batch to device
        if i == 50: break # For testing purposes, remove this line in production
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

        # Example: [CLS] 而 对 楼 市 成 交 抑 制 作 用 最 大 的 限 购 [SEP] [PAD] [PAD] [PAD]
        training_input_ids = batch['transcript_ids']  # Batch_size, seq_length # Training Data because it is right-shifted
        downsampled_features = batch['downsampled_features']  # Batch_size, seq_length, feature_dim
        attention_mask = batch['transcript_attention_mask']  # Batch_size, seq_length

        # Shifted Left input_ids for loss calculation
        # Ground-Truth data because it have to be left-shifted
        # Example: 而 对 楼 市 成 交 抑 制 作 用 最 大 的 限 购 [SEP] [PAD] [PAD] [PAD] [PAD]
        ground_truth_ids = _shift_left(training_input_ids, tokenizer.pad_token_id)  # Batch_size, seq_length 

        if i == 0: # First batch
            ground_truth_text = tokenizer.batch_decode(ground_truth_ids)
            training_input_text = tokenizer.batch_decode(training_input_ids)
            print(f"Ground truth text: {ground_truth_text}")
            print(f"Training input text: {training_input_text}\n")

        # Forward pass
        optimizer.zero_grad()
        outputs = model(training_input_ids, attention_mask, downsampled_features)  # Batch_size, seq_length, vocab_size

        # Compute CER and loss
        ids_prediction = outputs.argmax(dim=-1)  # Batch_size, seq_length
        predictions = tokenizer.batch_decode(ids_prediction, skip_special_tokens=True)  # Batch_size
        references = tokenizer.batch_decode(ground_truth_ids, skip_special_tokens=True)  # Batch_size
        cer_score = cer.compute(predictions=predictions, references=references)

        # Ignore padding tokens in the predictions and labels
        outputs_, labels_ = ignore_padding(ids_prediction, ground_truth_ids, padding_values=tokenizer.pad_token_id)
        predictions_ = convert_ids_to_string(outputs_, tokenizer)
        references_ = convert_ids_to_string(labels_, tokenizer)
        ignore_padding_cer_score = cer.compute(predictions=predictions_, references=references_)

        # Use the smaller CER for loss calculation
        print(f"Batch {i + 1}/{num_batches}")
        print(f"CER: {cer_score}, Ignore Padding CER: {ignore_padding_cer_score}")
        effective_cer_score = min(cer_score, ignore_padding_cer_score)
        print(f"Effective CER Score: {effective_cer_score}")
        loss = loss_fn(outputs, ground_truth_ids, effective_cer_score, criterion, ignore_index = tokenizer.pad_token_id)
        print(f"Loss: {loss}")
        
        if train:
            print(f"In Optimizing Process")
            loss.backward()
            optimizer.step()
            print(f"Out Optimizing Process\n")

        total_loss += loss.item()
        total_cer += cer_score
        total_ignore_padding_cer += ignore_padding_cer_score

        for ref, pred in zip(references[:5], predictions[:5]):  # Print first 5 examples in the batch
            print(f"Reference: {ref}")
            print(f"Prediction: {pred}")
            print("-" * 50)
        print('\n')

    mean_loss = total_loss / num_batches
    mean_cer = total_cer / num_batches
    mean_ignore_padding_cer = total_ignore_padding_cer / num_batches

    return {
        'mean_loss': mean_loss,
        'mean_cer': mean_cer,
        'mean_ignore_padding_cer': mean_ignore_padding_cer,
    }

def train(model, tokenizer, train_dataloader, optimizer, criterion, scheduler, epochs, cer):
    train_metrics_list = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        print("Training")
        train_metrics = step(
            model=model,
            tokenizer=tokenizer,
            data_loader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            cer=cer,
            train=True,
        )

        # scheduler.step()

        print(f"Train - Loss: {train_metrics['mean_loss']:.4f}, CER: {train_metrics['mean_cer']:.4f}, Ignore Padding CER: {train_metrics['mean_ignore_padding_cer']:.4f}")

        # Save metrics to lists
        train_metrics_list.append(train_metrics)

    return train_metrics_list

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

def generate_tokens(tokenizer, model, batch, cer, max_length=50):
    # Initialize variables
    model.eval()
    first_token = torch.tensor([tokenizer.cls_token_id], device=batch['downsampled_features'].device)  # Start with CLS token
    print(f'First_token: {first_token}')
    attention_mask = torch.tensor([1], device=batch['downsampled_features'].device)  # Start with attention mask for CLS token
    downsampled_features = batch['downsampled_features']
    generated_tokens = []

    for i in range(max_length):  # Limit the maximum generation length
        # Generate outputs from the model
        outputs = model(first_token.unsqueeze(dim=0), attention_mask.unsqueeze(dim=0), downsampled_features)
        predicted_tokens = outputs.argmax(dim=-1)  # Get the predicted token

        last_predicted_token = predicted_tokens[:, -1]
        print(f'The predicted token in step {i}: {last_predicted_token}')
        generated_tokens.append(last_predicted_token.item())

        # Check if the generated token is the SEP token (stop condition)
        if last_predicted_token.item() == tokenizer.sep_token_id:
            break

        # Update first_token and attention_mask for the next iteration
        first_token = torch.cat([first_token, last_predicted_token], dim=0)
        print(f'The tokens for the next generation: {first_token}\n')
        attention_mask = torch.cat([attention_mask, torch.tensor([1], device=attention_mask.device)], dim=0)

    # Decode predictions and references
    decoded_predictions = tokenizer.batch_decode(torch.tensor([generated_tokens]), skip_special_tokens=True)
    decoded_references = tokenizer.batch_decode(batch['transcript_ids'].tolist(), skip_special_tokens=True)

    # Compute CER score
    cer_score = cer.compute(predictions=decoded_predictions, references=decoded_references)

    return {
        'decoded_predictions': decoded_predictions,
        'decoded_references': decoded_references,
        'cer_score': cer_score,
    }

def teacher_forcing_generate_tokens(tokenizer, model, batch, cer, device):
    # Initialize variables
    model.eval()
    input_ids = batch['transcript_ids']
    attention_mask = batch['transcript_attention_mask']
    downsampled_features = batch['downsampled_features']
    shifted_left_outputs = torch.cat([input_ids[:, 1:], torch.full((input_ids.size(0), 1), tokenizer.pad_token_id, dtype=torch.long, device=device)], dim=1)

    # Forward pass
    outputs = model(input_ids, attention_mask, downsampled_features)

    # Decode predictions and references
    ids_prediction = outputs.argmax(dim=-1)
    decoded_predictions = tokenizer.batch_decode(ids_prediction, skip_special_tokens=True)
    decoded_references = tokenizer.batch_decode(shifted_left_outputs, skip_special_tokens=True)
    cer_score = cer.compute(predictions=decoded_predictions, references=decoded_references)

    return {
        'decoded_predictions': decoded_predictions,
        'decoded_references': decoded_references,
        'cer_score': cer_score,
    }
