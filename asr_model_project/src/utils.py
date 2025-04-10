import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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
            scores = scores.masked_fill(att_mask.transpose(-2, -1) == 0, float('-inf'))  # Mask columns

        # Apply causal mask if enabled
        if self.casual:
            seq_len = scores.size(-1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device)).bool()
            scores = scores.masked_fill(~causal_mask, float('-inf'))
            
        p_attn = F.softmax(scores, dim=-1)
        p_attn = dropout(p_attn)

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
