
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils import MultiHeadedDotAttention, FFN

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dim_feedforward, dropout):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadedDotAttention(num_heads=heads, features_size=embed_size)
        # self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.feed_forward = FFN(features_size = embed_size, 
                                dim_feedforward = dim_feedforward, 
                                dropout=dropout)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attention_mask = None):
        # Multi-Head Attention
        attention_output = self.multihead_attention(query, key, value, att_mask=attention_mask)
        x = self.dropout(self.norm1(attention_output + query))
        
        # Feed Forward
        forward_output = self.feed_forward(x)
        return forward_output

class Contextual_Encoder(nn.Module):
    def __init__(self, embed_size = 768, num_layers = 4, heads = 6, dim_feedforward = 2048, dropout = 0.1):
        super(Contextual_Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_size,
                    heads,
                    dropout=dropout,
                    dim_feedforward=dim_feedforward
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def rope_positional_encoding(self, seq_length):
        """
        Implements 2D Relative and Absolute Positional Encoding (2DRopE).
        """
        position = torch.arange(0, seq_length, dtype=torch.float32, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_size, 2, dtype=torch.float32, device=self.device) * 
                            -(math.log(10000.0) / self.embed_size))
        
        # Absolute positional encoding
        pos_encoding = torch.zeros(seq_length, self.embed_size, device=self.device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        # Relative positional encoding (example: pairwise distances)
        relative_positions = torch.arange(-seq_length + 1, seq_length, device=self.device)
        relative_encoding = torch.zeros(seq_length, seq_length, self.embed_size, device=self.device)
        for i in range(seq_length):
            for j in range(seq_length):
                relative_encoding[i, j] = pos_encoding[abs(relative_positions[i - j])]
        
        return pos_encoding, relative_encoding

    def forward(self, ae_outputs, attention_mask = None):
        ae_outputs = ae_outputs.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        _, seq_length = ae_outputs.shape[0], ae_outputs.shape[1]
        pos_encoding, _ = self.rope_positional_encoding(seq_length)

        position_vector = self.dropout(ae_outputs + pos_encoding[:seq_length, :]).to(self.device)
        encoded_features = self.layers[0](query = position_vector, 
                                          key = ae_outputs,
                                          value = ae_outputs, 
                                          attention_mask = attention_mask)
        for layer in self.layers[1:]:
            encoded_features = layer(query = encoded_features, 
                                     key = encoded_features,
                                     value = encoded_features, 
                                     attention_mask = attention_mask)

        return encoded_features
