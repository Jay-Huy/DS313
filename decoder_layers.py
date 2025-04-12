import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils import MultiHeadedDotAttention, FFN

class DecoderLayerA(nn.Module):
    def __init__(self, embed_size, heads, dim_feedforward, dropout, casual = True):
        super(DecoderLayerA, self).__init__()
        self.masked_mha = MultiHeadedDotAttention(num_heads=heads, features_size=embed_size, casual = casual, dropout = dropout)  # Masked MHA
        self.cross_mha_1 = MultiHeadedDotAttention(num_heads=heads, features_size=embed_size, dropout = dropout)  # Cross MHA with CE outputs
        self.cross_mha_2 = MultiHeadedDotAttention(num_heads=heads, features_size=embed_size, dropout = dropout)  # Cross MHA with AE outputs

        self.ffn = FFN(features_size = embed_size, 
                       dim_feedforward = dim_feedforward,
                       dropout = dropout)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, shifted_right_outputs, ce_outputs, ae_outputs, attn_mask):
        # Masked MHA
        masked_outputs = self.masked_mha(query = shifted_right_outputs, 
                                         key = shifted_right_outputs, 
                                         value = shifted_right_outputs,
                                         att_mask = attn_mask)
        x1 = self.dropout(self.norm1(masked_outputs + shifted_right_outputs))

        # Cross MHA with AE outputs
        x2 = self.cross_mha_1(query = x1, 
                              key = ae_outputs, 
                              value = ae_outputs,
                              att_mask = attn_mask)
        x2 = self.dropout(self.norm2(x2 + x1))

        # Cross MHA with CE outputs
        x3 = self.cross_mha_2(query = x2, 
                              key = ce_outputs, 
                              value = ce_outputs,
                              att_mask = attn_mask)
        x3 = self.dropout(self.norm3(x3 + x2))

        # Feed Forward
        ffn_output = self.ffn(x3)
        return ffn_output
    
class DecoderLayerB(nn.Module):
    def __init__(self, embed_size, heads, dim_feedforward, dropout, casual = True):
        super(DecoderLayerB, self).__init__()
        self.masked_mha = MultiHeadedDotAttention(num_heads=heads, features_size=embed_size, casual = casual, dropout = dropout)  # Masked MHA
        self.cross_mha_1 = MultiHeadedDotAttention(num_heads=heads, features_size=embed_size, dropout = dropout)  # Cross MHA with CE outputs
        self.cross_mha_2 = MultiHeadedDotAttention(num_heads=heads, features_size=embed_size, dropout = dropout)  # Cross MHA with AE outputs

        self.ffn = FFN(features_size = embed_size, 
                       dim_feedforward = dim_feedforward,
                       dropout = dropout)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, shifted_right_outputs, ce_outputs, ae_outputs, attn_mask):
        # Masked MHA
        masked_outputs = self.masked_mha(query = shifted_right_outputs, 
                                         key = shifted_right_outputs, 
                                         value = shifted_right_outputs,
                                         att_mask = attn_mask)
        x1 = self.dropout(self.norm1(masked_outputs + shifted_right_outputs))

        # Cross MHA with CE outputs
        x2 = self.cross_mha_1(query = x1, 
                              key = ce_outputs, 
                              value = ce_outputs,
                              att_mask = attn_mask)
        x2 = self.dropout(self.norm2(x2 + x1))

        # Cross MHA with AE outputs
        x3 = self.cross_mha_2(query = x2, 
                              key = ae_outputs, 
                              value = ae_outputs,
                              att_mask = attn_mask)
        x3 = self.dropout(self.norm3(x3 + x2))

        # Feed Forward
        ffn_output = self.ffn(x3)
        return ffn_output
    
class DecoderLayerC(nn.Module):
    def __init__(self, embed_size, heads, dim_feedforward, dropout, casual = True):
        super(DecoderLayerC, self).__init__()
        self.masked_mha = MultiHeadedDotAttention(num_heads=heads, features_size=embed_size, casual = casual, dropout = dropout)  # Masked MHA
        self.cross_mha_1 = MultiHeadedDotAttention(num_heads=heads, features_size=embed_size, dropout = dropout)  # Cross MHA with AE outputs
        self.cross_mha_2 = MultiHeadedDotAttention(num_heads=heads, features_size=embed_size, dropout = dropout)  # Cross MHA with Contextual Encoder
        self.cross_mha_3 = MultiHeadedDotAttention(num_heads=heads, features_size=embed_size, dropout = dropout)  # Cross MHA with CE outputs

        self.ffn = FFN(features_size = embed_size, 
                       dim_feedforward = dim_feedforward,
                       dropout = dropout)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_size = embed_size
        self.dim_feedforward = dim_feedforward
        self.heads = heads
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

    def forward(self, shifted_right_outputs, ae_outputs, attn_mask, ce_outputs = None):
        # Masked MHA
        masked_outputs = self.masked_mha(query = shifted_right_outputs, 
                                         key = shifted_right_outputs, 
                                         value = shifted_right_outputs,
                                         att_mask = attn_mask)
        x1 = self.dropout(self.norm1(masked_outputs + shifted_right_outputs))

        # Cross MHA with AE outputs
        x2 = self.cross_mha_1(query = x1, 
                              key = ae_outputs, 
                              value = ae_outputs,
                              att_mask = attn_mask)
        x2 = self.dropout(self.norm2(x2 + x1))

        # Contextual Encoder
        _, seq_length = ae_outputs.shape[0], ae_outputs.shape[1]
        pos_encoding, _ = self.rope_positional_encoding(seq_length)

        position_vector = self.dropout(ae_outputs + pos_encoding[:seq_length, :]).to(self.device)

        x3 = self.cross_mha_2(query = position_vector, 
                              key = ae_outputs, 
                              value = ae_outputs,
                              att_mask = attn_mask)
        x3 = self.dropout(self.norm3(x3 + position_vector))

        # Cross MHA with CE outputs
        x4 = self.cross_mha_3(query = x2, 
                              key = x3, 
                              value = x3,
                              att_mask = attn_mask)
        x4 = self.dropout(self.norm2(x4 + x2))

        # Feed Forward
        ffn_output = self.ffn(x4)
        return ffn_output
