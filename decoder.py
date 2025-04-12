import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder_layers import DecoderLayerA, DecoderLayerB, DecoderLayerC

class Decoder(nn.Module):
    def __init__(self, embed_size = 768, vocab_size = 21128, num_layers = 6, heads = 6, dim_feedforward = 2048, dropout = 0.1, mode = 'A', embedding_layer = None):
        super(Decoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.mode = mode

        # Layers
        self.embedding_layer = embedding_layer
        self.layers = self._layers_selection(mode, embed_size, num_layers, heads, dim_feedforward, dropout)
        self.output_prediction = nn.Linear(embed_size, self.vocab_size)

        # Params
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _layers_selection(self, mode, embed_size, num_layers, heads, dim_feedforward, dropout):
        if mode == 'A':
            return nn.ModuleList(
                [
                    DecoderLayerA(
                        embed_size,
                        heads,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout
                    )
                    for _ in range(num_layers)
                ]
            )
        elif mode == 'B':
            return nn.ModuleList(
                [
                    DecoderLayerB(
                        embed_size,
                        heads,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout
                    )
                    for _ in range(num_layers)
                ]
            )
        elif mode == 'C':
            return nn.ModuleList(
                [
                    DecoderLayerC(
                        embed_size,
                        heads,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown decoder mode: {mode}")
    
    def forward(self, input_ids, ce_outputs, ae_outputs, attention_mask):
        embedded_inputs = self.embedding_layer(input_ids).to(self.device)
        x = embedded_inputs
        for layer in self.layers:
            x = layer(shifted_right_outputs = x, 
                      ce_outputs = ce_outputs,
                      ae_outputs = ae_outputs, 
                      attn_mask = attention_mask)
            
        predictions = self.output_prediction(x)
        return predictions
