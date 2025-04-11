import torch.nn as nn
import torch.nn.functional as F
from src.contextual_encoder import Contextual_Encoder
from src.decoder import Decoder
from transformers import AutoModel
import torch

class ASRModel(nn.Module):
    def __init__(self, model_dim = 768, mode = 'A', ):
        super(ASRModel, self).__init__()
        # Acoustic Encoder
        self.projector = nn.Sequential(
            nn.Linear(10240, model_dim),  # Reduce dimensionality
            nn.ReLU(),              # Add non-linearity
            nn.Dropout(0.1),        # Add dropout for regularization
        )
        self.acoustic_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=6), num_layers=6
        )

        # Contextual Encoder
        if mode != 'C': # If the model uses Structure A or B, include the contextual encoder
            self.contextual_encoder = Contextual_Encoder(embed_size=model_dim, num_layers=4, heads=6, dim_feedforward=2048, dropout=0.1)
        else: # If the model uses Structure C
            self.contextual_encoder = None
        
        # Decoder
        # Load the pre-trained embedding layer from the Chinese BERT model
        embedding_layer = AutoModel.from_pretrained("bert-base-chinese").embeddings
        self.vocab_size = embedding_layer.word_embeddings.weight.shape[0] # 21128
        self.decoder = Decoder(embed_size=model_dim, 
                               vocab_size=self.vocab_size, 
                               num_layers=6, heads=6, 
                               dim_feedforward=2048, 
                               dropout=0.1, 
                               mode=mode, 
                               embedding_layer=embedding_layer)

        # Parameters
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, input_ids, attention_mask, audio_features):
        """
        Args:
            acoustic_input: Input tensor for the acoustic encoder ((B, C, T, F) or (B, T, C*F))
            contextual_input: Input tensor for the contextual encoder
            src_key_padding_mask: Optional mask for padding (batch_size, seq_len)
        Returns:
            Output from the decoder
        """

        # Acoustic Encoder
        audio_features = self.projector(audio_features) # Project the audio features to the model dimension
        ae_outputs = self.acoustic_encoder(audio_features) # (B, T, 768)
        
        # Contextual Encoder
        if self.contextual_encoder is not None: # If the model uses Structure A or B
            ce_outputs = self.contextual_encoder(ae_outputs) # (B, T, 768)
        else:
            ce_outputs = None # If the model uses Structure C, skip the contextual encoder

        output = self.decoder(input_ids = input_ids, 
                              ce_outputs = ce_outputs,
                              ae_outputs = ae_outputs,
                              attention_mask = attention_mask)
        return output 
