import torch.nn as nn
import torch.nn.functional as F
from contextual_encoder import Contextual_Encoder
from decoder import Decoder
from transformers import T5ForConditionalGeneration
import torch

class ASRModel(nn.Module):
    def __init__(self, model_dim = 768, mode = 'A' , layer_selection_mode = 'last6'):
        super(ASRModel, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        # Load the pre-trained embedding layer from the Chinese BERT model uer/t5-base-chinese-cluecorpussmall
        # t5_module = T5ForConditionalGeneration.from_pretrained("Langboat/mengzi-t5-base").to(device)
        t5_module = T5ForConditionalGeneration.from_pretrained("uer/t5-base-chinese-cluecorpussmall").to(device)
        
        decoder_module = t5_module.decoder
        self.vocab_size = decoder_module.config.vocab_size
        self.config = t5_module.config
        self.decoder = Decoder(mode=mode, 
                               layer_selection_mode = layer_selection_mode,
                               decoder_module=decoder_module,
                               config = self.config)
        
        self.lm_head = t5_module.lm_head

        # Parameters
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.mode = mode
        self.layer_selection_mode = layer_selection_mode
        self._init_weights()

    def _init_weights(self):
        """
        Apply Xavier initialization to all linear layers and attention weights.
        """
        for module in self.acoustic_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.TransformerEncoderLayer):
                # Initialize weights in TransformerEncoderLayer
                for param in module.parameters():
                    if param.dim() > 1:  # Only initialize weight matrices
                        nn.init.xavier_uniform_(param)
        
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
        if self.mode == 'A':
            first_encoder_hidden_states = ae_outputs
            second_encoder_hidden_states = ce_outputs

        elif self.mode == 'B':
            first_encoder_hidden_states = ce_outputs
            second_encoder_hidden_states = ae_outputs

        output = self.decoder(input_ids = input_ids, 
                            first_encoder_hidden_states = first_encoder_hidden_states,
                            second_encoder_hidden_states = second_encoder_hidden_states,
                            attention_mask = attention_mask)
        prediction = self.lm_head(output['last_hidden_state']) # (B, Sequence-Length, vocab_size)
        return prediction
