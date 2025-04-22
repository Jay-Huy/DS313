import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MultiHeadedDotAttention
import math

class Decoder(nn.Module):
    def __init__(self, mode = 'A', layer_selection_mode = 'last6', decoder_module = None, config = None, _update_causal_mask=None, invert_attention_mask=None, get_head_mask=None):
        super(Decoder, self).__init__()
        
        """
        Args:
            mode (str): Mode of the decoder.
            decoder_module (nn.Module): The full decoder module (e.g., T5 decoder).
            config (Config): Configuration object for the decoder.
            layer_selection_mode (str): Determines which layers to use. Options:
                - "last6": Use the last 6 layers.
                - "first3_last3": Use the first 3 and last 3 layers.
        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        self.mode = mode
        self.layer_selection_mode = layer_selection_mode

        # Layers
        self.embed_tokens = decoder_module.embed_tokens
        # Select layers based on the layer_selection_mode
        if self.layer_selection_mode == "last6":
            self.block = decoder_module.block[-6:]  # Use the last 6 layers
        elif self.layer_selection_mode == "first3_last3":
            self.block = decoder_module.block[:3] + decoder_module.block[-3:]  # Use the first 3 and last 3 layers
        else:
            raise ValueError(f"Invalid layer_selection_mode: {self.layer_selection_mode}. Choose 'last6' or 'first3_last3'.")
        
        self.final_layer_norm = decoder_module.final_layer_norm

        # Params
        self.dropout = nn.Dropout(0.1)
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Others
        self.is_decoder = True
        self.model_parallel = True if torch.cuda.device_count() > 1 else False
        self.first_device = torch.device("cuda:0") if self.model_parallel else None
        self.config = config
        self.config.is_decoder = True
        self.gradient_checkpointing = False
        self.embed_size = self.config.d_model

        self._update_causal_mask = decoder_module._update_causal_mask
        self.invert_attention_mask = decoder_module.invert_attention_mask
        self.get_head_mask = decoder_module.get_head_mask

        if self.mode == 'C':
            self.multihead_attention_layers = nn.ModuleList([
                MultiHeadedDotAttention(num_heads=6, features_size=self.config.d_model) for _ in range(len(self.block))
            ])

    def rope_positional_encoding(self, seq_length): # This function will be used for C structure
        """
        Implements 2D Relative and Absolute Positional Encoding (2DRoPE).
        """
        position = torch.arange(0, seq_length, dtype=torch.float32, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_size, 2, dtype=torch.float32, device=self.device) * 
                            -(math.log(10000.0) / self.embed_size))
        
        # Absolute positional encoding
        pos_encoding = torch.zeros(seq_length, self.embed_size, device=self.device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        # Relative positional encoding
        relative_positions = torch.arange(-seq_length + 1, seq_length, device=self.device).unsqueeze(0)
        relative_encoding = torch.zeros(seq_length, seq_length, self.embed_size, device=self.device)
        
        # Efficient computation of relative encoding
        for i in range(seq_length):
            relative_encoding[i, :, 0::2] = torch.sin((position - i) * div_term)
            relative_encoding[i, :, 1::2] = torch.cos((position - i) * div_term)
        
        return pos_encoding, relative_encoding

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        first_encoder_hidden_states=None,
        first_encoder_attention_mask=None,
        second_encoder_hidden_states=None,
        second_encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cache_position=None,
    ):
        # Initialize cache_position
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if self.gradient_checkpointing and self.training:
            use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )
        if self.is_decoder:
            if attention_mask is not None:
                # Expand attention_mask to 4D: [batch_size, 1, 1, seq_length]
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min  # Invert mask
            else:
                attention_mask = None

            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values.self_attention_cache if past_key_values is not None else None,
                output_attentions,
            )
        else:
            causal_mask = None

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder:
            if first_encoder_hidden_states is not None:
                first_encoder_batch_size, first_encoder_sequence_length, _ = first_encoder_hidden_states.size()
                first_encoder_hidden_shape = (first_encoder_batch_size, first_encoder_sequence_length)

                first_encoder_attention_mask = torch.ones(
                    first_encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
                first_encoder_extended_attention_mask = self.invert_attention_mask(first_encoder_attention_mask)
            else:
                first_encoder_extended_attention_mask = None

            if second_encoder_hidden_states is not None:
                second_encoder_batch_size, second_encoder_sequence_length, _ = second_encoder_hidden_states.size()
                second_encoder_hidden_shape = (second_encoder_batch_size, second_encoder_sequence_length)

                second_encoder_attention_mask = torch.ones(
                    second_encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
                second_encoder_extended_attention_mask = self.invert_attention_mask(second_encoder_attention_mask)
            else:
                second_encoder_extended_attention_mask = None
        else:
            first_encoder_extended_attention_mask, second_encoder_extended_attention_mask = None, None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        position_bias = None
        first_encoder_decoder_position_bias = None
        second_encoder_decoder_position_bias = None

        # Embedding Layer
        hidden_states = self.dropout(inputs_embeds)

        # Process through decoder layers
        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if causal_mask is not None:
                    causal_mask = causal_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                
                # All First Encoder and Second Encoder related tensors should be on the same device as hidden_states
                if first_encoder_hidden_states is not None:
                    first_encoder_hidden_states = first_encoder_hidden_states.to(hidden_states.device)
                if second_encoder_hidden_states is not None:
                    second_encoder_hidden_states = second_encoder_hidden_states.to(hidden_states.device)

                if first_encoder_extended_attention_mask is not None:
                    first_encoder_extended_attention_mask = first_encoder_extended_attention_mask.to(hidden_states.device)

                if second_encoder_extended_attention_mask is not None:
                    second_encoder_extended_attention_mask = second_encoder_extended_attention_mask.to(hidden_states.device)

                if first_encoder_decoder_position_bias is not None:
                    first_encoder_decoder_position_bias = first_encoder_decoder_position_bias.to(hidden_states.device)

                if second_encoder_decoder_position_bias is not None:
                    second_encoder_decoder_position_bias = second_encoder_decoder_position_bias.to(hidden_states.device)

                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Process the layer
            layer_outputs = self.process_layer(
                layer_module,
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                first_encoder_hidden_states=first_encoder_hidden_states,
                first_encoder_attention_mask=first_encoder_extended_attention_mask,
                first_encoder_decoder_position_bias=first_encoder_decoder_position_bias,
                second_encoder_hidden_states=second_encoder_hidden_states,
                second_encoder_attention_mask=second_encoder_extended_attention_mask,
                second_encoder_decoder_position_bias=second_encoder_decoder_position_bias,
                head_mask=head_mask[i] if head_mask is not None else None,
                cross_attn_head_mask=cross_attn_head_mask[i] if cross_attn_head_mask is not None else None,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                layer_index=i,
            )

            # Update position biases
            position_bias = layer_outputs[2]
            first_encoder_decoder_position_bias = layer_outputs[3]
            second_encoder_decoder_position_bias = layer_outputs[4]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[5],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[6],)

        # Final Layer Normalization and Output
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "cross_attentions": all_cross_attentions,
        }
    
    def process_layer(
        self,
        layer_module,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        first_encoder_hidden_states=None,
        first_encoder_attention_mask=None,
        first_encoder_decoder_position_bias=None,
        second_encoder_hidden_states=None,
        second_encoder_attention_mask=None,
        second_encoder_decoder_position_bias=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
        layer_index=None, # Added for mode 'C'
    ):
        # Self-Attention
        self_attention_outputs = layer_module.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states, past_key_value = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # Cross-Attention for First Encoder
        if first_encoder_hidden_states is not None:
            cross_attention_outputs = layer_module.layer[1](
                hidden_states,
                key_value_states=first_encoder_hidden_states,
                attention_mask=first_encoder_attention_mask,
                position_bias=first_encoder_decoder_position_bias,
                layer_head_mask=cross_attn_head_mask,
                past_key_value=past_key_value,
                query_length=cache_position[-1] + 1 if cache_position is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states, past_key_value = cross_attention_outputs[:2]
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

            # Update position bias for the first encoder
            first_encoder_decoder_position_bias = cross_attention_outputs[2]

        # Cross-Attention for Second Encoder
        if second_encoder_hidden_states is not None:
            cross_attention_outputs = layer_module.layer[1](  # Reuse the same layer
                hidden_states,
                key_value_states=second_encoder_hidden_states,
                attention_mask=second_encoder_attention_mask,
                position_bias=second_encoder_decoder_position_bias,
                layer_head_mask=cross_attn_head_mask,
                past_key_value=past_key_value,
                query_length=cache_position[-1] + 1 if cache_position is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states, past_key_value = cross_attention_outputs[:2]
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

            # Update position bias for the second encoder
            second_encoder_decoder_position_bias = cross_attention_outputs[2]
        else:
            _, seq_length = first_encoder_hidden_states.size(0), first_encoder_hidden_states.size(1)
            pos_encoding, _ = self.rope_positional_encoding(seq_length)
            position_vector = self.dropout(first_encoder_hidden_states + pos_encoding[:seq_length, :]).to(self.device)
            
            second_encoder_hidden_states = self.multihead_attention_layers[layer_index](
                query=position_vector,
                key=first_encoder_hidden_states,
                value=first_encoder_hidden_states,
            )
            cross_attention_outputs = layer_module.layer[1](  # Reuse the same layer
                hidden_states,
                key_value_states=second_encoder_hidden_states,
                attention_mask=second_encoder_attention_mask,
                position_bias=second_encoder_decoder_position_bias,
                layer_head_mask=cross_attn_head_mask,
                past_key_value=past_key_value,
                query_length=cache_position[-1] + 1 if cache_position is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states, past_key_value = cross_attention_outputs[:2]
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

            # Update position bias for the second encoder
            second_encoder_decoder_position_bias = cross_attention_outputs[2]
        # Feed Forward
        hidden_states = layer_module.layer[-1](hidden_states)

        return (
            hidden_states,
            past_key_value,
            position_bias,
            first_encoder_decoder_position_bias,
            second_encoder_decoder_position_bias,
        ) + attention_outputs
