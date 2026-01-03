from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    out = dense_layer(output)
    out = dropout(out)
    return input + out  # residual connection

  def forward(self, hidden_states, attention_mask):
    # Attention block
    normed_hidden_states = self.attention_layer_norm(hidden_states)
    attention_output = self.self_attention(normed_hidden_states, attention_mask)
    attention_output = self.add(hidden_states, attention_output, self.attention_dense, self.attention_dropout)

    # Feed-forward block
    normed_attention_output = self.out_layer_norm(attention_output)
    interm_output = self.interm_af(self.interm_dense(normed_attention_output))
    output = self.add(attention_output, interm_output, self.out_dense, self.out_dropout)

    return output
