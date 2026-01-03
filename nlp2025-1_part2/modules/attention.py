import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # key, value, query에 대한 선형변환 layer 초기화.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    # 이 드롭아웃은 트랜스포머의 원래 구현에 따라 normalized attention scores에 적용된다.
    # 다소 이례적이지만, 경험적으로 이것이 더 나은 성능을 제공한다고 알려져 있다.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # hidden_state (x) 를 사영하기 위해 k, v, q의 해당 linear_layer가 사용된다.
    proj = linear_layer(x)
    # 다음으로, 프로젝션에 대해 여러 헤드를 생성해야 한다. 
    # 이는 은닉 상태를 self.num_attention_heads로 분할하며, 
    # 각 헤드는 self.attention_head_size 크기를 갖도록 한다.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # 적절히 전치하여 크기 [bs, num_attention_heads, seq_len, attention_head_size]인 프로젝션을 얻는다.
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    # query: [bs, heads, seq_len, head_dim]
    # key:   [bs, heads, seq_len, head_dim]
    # value: [bs, heads, seq_len, head_dim]

    # 1. Scaled dot-product attention score
    attn_scores = torch.matmul(query, key.transpose(-1, -2))  # [bs, heads, seq_len, seq_len]
    attn_scores = attn_scores / (self.attention_head_size ** 0.5)

    # 2. Attention mask (causal + padding)
    #    attention_mask: [bs, 1, 1, seq_len], 값이 0인 위치에 큰 음수를 더해 softmax 후 0되게
    attn_scores = attn_scores + attention_mask

    # 3. Softmax로 attention weights 생성
    attn_probs = nn.functional.softmax(attn_scores, dim=-1)
    attn_probs = self.dropout(attn_probs)  # 드롭아웃 적용

    # 4. attention weights를 value에 곱해서 최종 attention 출력값 계산
    attn_output = torch.matmul(attn_probs, value)  # [bs, heads, seq_len, head_dim]

    # 5. 다시 head들을 concat (b, h, t, d) → (b, t, h*d)
    attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')

    return attn_output



  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # 먼저, self.transform을 사용하여 multi-head attention에 필요한
    # 각 토큰의 key, value, query를 생성해야 한다(함수 내부에 자세한 내용 있음).
    # *_layer의 크기 = [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # multi-head attention 계산.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
