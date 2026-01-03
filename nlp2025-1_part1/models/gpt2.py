import torch
from torch import nn
from transformers import GPT2Model as OpenAIGPT2Model

from config import GPT2Config
from models.base_gpt import GPTPreTrainedModel
from modules.gpt2_layer import GPT2Layer
from utils import get_extended_attention_mask


class GPT2Model(GPTPreTrainedModel):
  """
  GPT 모델은 문장 내 각 토큰에 대한 최종 임베딩을 반환한다.

  모델 구성은 다음과 같다:
  1. 임베딩 층 (self.embed 에서 사용).
  2. n 개의 GPT 층의 적층 (self.encode 에서 사용).
  3. [CLS] 토큰에 대한 선형변환 층(self.forward 에서 그대로 사용).
  """

  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

    # (1, position_임베딩_길이)의 position_ids는 학습되지 않는 상수이므로 버퍼에 저장해둔다.
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # GPT-2 layers.
    self.gpt_layers = nn.ModuleList([GPT2Layer(config) for _ in range(config.num_hidden_layers)])

    # [CLS] 토큰 변환.
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    # Final layer norm.
    self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # 입력 토큰 임베딩
    inputs_embeds = self.word_embedding(input_ids)

    # 위치 임베딩 가져오기
    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)

    # 두 임베딩을 더한 후 dropout 적용
    embeddings = inputs_embeds + pos_embeds
    embeddings = self.embed_dropout(embeddings)

    return embeddings



    ### TODO: pos_ids를 사용하여 self.pos_embedding에서 위치 임베딩을 가져와 pos_embeds에 저장한다.
    ###       그런 다음, 두 개의 임베딩을 더하고, 드롭아웃을 적용한 뒤 반환한다.
    ### 완성시켜야 할 빈 코드 블록

  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: 임베딩 층으로부터의 출력 [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # self-attention을 위한 extended attention mask를 구한다.
    # 크기 [batch_size, 1, 1, seq_len]인 extended_attention_mask를 반환.
    # (0 값이 포함된) non-padding token과 (큰 음수들로 된) padding token을 구별할 것.
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # encoder 층을 통해 hidden states 전달.
    for i, layer_module in enumerate(self.gpt_layers):
      # 마지막 bert_layer에서 인코딩를 가져다가 다음 층에 주입.
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len은 batch의 최대 길이
    attention_mask: input_ids 와 크기가 같으며, 1 은 non-padding token을, 0 은 padding token을 나타낸다.  
    """
    # 각 입렵 토큰에 대한 임베딩 구하기기
    embedding_output = self.embed(input_ids=input_ids)

    # GPYLayers의 stack인 트랜스포머에 주입.
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
    sequence_output = self.final_layer_norm(sequence_output)

    # 마지막 토큰의 hidden state 구하기.
    last_non_pad_idx = attention_mask.sum(dim=1) - 1  # 마지막 인덱스를 구하려면 1을 뺀다.
    last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]

    return {'last_hidden_state': sequence_output, 'last_token': last_token}

  def hidden_state_to_token(self, hidden_state):
    # weight tying: 단어 임베딩 가중치와 출력 로짓 가중치 공유
    logits = torch.matmul(hidden_state, self.word_embedding.weight.T)
    return logits

  @classmethod
  def from_pretrained(cls, model='gpt2', d=768, l=12, num_heads=12):
    gpt_model = OpenAIGPT2Model.from_pretrained(model).eval()
    our_model = GPT2Model(GPT2Config(hidden_size=d, num_hidden_layers=l,num_attention_heads=num_heads,
                                     intermediate_size=d*3)).eval()

    # Load word and positional embeddings.
    our_model.word_embedding.load_state_dict(gpt_model.wte.state_dict())
    our_model.pos_embedding.load_state_dict(gpt_model.wpe.state_dict())

    for i in range(l):
      l = our_model.gpt_layers[i]
      # Q, K, V 가중치를 conv1d에서 3개의 선형 프로젝션으로 재매핑.
      l.self_attention.query.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, :d].T
      l.self_attention.query.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][:d]
      l.self_attention.key.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, d:d*2].T
      l.self_attention.key.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][d:d*2]
      l.self_attention.value.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, d*2:].T
      l.self_attention.value.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][d*2:]

      # MHA의 마지막 dense layer를 재매핑.
      l.attention_dense.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_proj.weight'].T
      l.attention_dense.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_proj.bias']

      # Attention layer norm을 재매핑.
      l.attention_layer_norm.weight.data = gpt_model.state_dict()[f'h.{i}.ln_1.weight']
      l.attention_layer_norm.bias.data = gpt_model.state_dict()[f'h.{i}.ln_1.bias']

      # Post-attention MLP layer들을 재매핑
      l.interm_dense.weight.data = gpt_model.state_dict()[f'h.{i}.mlp.c_fc.weight'].T
      l.interm_dense.bias.data = gpt_model.state_dict()[f'h.{i}.mlp.c_fc.bias']
      l.out_dense.weight.data = gpt_model.state_dict()[f'h.{i}.mlp.c_proj.weight'].T
      l.out_dense.bias.data = gpt_model.state_dict()[f'h.{i}.mlp.c_proj.bias']

      # 두번째 layer norm weights를 재매핑.
      l.out_layer_norm.weight.data = gpt_model.state_dict()[f'h.{i}.ln_2.weight']
      l.out_layer_norm.bias.data = gpt_model.state_dict()[f'h.{i}.ln_2.bias']

    # 마지막 layer norm 값들을 재매핑.
    our_model.final_layer_norm.weight.data = gpt_model.state_dict()['ln_f.weight']
    our_model.final_layer_norm.bias.data = gpt_model.state_dict()['ln_f.bias']

    return our_model
