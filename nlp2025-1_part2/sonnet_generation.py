import argparse      # 실행 인자 파싱

# 랜덤 및 수치 계산 관련
import random
import torch
import numpy as np

# PyTorch 관련
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


from tqdm import tqdm                       # 진행률 표시
from transformers import GPT2Tokenizer      # Huggingface GPT2 tokenizer
from einops import rearrange                # Tensor 차원 재구성    

# 소넷 데이터셋 로드
from datasets import (
  SonnetsDataset,
)

# 사용자 정의 GPT2 모델 및 옵티마이저
from models.gpt2 import GPT2Model
from optimizer import AdamW

# 운율 분석, 문자 처리 등
import re
from collections import Counter
import pronouncing      # 발음 기반 운율 판단을 위한 라이브러리
import bert_score       # BERT 기반 텍스트 유사도 평가
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # 자연어 처리 BLEU 점수 계산용 함수

TQDM_DISABLE = False        # tqdm 출력 여부 제어



# 1. Perplexity: 모델의 언어 생성 능력 평가
def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for batch in dataloader:
        # 배치 데이터 → GPU
        b_ids = batch['token_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            # 모델 예측
            logits = model(b_ids, b_mask)
            logits = logits[:, :-1, :].contiguous()     # 마지막 토큰은 예측 대상 제외
            labels = b_ids[:, 1:].contiguous()          # 입력의 다음 토큰이 정답 레이블
            # Cross entropy 손실 (sum 모드)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += labels.numel()
    # 평균 loss를 기반으로 perplexity 계산        
    ppl = torch.exp(torch.tensor(total_loss / total_tokens))
    return ppl.item()



# 2. BLEU / BERTScore 계산 함수
def compute_text_similarity_scores(preds, refs, tokenizer):

    smoothie = SmoothingFunction().method4      # BLEU 스무딩 함수


    # tokenizer를 사용한 간단한 토큰화 함수 정의
    def simple_tokenize(text):
        return tokenizer.tokenize(text)
    
    # 예측/정답 개수가 다를 경우, 최소 길이로 절단
    if len(preds) != len(refs):
        print(f"[Warning] preds({len(preds)}) != refs({len(refs)}). Truncating to min length.")
        min_len = min(len(preds), len(refs))
        preds = preds[:min_len]
        refs = refs[:min_len]

    # 각 샘플에 대해 BLEU 계산
    bleu_scores = [sentence_bleu([simple_tokenize(ref)], simple_tokenize(pred), smoothing_function=smoothie)
                   for pred, ref in zip(preds, refs)]
    bleu_avg = sum(bleu_scores) / len(bleu_scores)

    # BERTScore 계산
    _, _, F1 = bert_score.score(preds, refs, lang="en", verbose=False)
    bert_f1_avg = F1.mean().item()

    return bleu_avg, bert_f1_avg



# 3. Rhyming Accuracy 평가 (각 행 마지막 단어 기준으로 운율 평가)
def compute_rhyme_accuracy(preds):

    # 문장에서 마지막 단어를 추출하는 함수
    def get_last_word(line):
        words = re.findall(r"\b\w+\b", line)
        return words[-1].lower() if words else ''

    # 두 단어가 운율(rhyme)이 맞는지 판단하는 함수
    def rhymes(word1, word2):
        phones1 = pronouncing.phones_for_word(word1)
        phones2 = pronouncing.phones_for_word(word2)
        if not phones1 or not phones2:
            return False
        rhyme1 = pronouncing.rhyming_part(phones1[0])
        rhyme2 = pronouncing.rhyming_part(phones2[0])
        return rhyme1 == rhyme2

    rhyme_correct = 0
    total_pairs = 0

    # 각 소넷에 대해 평가
    for sonnet in preds:
        lines = [l.strip() for l in sonnet.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            continue
        
        last_words = [get_last_word(line) for line in lines]

        # 전통적 소넷의 운율 구조에 맞는 쌍들
        rhyme_pairs = [(0,2), (1,3), (4,6), (5,7), (8,10), (9,11), (12,13)]
        for i, j in rhyme_pairs:
            if i >= len(last_words) or j >= len(last_words):
                continue  # 줄 수가 부족한 경우 해당 쌍을 건너뜀
            if rhymes(last_words[i], last_words[j]):
                rhyme_correct += 1
            total_pairs += 1

    # 정확도 = rhyme이 맞은 쌍 / 전체 쌍
    return rhyme_correct / total_pairs if total_pairs > 0 else 0.0



# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True



# GPT-2 기반 소넷 생성 모델 정의
class SonnetGPT(nn.Module):

  def __init__(self, args):

    super().__init__()
    # Hugging Face GPT2 모델을 사용자 지정 hidden dim, layer 수 등으로 로드
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)

    # HuggingFace GPT2 tokenizer 로드
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # 전체 파라미터를 fine-tuning 하도록 설정
    for param in self.gpt.parameters():
      param.requires_grad = True

    # dropout layer 추가 (과적합 방지)
    self.dropout = nn.Dropout(p=0.2) 


  # 순전파 함수 (forward pass)
  def forward(self, input_ids, attention_mask):

    # GPT-2 모델의 출력(hidden states) 얻기
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs["last_hidden_state"]  # [batch_size, seq_len, hidden_dim]
    hidden_states = self.dropout(hidden_states)   # dropout 적용

    # weight tying: embedding 가중치를 projection에도 사용
    embedding_weight = self.gpt.word_embedding.weight  # [vocab_size, hidden_dim]
    logits = torch.matmul(hidden_states, embedding_weight.T)  # [batch_size, seq_len, vocab_size]

    return logits


  # 모델의 device 반환 (GPU/CPU)
  def get_device(self):
    for param in self.gpt.parameters():
      return param.device
    

  # 소넷 생성 함수
  @torch.no_grad()
  def generate(self, encoding, temperature=0.9, top_k=30, beam_width=5, max_length=112, min_length=80, use_beam=False):

    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    # Beam Search 방식
    if use_beam:
        sequences = [(token_ids, 0.0)]    # 초기 시퀀스와 log 확률
        for _ in range(max_length):
            all_candidates = []
            for seq, score in sequences:
                logits = self.forward(seq, torch.ones_like(seq))[:, -1, :]
                probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_indices = probs.topk(beam_width)
                for i in range(beam_width):
                    next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                    next_seq = torch.cat([seq, next_token], dim=1)
                    next_score = score + topk_probs[0, i].item()
                    all_candidates.append((next_seq, next_score))
            # 확률 높은 상위 beam_width 개 선택
            sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]
            # eos 토큰 등장 & 최소 길이 조건 충족 시 종료
            if sequences[0][0][0, -1].item() == self.tokenizer.eos_token_id and sequences[0][0].shape[1] >= min_length:
                break
        token_ids = sequences[0][0]

    # Top-k sampling 방식    
    else:
        for _ in range(max_length):
            logits = self.forward(token_ids, attention_mask)
            logits_last_token = logits[:, -1, :] / temperature    # softmax 온도 조절
            topk_logits, topk_indices = torch.topk(logits_last_token, top_k, dim=-1)
            probs = F.softmax(topk_logits, dim=-1)
            sampled_index = torch.multinomial(probs, 1)
            sampled_token = topk_indices.gather(dim=-1, index=sampled_index)

            # 길이 제한 확인
            if token_ids.shape[1] >= max_length:
                break
            
             # 토큰 추가
            token_ids = torch.cat([token_ids, sampled_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1)

            # 줄 수가 14줄을 넘으면 종료 (전통 소넷 구조 반영)
            generated_text = self.tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)
            num_lines = generated_text.count('\n')
            if num_lines >= 14:
                break
            
            # 최대 길이 도달 시 종료
            if sampled_token.item() == self.tokenizer.eos_token_id:
                if token_ids.shape[1] >= min_length:
                    break
                else:
                    continue  # 최소 길이까지 eos 무시

    # 최종 생성 텍스트 디코딩
    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist(), skip_special_tokens=True)
    lines = [line.strip() for line in generated_output.splitlines() if line.strip()]
    generated_output = '\n'.join(lines)
    return token_ids, generated_output



# validation 성능 개선 없을 시 학습 조기 종료 (overfitting 방지용)
class EarlyStopping:
    def __init__(self, patience=3, delta=0.0):
        self.patience = patience        # 개선 없을 때 허용할 epoch 수
        self.delta = delta              # 개선 판단 시 기준 값
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None or val_score > self.best_score + self.delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True



# 훈련 중 rhyme 정확도를 높이기 위한 보조 손실 함수
def compute_rhyme_loss(pred_texts):

    rhyme_loss = 0
    count = 0

    for text in pred_texts:
        # 각 줄 추출 및 마지막 단어 수집
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        if len(lines) < 8:
            continue
        last_words = [re.findall(r"\b\w+\b", line)[-1].lower() if re.findall(r"\b\w+\b", line) else '' for line in lines[:14]]
        rhyme_pairs = [(0,2), (1,3), (4,6), (5,7)]

        # 각 rhyme 쌍 평가
        for i, j in rhyme_pairs:
            if last_words[i] and last_words[j]:
                phones_i = pronouncing.phones_for_word(last_words[i])
                phones_j = pronouncing.phones_for_word(last_words[j])
                if phones_i and phones_j:
                    rhyme_i = pronouncing.rhyming_part(phones_i[0])
                    rhyme_j = pronouncing.rhyming_part(phones_j[0])
                    # 운율이 다르면 패널티 부여
                    if rhyme_i != rhyme_j:
                        rhyme_loss += 1.0
                    count += 1

    # 평균 rhyme loss 반환
    if count == 0:
        return torch.tensor(0.0)
    return torch.tensor(rhyme_loss / count).to(torch.device("cuda") if torch.backends.cuda.is_available() else "cpu")



# 모델을 학습시키는 메인 함수
def train(args):
  
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')    # 디바이스 설정

  # 학습 데이터셋 로드 및 DataLoader 생성
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  val_dataset = SonnetsDataset(args.held_out_sonnet_path)   # 검증용 held-out 데이터셋 로드
  args = add_arguments(args)    # 모델 크기에 따라 hidden dim, layer 수 등을 설정
  model = SonnetGPT(args).to(device)    # SonnetGPT 모델 생성 및 GPU로 이동
  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)  # 사용자 정의 AdamW 옵티마이저 설정

  # early stopping 기준 객체 생성
  early_stopper = EarlyStopping(patience=args.patience)
  best_score = float('inf')     # 초기 perplexity를 무한대로 설정

  # 에폭 반복 학습
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    # 각 배치에 대해 학습 수행
    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 배치 데이터를 GPU로 전송
      b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)

      # 그래디언트 초기화 및 forward
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)

      # 시퀀스의 마지막 토큰은 예측 대상 제외
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  
      labels = b_ids[:, 1:].contiguous().flatten()  # 첫 토큰은 정답에서 제외

      # 기본 loss 계산 (cross entropy)
      loss = F.cross_entropy(logits, labels, reduction='mean')

      # 보조 rhyme loss를 포함하는 경우 (현재 배치 텍스트 기준)
      rhyme_weight = getattr(args, "rhyme_loss_weight", 0.1)
      if rhyme_weight > 0:
          decoded_texts = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in b_ids]
          rhyme_loss = compute_rhyme_loss(decoded_texts)
          total_loss = loss + rhyme_weight * rhyme_loss
      else:
          total_loss = loss
          
      # 역전파 및 파라미터 업데이트
      total_loss.backward()
      optimizer.step()

      # 손실 누적
      train_loss += total_loss.item()
      num_batches += 1

    # 에폭별 평균 손실 출력
    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')

    # 검증 모드로 전환
    model.eval()
    generated_sonnets = []

    # 검증용 소넷 생성 (prompt로 시작)
    for i in range(len(val_dataset)):
        label, text = val_dataset[i]
        lines = text.split('\n')

        # 프롬프트 길이: 3줄, 5줄, 8줄 중 하나 무작위 선택
        prompt_lines = lines[:random.choice([3, 5, 8])]     
        prompt = '\n'.join(prompt_lines).strip()

        # 프롬프트 인코딩
        encoding = model.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # 라인 수 조정 위한 초기화
        max_resample_attempts = 5
        target_line_count = 14
        attempt = 0
        final_lines = []

        # 최대 5회 재샘플링하여 14줄 맞추기 시도
        while attempt < max_resample_attempts:
            _, generated_text = model.generate(encoding['input_ids'], temperature=args.temperature, max_length=130, min_length=80)
            lines = [line.strip() for line in generated_text.splitlines() if line.strip()]

            # 정확히 14줄이면 종료
            if len(lines) == target_line_count:
                final_lines = lines
                break
            # 14줄 초과 시 앞부분 자르기
            elif len(lines) > target_line_count:
                final_lines = lines[:target_line_count]
                break
            # 14줄 미만이면 다시 생성 (단, 가장 긴 결과 저장)
            else:
                attempt += 1
                if len(lines) > len(final_lines):
                    final_lines = lines

        # 아직 부족한 경우 빈 줄로 채움
        while len(final_lines) < target_line_count:
            final_lines.append("")

        # 최종 소넷 텍스트 정리
        cleaned_text = '\n'.join(final_lines)
        generated_sonnets.append((label, cleaned_text))

    # 생성된 텍스트만 추출
    preds = [s[1] for s in generated_sonnets]

    # 생성 결과 출력
    print("\n[Generated Sonnets Preview]")
    for i, sonnet in enumerate(preds): 
        print(f"\n-- Sonnet {i+1} --")
        print(sonnet)
        print("\n----------------\n")

    # rhyme 정확도 출력
    rhyme_acc = compute_rhyme_accuracy(preds)
    for i, sonnet in enumerate(preds):
        lines = sonnet.strip().split('\n')
        print(f"[Sonnet {i+1}] Number of lines: {len(lines)}")
    print(f"[VAL] Rhyme Acc: {rhyme_acc:.2f}")

    # perplexity 평가용 데이터로더 생성
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=val_dataset.collate_fn)
    ppl = compute_perplexity(model, val_loader, device)
    print(f"[VAL] Perplexity: {ppl:.2f}")

    # BLEU와 BERTScore는 정답이 없으므로 생략
    # perplexity 기준으로 best 모델 저장
    if epoch == 0 or ppl < best_score:
        best_score = ppl
        save_model(model, optimizer, args, f'best_{args.filepath}')

    # early stopping 조건 검사
    val_score = -ppl    # perplexity가 낮을수록 좋기 때문에 음수 부호
    if early_stopper(val_score):
        print("Early stopping triggered.")
        break
    
  # 학습 완료 후 최종 모델 및 디바이스 반환
  return model, device



# 모델과 옵티마이저 상태를 저장하는 함수
def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")



# 참조용 정답 소넷들을 불러오는 함수
def load_refs_by_sonnet(file_path):
    with open(file_path, 'r') as f:
        raw = f.read()
    sonnets = [s.strip() for s in raw.strip().split("\n\n") if s.strip()]   # 소넷은 빈 줄 기준으로 나뉨
    return sonnets



# 제출용 소넷 생성 함수 (held-out 데이터 기반)
@torch.no_grad()
def generate_submission_sonnets(args):
  
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  # 저장된 best 모델 로드
  saved = torch.load(f'best_{args.filepath}', map_location='cpu', weights_only=False)

  # 모델 초기화 및 파라미터 불러오기
  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # 테스트 데이터셋 및 참조 정답 로드
  test_dataset = SonnetsDataset("data/sonnets_held_out_dev.txt")
  test_refs = load_refs_by_sonnet("data/TRUE_sonnets_held_out_dev.txt")

  generated_sonnets = []    # 생성된 소넷 저장 리스트

  # 테스트 데이터셋의 각 샘플에 대해 소넷 생성
  for batch in test_dataset:

    # 입력 문장을 토크나이즈 및 인코딩
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # 생성 줄 수를 맞추기 위한 변수 초기화
    max_resample_attempts = 5       # 최대 재생성 시도 횟수
    target_line_count = 14          # 소넷의 목표 줄 수 (14줄)
    attempt = 0                     # 재시도 횟수 카운터
    final_lines = []                # 최종 생성 결과 (줄 단위)

    # 최대 5번까지 재생성하여 목표 줄 수를 만족하는 결과 확보
    while attempt < max_resample_attempts:
        _, generated_text = model.generate(encoding['input_ids'], temperature=args.temperature, max_length=130)
    
        # 줄 단위로 나누고 공백 제거
        lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
    
        # 줄 수가 정확히 14줄이면 바로 사용
        if len(lines) == target_line_count:
            final_lines = lines
            break
        # 초과 시 앞부분만 자름
        elif len(lines) > target_line_count:
            final_lines = lines[:target_line_count]
            break
        # 줄 수 부족 시 재시도 (그동안 가장 많은 줄을 유지)
        else:
            attempt += 1
            if len(lines) > len(final_lines):  # 가장 많은 줄을 기록
                final_lines = lines

    # 여전히 부족하면 빈 줄("")로 채움
    while len(final_lines) < target_line_count:
        final_lines.append("")

    # 최종 소넷 텍스트로 정제하여 저장
    cleaned_text = '\n'.join(final_lines)
    generated_sonnets.append((batch[0], cleaned_text))  # (id, 소넷 텍스트)

    lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
    cleaned_text = '\n'.join(lines)

  # 생성된 소넷 텍스트만 추출
  preds = [s[1] for s in generated_sonnets]

  # 성능 평가: BERTScore (텍스트 유사도), rhyme 정확도, perplexity
  _, bert_f1 = compute_text_similarity_scores(preds, test_refs, model.tokenizer)
  rhyme_acc = compute_rhyme_accuracy(preds)
  ppl = compute_perplexity(model, DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn), device)

  # 평가 결과 출력
  print(f"[TEST] BERT-F1: {bert_f1:.3f} | Rhyme Acc: {rhyme_acc:.2f} | Perplexity: {ppl:.2f}")

  # 생성된 소넷들을 결과 파일로 저장
  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")   # sample id
      f.write(sonnet[1])            # 생성된 소넷
  
  # 최종 결과 반환
  return generated_sonnets



# 명령줄 인자 파서 정의
def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--use_gpu", action='store_true')     # GPU 사용 여부

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=16)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  parser.add_argument("--eval_ppl", action='store_true')    # perplexity 평가 여부
  parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
  parser.add_argument("--rhyme_loss_weight", type=float, default=1.0)    # rhyme 보조 손실 가중치
  parser.add_argument("--rhyme_loss_interval", type=int, default=2)

  args = parser.parse_args()
  return args



# 모델 크기에 따라 hidden dim, layer 수 등 설정
def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
    args.rhyme_loss_weight = 0.2
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
    args.rhyme_loss_weight = 0.2
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
    args.rhyme_loss_weight = 0.2
  else:
    raise Exception(f'{args.model_size} is not supported.')     # 기타 사이즈는 미지원
  return args



# main 진입점
if __name__ == "__main__":
  args = get_args()                         # 인자 파싱
  args.eval_ppl = True                      # 항상 perplexity 평가
  args.filepath = 'sonnet_generation.pt'    # 모델 저장 파일 경로
  seed_everything(args.seed)                # 시드 고정 (재현성)

  model, device = train(args)               # 모델 학습
  generated_sonnets = generate_submission_sonnets(args)     # 제출용 소넷 생성