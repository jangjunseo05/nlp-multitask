# 라이브러리 임포트
import gc                                 # Garbage Collection 제어용
import argparse                           # 커맨드라인 인자 처리용
import random                             # 파이썬 random seed 설정용
import torch                              # PyTorch 메인 라이브러리
import numpy as np                        # 넘파이, 수치 연산용
import torch.nn.functional as F           # 손실함수 등 신경망 연산 함수
from torch import nn                      # 신경망 구성 요소
from torch.utils.data import DataLoader   # 배치 단위 데이터 로더
from tqdm import tqdm                     # 진행률 시각화

# 사용자 정의 데이터셋 및 로더 관련 함수
from datasets import (
  ParaphraseDetectionDataset,         # 학습/검증용 커스텀 Dataset 클래스
  ParaphraseDetectionTestDataset,     # 테스트용 Dataset 클래스
  load_paraphrase_data                # CSV 파일에서 데이터를 로드하는 함수
)

# 모델 평가 및 테스트용 함수
from evaluation import model_eval_paraphrase, model_test_paraphrase
from optimizer import AdamW                               # 사용자 정의 AdamW 옵티마이저 (torch.optim.AdamW와 동일하거나 커스텀)
from sklearn.metrics import accuracy_score, f1_score      # 정확도, F1-score 등 성능 측정용
from transformers import GPT2Model                        # Huggingface GPT-2 모델
from peft import get_peft_model, LoraConfig, TaskType     # LoRA 적용을 위한 PEFT 관련 라이브러리
from torch.nn.utils.rnn import pad_sequence               # 배치 단위 패딩을 위한 함수

TQDM_DISABLE = False    # tqdm 비활성화 여부



# 랜덤 시드 고정 함수 (재현성 확보 목적)
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True



# LoRA가 적용된 GPT2 래퍼 클래스 정의
class LoraGPT2Wrapper(nn.Module):
    
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.model_name = model_name

        # HuggingFace에서 사전학습된 GPT2 모델 로드
        self.base_model = GPT2Model.from_pretrained(model_name)

        # LoRA 설정
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["c_attn", "c_proj"]
        )

        # PEFT 모듈로 모델 변환 (LoRA 적용)
        self.peft_model = get_peft_model(self.base_model, peft_config)


    # LoRA가 적용된 GPT-2 모델 forward 수행
    def forward(self, input_ids, attention_mask):
        return self.peft_model(input_ids=input_ids, attention_mask=attention_mask)



# Paraphrase Detection GPT-2 모델 정의
class ParaphraseGPT(nn.Module):
    
    def __init__(self, args):
        super().__init__()    
        self.gpt = LoraGPT2Wrapper(model_name=args.model_size)    # LoRA가 적용된 GPT2 래퍼 클래스 초기화

        # GPT2 모델 크기에 따라 hidden size 지정
        if args.model_size == 'gpt2':
            hidden_size = 768
        elif args.model_size == 'gpt2-medium':
            hidden_size = 1024
        elif args.model_size == 'gpt2-large':
            hidden_size = 1280
        else:
            raise ValueError(f"Unsupported model size: {args.model_size}")

        self.paraphrase_detection_head = nn.Linear(hidden_size, 2)    # 마지막 hidden state를 이진 분류를 위한 선형 레이어로 투사


    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)                  # LoRA 적용 GPT-2 forward 수행
        seq_lengths = attention_mask.sum(dim=1) - 1                                             # 시퀀스의 마지막 실제 토큰 위치 인덱스 계산 (패딩 제외)
        last_hidden = outputs.last_hidden_state[torch.arange(input_ids.size(0)), seq_lengths]   # 마지막 토큰의 hidden state 추출
        logits = self.paraphrase_detection_head(last_hidden)                                    # 선형 분류기 통과하여 logits 생성 (2 클래스)
        return logits 



# Hard Negative 샘플 수집 함수
def collect_hard_negatives(dataloader, model, device, threshold=0.7):
  
  # 모델이 틀리게 예측했으나 확신(confidence)은 높은 샘플 수집
  model.eval()
  hard_negatives = []
  
  with torch.no_grad(): 
    for batch in tqdm(dataloader, desc="Collecting Hard Negatives", disable=TQDM_DISABLE):
        b_ids = batch['token_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).flatten()

        # 예측 및 softmax 확률 계산
        logits = model(b_ids, b_mask)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # 모델의 예측이 정답과 다르면서도 확신(confidence)이 높은 경우
        confidences = probs[torch.arange(len(preds)), preds]
        mask = (preds != labels) & (confidences > threshold)

        # 해당 샘플을 hard negative로 수집
        for i in range(len(labels)):
            if mask[i]:
                hard_negatives.append((
                    b_ids[i].detach().cpu(),    # 입력 ID
                    b_mask[i].detach().cpu(),   # attention mask
                    labels[i].detach().cpu()    # 실제 정답 라벨
                ))

  # 수집된 hard negative의 레이블 분포 출력
  print(f" ▶ HNP 라벨 분포: label=0 → {sum(x[2].item() == 0 for x in hard_negatives)}, label=1 → {sum(x[2].item() == 1 for x in hard_negatives)}")
  
  return hard_negatives



# Hard Negative Fine-Tuning 함수
def fine_tune_on_hard_negatives(model, args, device, train_dataloader, dev_dataloader):
    
    # 1단계: Hard Negative 수집
    print("\n🔍 Collecting hard negatives from train set...")
    hard_negatives = collect_hard_negatives(train_dataloader, model, device)
    print(f" Collected {len(hard_negatives)} hard negatives.")

    # 2단계: Soft Positive 샘플 수집 (정답 맞췄지만 confidence 낮은 샘플 중 일부만 랜덤 선택)
    print("📥 Sampling soft positives from original training data...")
    model.eval()
    soft_positives = []

    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Collecting Soft Positives", disable=TQDM_DISABLE):
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].flatten().to(device)

            logits = model(b_ids, b_mask)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            confidences = probs[torch.arange(len(preds)), preds]
            correct_mask = (preds == labels) & (confidences < 0.8)  # 낮은 confidence의 정답

            for i in range(len(labels)):
                if correct_mask[i] and random.random() < args.soft_pos_ratio:   # soft_pos_ratio 기반 랜덤 샘플링
                    soft_positives.append((
                        b_ids[i].detach().cpu(),
                        b_mask[i].detach().cpu(),
                        labels[i].detach().cpu()
                    ))

    print(f" ✅ Added {len(soft_positives)} soft positives to hard negatives.")

    # HNP + soft positive 합치기
    combined_samples = hard_negatives + soft_positives

    # 클래스 별 개수 세기 및 가중치 계산 (불균형 보정)
    label_counts = [0, 0]
    for item in combined_samples:
        label_counts[item[2].item()] += 1

    total = sum(label_counts)
    class_weights = [total / c if c > 0 else 0 for c in label_counts]
    class_weights = torch.tensor(class_weights).to(device)

    print(f" ▶ Combined Label Count: 0 → {label_counts[0]}, 1 → {label_counts[1]}")
    print(f" ▶ Computed Class Weights: {class_weights}")

    # HNP 학습용 옵티마이저 설정 (weight_decay로 정규화 효과)
    hnp_optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)
    

    # 하드 네거티브를 위한 배치 구성 함수 정의
    def collate_batch(batch):
        token_ids = pad_sequence([x[0] for x in batch], batch_first=True)
        attention_masks = pad_sequence([x[1] for x in batch], batch_first=True)
        labels = torch.stack([x[2] for x in batch])
        return token_ids, attention_masks, labels
    
    # 7단계: 결합된 데이터로 DataLoader 구성
    combined_dataloader = DataLoader(combined_samples, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=collate_batch)
    
    # HNP 학습 (1 epoch)
    for epoch in range(1):
        model.train()
        train_loss = 0
        for batch in tqdm(combined_dataloader, desc='fine-tune', disable=TQDM_DISABLE):
            b_ids, b_mask, labels = [x.to(device) for x in batch]
            hnp_optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, labels, reduction='mean', weight=class_weights)
            loss.backward()
            hnp_optimizer.step()
            train_loss += loss.item()

        print(f"Fine-tune Epoch {epoch}: loss = {train_loss / len(combined_dataloader):.4f}")
        dev_acc, dev_f1, *_ = model_eval_paraphrase(dev_dataloader, model, device)
        print(f"Dev accuracy after fine-tune-{epoch}: {dev_acc:.4f}, f1-score: {dev_f1:.4f}")

    return hnp_optimizer



# 모델 학습 함수 (기본 학습 + HNP Fine-tuning 포함)
def train(args):
  
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  # Quora 데이터 로드 (학습 및 dev 데이터)
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  # 커스텀 Dataset 클래스 적용
  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  # DataLoader 생성 (shuffle은 train만 적용)
  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)    # 모델 크기에 따른 인자 보완

  # 모델 초기화 및 병렬 처리 설정
  model = ParaphraseGPT(args)
  model = nn.DataParallel(model)
  model = model.to(device)
  print(device)

  # 옵티마이저 설정
  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
  best_dev_acc = 0    # dev 최고 정확도 기록용

  # 기본 학습 루프 (Epoch 단위)
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 입력 데이터와 라벨 GPU로 이동
        b_ids = batch['token_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].flatten().to(device)

      # 순전파, 손실 계산, 역전파, 파라미터 업데이트
        optimizer.zero_grad()
        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits, labels, reduction='mean')
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    train_loss = train_loss / num_batches   # 평균 학습 손실 계산
    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)   # 개발 데이터셋 성능 평가

    # 최고 성능 갱신 시 모델 저장
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")
    gc.collect()    # 메모리 정리
    torch.cuda.empty_cache()

  gc.collect()    # 메모리 정리
  torch.cuda.empty_cache()

  # HNP 학습 전: best 모델 다시 로드
  print("\n🔁 Loading best model before HNP fine-tuning...")
  saved = torch.load(args.filepath, weights_only=False)
  model.load_state_dict(saved['model'], strict=False)
  model = model.to(device)

  # 하드 네거티브 샘플 기반 추가 학습 수행
  hnp_optimizer = fine_tune_on_hard_negatives(model, args, device, para_train_dataloader, para_dev_dataloader)

  # Fine-tune 후 모델 저장
  finetuned_path = args.filepath.replace(".pt", "-hnp.pt")
  save_model(model, hnp_optimizer, args, finetuned_path)

  # HNP 학습 모델 평가 결과 출력
  dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"✅ [After HNP Fine-Tune] Dev accuracy: {dev_acc:.4f}, f1-score: {dev_f1:.4f}")

  # ✅ 5 epoch 중 best 모델 성능 다시 출력 (최종 정리)
  print("\n📊 Final Evaluation of Best 5epoch Model (before HNP)...")
  model.load_state_dict(saved['model'], strict=False)
  model = model.to(device)
  model.eval()

  dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f" ✅ [Best of 5 Epochs] Dev accuracy: {dev_acc:.4f}, f1-score: {dev_f1:.4f}")



# 모델 저장 함수
def save_model(model, optimizer, args, filepath):
  # 모델과 옵티마이저 상태, 학습 설정, 난수 시드 상태까지 저장
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



# 테스트 함수: dev/test 데이터에 대한 예측 수행 및 결과 파일 저장
@torch.no_grad()
def test(args):

  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  # 저장된 모델 로드
  saved = torch.load(args.filepath, weights_only=False)
  model = ParaphraseGPT(saved['args'])
  model = nn.DataParallel(model)
  model.load_state_dict(saved['model'], strict=False)
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  # 데이터 로드 및 전처리
  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  # 개발셋 및 테스트셋에 대해 예측 수행
  dev_para_acc, dev_para_f1, dev_para_y_pred, dev_para_y_true, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  # 예측 결과 파일 저장 (dev)
  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      label_str = "yes" if s == 1 else "no"
      f.write(f"{p}, {label_str} \n")
  print(f"📁 예측 결과 저장 완료: {args.para_dev_out}")

  # 예측 결과 파일 저장 (test)
  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      label_str = "yes" if s == 1 else "no"
      f.write(f"{p}, {label_str} \n")
  print(f"📁 예측 결과 저장 완료: {args.para_test_out}")
  print(f"✅ [{args.filepath}]에 기반한 예측 결과 저장 완료!")



# ✅ 커맨드라인 인자 정의 함수
def get_args():

  parser = argparse.ArgumentParser()

  # 파일 경로 관련 인자
  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  # 학습 관련 설정 인자
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=64)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  parser.add_argument("--soft_pos_ratio", type=float, default=0.1,
                    help="비교적 정답을 맞춘 low-confidence 샘플의 샘플링 비율 (default=0.1)")

  args = parser.parse_args()
  return args



# 모델 크기에 따라 관련 인자 자동 설정 함수
def add_arguments(args):
  
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args



# 실행 시작점: 학습 및 평가 전체 파이프라인 실행
if __name__ == "__main__":
  args = get_args()                                             # 인자 파싱
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'      # 모델 저장 파일명 구성
  seed_everything(args.seed)                                    # 재현성을 위한 random seed 고정.

  train(args)   # 전체 훈련 + HNP 수행

  # HNP 전 모델로 예측 결과 생성
  args.para_dev_out = "predictions/para-dev-output-best5.csv"
  args.para_test_out = "predictions/para-test-output-best5.csv"
  test(args)

  # HNP 후 모델로 예측 결과 생성
  args.filepath = args.filepath.replace(".pt", "-hnp.pt")
  args.para_dev_out = "predictions/para-dev-output-hnp6.csv"
  args.para_test_out = "predictions/para-test-output-hnp6.csv"
  test(args)