#!/usr/bin/env python3

# 필수 라이브러리 임포트
import random, numpy as np, argparse
from types import SimpleNamespace
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score
from models.gpt2 import GPT2Model   # 사용자 정의 GPT2 모델 모듈
from optimizer import AdamW         # 사용자 정의 AdamW 옵티마이저
from tqdm import tqdm               # 진행상황 시각화용
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# tqdm 출력 여부 설정
TQDM_DISABLE = False



# 재현성을 위한 모든 시드 고정 함수
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True



# GPT2 기반 감정 분류기 정의
class GPT2SentimentClassifier(torch.nn.Module):

  def __init__(self, config):
    super(GPT2SentimentClassifier, self).__init__()
    self.num_labels = config.num_labels
    self.gpt = GPT2Model.from_pretrained()  # 사전학습된 GPT2 로드

    # fine-tune 여부에 따라 GPT2 가중치 업데이트 설정
    assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
    for param in self.gpt.parameters():
      if config.fine_tune_mode == 'last-linear-layer':
        param.requires_grad = False
      elif config.fine_tune_mode == 'full-model':
        param.requires_grad = True

    # 감정 분류기용 레이어 추가
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)               # 드롭아웃 추가
    self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)  # 분류기 레이어


  def forward(self, input_ids, attention_mask):
    outputs = self.gpt(input_ids, attention_mask)   # GPT2 출력 얻기
    hidden_states = outputs["last_hidden_state"]    # 마지막 hidden layer
    last_token_idx = attention_mask.sum(dim=1) - 1  # 마지막 유효 토큰 인덱스
    last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_idx]  # 해당 위치 hidden 값 추출
    output = self.dropout(last_hidden)
    logits = self.classifier(output)  # 감정 분류 결과 (logits)
    return logits



# 학습/검증용 데이터셋 클래스
class SentimentDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # GPT2 토크나이저 로드
    self.tokenizer.pad_token = self.tokenizer.eos_token     # 패딩 토큰 지정


  def __len__(self):
    return len(self.dataset)


  def __getitem__(self, idx):
    return self.dataset[idx]


  # 배치 데이터를 패딩하여 텐서로 변환
  def pad_data(self, data):
    sents = [x[0] for x in data]      # 입력 문장 리스트
    labels = [x[1] for x in data]     # 정답 레이블 리스트
    sent_ids = [x[2] for x in data]   # 문장 ID 리스트

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])
    labels = torch.LongTensor(labels)

    return token_ids, attention_mask, labels, sents, sent_ids


  # DataLoader에서 사용하는 collate 함수 정의
  def collate_fn(self, all_data):
    token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data


# 테스트용 데이터셋 클래스 (label 없음)
class SentimentTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def pad_data(self, data):
    sents = [x[0] for x in data]      # 입력 문장 리스트
    sent_ids = [x[1] for x in data]   # 문장 ID 리스트

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    return token_ids, attention_mask, sents, sent_ids

  def collate_fn(self, all_data):
    token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sents': sents,
      'sent_ids': sent_ids
    }

    return batched_data



# 데이터 로딩 함수
def load_data(filename, flag='train'):
  num_labels = {}   # 레이블 수 카운트용
  data = []         # 데이터를 담을 리스트
  if flag == 'test':
    # 테스트셋은 label이 없으므로 문장과 id만 처리
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        data.append((sent, sent_id))
  else:
    # 학습 및 검증셋은 label 포함
    with open(filename, 'r', encoding='utf-8') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        label = int(record['sentiment'].strip())
        if label not in num_labels:
          num_labels[label] = len(num_labels)   # 새로운 레이블 카운트
        data.append((sent, label, sent_id))
    print(f"load {len(data)} data from {filename}")

  if flag == 'train':
    return data, len(num_labels)
  else:
    return data



def log_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)



  # 모델 저장 함수
def save_model(model, optimizer, args, config, filepath):
  save_info = {
    'model': model.state_dict(),      # 모델 가중치
    'optim': optimizer.state_dict(),  # 옵티마이저 상태
    'args': args,                     # 학습 인자
    'model_config': config,           # 설정 정보
    'system_rng': random.getstate(),  # 랜덤 시드 상태 저장
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")



# 모델 학습 함수
def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  wandb.login(key="24401929caaf123d1efe18d29a5e6a4f27e273a1")    # wandb 로그인
  wandb.init(project="nlproj", name="sentiment_task", config=vars(args))

  # 학습 및 검증 데이터 로드
  train_data, num_labels = load_data(args.train, 'train')
  dev_data = load_data(args.dev, 'valid')

  # 데이터셋 및 데이터로더 구성
  train_dataset = SentimentDataset(train_data, args)
  dev_dataset = SentimentDataset(dev_data, args)

  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                collate_fn=train_dataset.collate_fn)
  dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                              collate_fn=dev_dataset.collate_fn)

  # 설정 객체 정의 (hidden_dropout, 레이블 수, 사이즈 등)
  config = {'hidden_dropout_prob': args.hidden_dropout_prob,
            'num_labels': num_labels,
            'hidden_size': 768,
            'data_dir': '.',
            'fine_tune_mode': args.fine_tune_mode}
  config = SimpleNamespace(**config)

  # 모델 초기화
  model = GPT2SentimentClassifier(config)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)
  best_dev_acc = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    # 학습 루프 시작
    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask, b_labels = (batch['token_ids'],
                                 batch['attention_mask'], batch['labels'])

      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      b_labels = b_labels.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size     # 배치 평균 손실 사용

      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / (num_batches)

    # 에폭 종료 후 성능 평가
    train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
    dev_acc, dev_f1, dev_pred, dev_true, *_ = model_eval(dev_dataloader, model, device)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

    wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "dev_acc": dev_acc,
            "dev_f1": dev_f1
        })

    # 베스트 모델 저장 조건: dev accuracy 기준
    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, config, args.filepath)

    log_confusion_matrix(dev_true, dev_pred, labels=list(range(num_labels)))

  wandb.finish()



# 모델 평가 함수 (dev)
def model_eval(dataloader, model, device):
  model.eval()  # 평가 모드 (드롭아웃 등 비활성화)
  y_true = []   # 정답
  y_pred = []   # 예측값
  sents = []    # 문장
  sent_ids = [] # 문장 ID

  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    # 배치 요소 가져오기
    b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                                   batch['labels'], batch['sents'], batch['sent_ids']
    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    # 모델 예측
    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    b_labels = b_labels.flatten()
    y_true.extend(b_labels)
    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sents, sent_ids



# 테스트셋 평가 함수 (label 없이 예측만 수행)
def model_test_eval(dataloader, model, device):
  model.eval()  # 평가 모드 (드롭아웃 등 비활성화)
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                         batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  return y_pred, sents, sent_ids



# 테스트 함수 (저장된 모델 로드 후 dev 및 test 평가 수행)
def test(args):
  with torch.no_grad():
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # 저장된 모델 로드 및 설정 불러오기
    saved = torch.load(args.filepath, weights_only=False)   
    config = saved['model_config']                          # 저장된 설정 로드
    model = GPT2SentimentClassifier(config)                 # 모델 초기화
    model.load_state_dict(saved['model'])                   # 가중치 불러오기
    model = model.to(device)
    print(f"load model from {args.filepath}")

    # Dev 데이터 로드 및 평가
    dev_data = load_data(args.dev, 'valid')
    dev_dataset = SentimentDataset(dev_data, args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    # Test 데이터 로드 및 평가
    test_data = load_data(args.test, 'test')
    test_dataset = SentimentTestDataset(test_data, args)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    # Dev 평가
    dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
    print(f"✅ Dev Acc: {dev_acc:.4f}, F1: {dev_f1:.4f}")

    # Test 예측
    test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
    print('DONE Test')

    # Dev 결과 저장
    with open(args.dev_out, "w+") as f:
      print(f"dev acc :: {dev_acc :.3f}")
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(dev_sent_ids, dev_pred):
        f.write(f"{p}, {s} \n")

    # Test 결과 저장
    with open(args.test_out, "w+") as f:
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(test_sent_ids, test_pred):
        f.write(f"{p}, {s} \n")



# 실행 인자 정의 함수
def get_args():
  parser = argparse.ArgumentParser()

  # 고정 시드 설정
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=30)

  # 파인튜닝 방식 선택
  parser.add_argument("--fine-tune-mode", type=str,
                      help='last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated; full-model: GPT parameters are updated as well',
                      choices=('last-linear-layer', 'full-model'), default="full-model")
  
  # GPU 사용 여부
  parser.add_argument("--use_gpu", action='store_true')

  # 배치 사이즈 (GPU 메모리에 따라 조정 가능)
  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=32)
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)

  # 학습률 설정
  parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                      default=1e-5)

  args = parser.parse_args()
  return args



if __name__ == "__main__":
  args = get_args()             # 실행 인자 파싱
  seed_everything(args.seed)    # 시드 고정 (재현성 확보)

   # SST 데이터셋 학습 및 평가
  print('Training Sentiment Classifier on SST...')
  
  config = SimpleNamespace(     # SST용 config 정의
    filepath='sst-classifier_base.pt',   
    lr=args.lr,
    use_gpu=args.use_gpu,
    epochs=args.epochs,
    batch_size=args.batch_size,
    hidden_dropout_prob=args.hidden_dropout_prob,
    train='data/ids-sst-train.csv',
    dev='data/ids-sst-dev.csv',
    test='data/ids-sst-test-student.csv',
    fine_tune_mode=args.fine_tune_mode,
    dev_out='predictions/' + args.fine_tune_mode + '-sst-dev-out_base.csv',
    test_out='predictions/' + args.fine_tune_mode + '-sst-test-out_base.csv'
  )

  train(config)   # SST 학습 수행

  # SST 평가 수행
  print('Evaluating on SST...')   
  test(config)

  # CFIMDB 데이터셋 학습 및 평가
  print('Training Sentiment Classifier on cfimdb...')
  config = SimpleNamespace(
    filepath='cfimdb-classifier_base.pt',
    lr=args.lr,
    use_gpu=args.use_gpu,
    epochs=args.epochs,
    batch_size=8,
    hidden_dropout_prob=args.hidden_dropout_prob,
    train='data/ids-cfimdb-train.csv',
    dev='data/ids-cfimdb-dev.csv',
    test='data/ids-cfimdb-test-student.csv',
    fine_tune_mode=args.fine_tune_mode,
    dev_out='predictions/' + args.fine_tune_mode + '-cfimdb-dev-out_base.csv',
    test_out='predictions/' + args.fine_tune_mode + '-cfimdb-test-out_base.csv'
  )

  # CFIMDB 학습 수행
  train(config)

  # CFIMDB 평가 수행
  print('Evaluating on cfimdb...')
  test(config)