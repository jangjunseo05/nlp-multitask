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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# tqdm 출력 여부 설정
TQDM_DISABLE = False



# 난수 고정 함수: 재현성 확보
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True



# Label smoothing을 위한 soft label 생성 함수
def get_soft_labels(hard_labels, num_classes, smoothing=0.1):
    assert 0 <= smoothing < 1, "Smoothing must be in [0, 1)"
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / (num_classes - 1)

    # 모든 위치에 smoothing_value 채운 후 hard label 위치에 confidence 대입
    soft_labels = torch.full(
        (hard_labels.size(0), num_classes),
        fill_value=smoothing_value,
        device=hard_labels.device,
        dtype=torch.float
    )
    soft_labels.scatter_(1, hard_labels.unsqueeze(1), confidence)

    return soft_labels



# 기본 단일 GPT-2 감정 분류기 정의
class GPT2SentimentClassifier(torch.nn.Module):

  def __init__(self, config):
    super().__init__()
    self.num_labels = config.num_labels
    self.gpt = GPT2Model.from_pretrained()  # GPT2 사전 학습 모델 로드

    # fine-tune 여부에 따라 GPT2 파라미터 업데이트 여부 설정.
    assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
    for param in self.gpt.parameters():
      if config.fine_tune_mode == 'last-linear-layer':
        param.requires_grad = False
      elif config.fine_tune_mode == 'full-model':
        param.requires_grad = True

    # ✅ 감정 분류기용 레이어 추가
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)              # 드롭아웃
    self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)   # 최종 분류기

    # 가중치 초기화
    torch.nn.init.xavier_uniform_(self.classifier.weight)
    if self.classifier.bias is not None:
        torch.nn.init.zeros_(self.classifier.bias)


  # forward 함수: 마지막 토큰의 hidden state → 분류기
  def forward(self, input_ids, attention_mask):
    outputs = self.gpt(input_ids, attention_mask)
    hidden_states = outputs["last_hidden_state"] 
    last_token_idx = attention_mask.sum(dim=1) - 1
    last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_idx]
    output = self.dropout(last_hidden)
    logits = self.classifier(output)
    return logits



# 클래스 불균형 보정을 위한 class weight 계산 함수
def get_class_weights(y_labels, num_classes):
    weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_labels)
    return torch.tensor(weights, dtype=torch.float)



# cascade 구조 분류기 정의 (coarse → fine0/fine1)
class CascadeSentimentClassifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # coarse/fine 분류기를 위한 config 복사 및 클래스 수 설정
        coarse_config = SimpleNamespace(**vars(config))
        coarse_config.num_labels = 2
        coarse_config.hidden_size = 768

        fine0_config = SimpleNamespace(**vars(config))
        fine0_config.num_labels = 3
        fine0_config.hidden_size = 768

        fine1_config = SimpleNamespace(**vars(config))
        fine1_config.num_labels = 2
        fine1_config.hidden_size = 768

        # 각각의 GPT-2 분류기 생성
        self.coarse = GPT2SentimentClassifier(coarse_config)
        self.fine0 = GPT2SentimentClassifier(fine0_config)
        self.fine1 = GPT2SentimentClassifier(fine1_config)

        # 각 분류기의 가중치 초기화
        self.coarse_weight = None
        self.fine0_weight = None
        self.fine1_weight = None


    # 외부에서 class weight 주입하는 함수
    def set_class_weights(self, coarse_w, fine0_w, fine1_w):
        self.coarse_weight = coarse_w
        self.fine0_weight = fine0_w
        self.fine1_weight = fine1_w


    # forward 함수: 전체 cascade loss 계산
    def forward(self, input_ids, attention_mask, labels):
        device = input_ids.device
        loss_total = 0
        weight_total = 0

        # coarse target (0~2: 0, 3~4: 1)
        coarse_target = (labels >= 3).long()
        coarse_logits = self.coarse(input_ids, attention_mask)
        coarse_probs = F.softmax(coarse_logits, dim=1)
        coarse_loss_weight = 0.5

        # coarse loss (가중치 적용 가능)
        coarse_loss = F.cross_entropy(coarse_logits, coarse_target, weight=self.coarse_weight if self.coarse_weight is not None else None, reduction='none')
        loss_total += coarse_loss_weight * coarse_loss.mean()
        weight_total += coarse_loss_weight

        # fine 0,1 분기 (0~2 / 3~4)
        mask_fine0 = (labels < 3)
        mask_fine1 = (labels >= 3)

        fine0_loss_vec = torch.zeros(input_ids.size(0), device=device)
        fine1_loss_vec = torch.zeros(input_ids.size(0), device=device)

        # fine0 분류기 처리 (label smoothing + KL loss)
        if mask_fine0.any():
            fine0_ids = input_ids[mask_fine0]
            fine0_mask = attention_mask[mask_fine0]
            fine0_labels = labels[mask_fine0]

            fine0_logits = self.fine0(fine0_ids, fine0_mask)
            soft_fine0_labels = get_soft_labels(fine0_labels, num_classes=3, smoothing=0.1) 
            fine0_log_probs = F.log_softmax(fine0_logits, dim=-1)
            kl = F.kl_div(fine0_log_probs, soft_fine0_labels, reduction='none').sum(dim=1)

            fine0_loss_vec[mask_fine0] = kl

        # fine1 분류기 처리 (cross entropy)
        if mask_fine1.any():
            fine1_ids = input_ids[mask_fine1]
            fine1_mask = attention_mask[mask_fine1]
            fine1_labels = labels[mask_fine1] - 3   # [3,4] → [0,1]

            fine1_logits = self.fine1(fine1_ids, fine1_mask)
            ce = F.cross_entropy(fine1_logits, fine1_labels, weight=self.fine1_weight if self.fine1_weight is not None else None)
            fine1_loss_vec[mask_fine1] = ce

        # fine loss 가중 평균 (coarse 확률 기반)
        with torch.no_grad():
            p0 = coarse_probs[:, 0]
            p1 = coarse_probs[:, 1]

        fine_loss = p0 * fine0_loss_vec + p1 * fine1_loss_vec
        loss_total += fine_loss.mean()
        weight_total += 1.0
           
        return loss_total / weight_total



# 학습/검증 데이터셋 클래스 정의
class SentimentDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # GPT-2 토크나이저 로드
    self.tokenizer.pad_token = self.tokenizer.eos_token     # 패딩 토큰 지정


  def __len__(self):
    return len(self.dataset)


  def __getitem__(self, idx):
    return self.dataset[idx]


  # 배치 데이터를 패딩 및 텐서 변환
  def pad_data(self, data):
    sents = [x[0] for x in data]      # 문장 리스트
    labels = [x[1] for x in data]     # 레이블 리스트
    sent_ids = [x[2] for x in data]   # 문장 ID

    encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])
    labels = torch.LongTensor(labels)

    return token_ids, attention_mask, labels, sents, sent_ids


  # DataLoader에서 사용하는 collate 함수
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



# 테스트 데이터셋 클래스 정의 (레이블 없음)
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
    sents = [x[0] for x in data]      # 문장
    sent_ids = [x[1] for x in data]   # 문장 ID

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
  num_labels = {}
  data = []
  if flag == 'test': # 테스트 데이터는 라벨 없이 문장과 id만
    with open(filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        data.append((sent, sent_id))
  else: # 학습/검증 데이터는 라벨 포함
    with open(filename, 'r', encoding='utf-8') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent = record['sentence'].lower().strip()
        sent_id = record['id'].lower().strip()
        label = int(record['sentiment'].strip())
        if label not in num_labels:
          num_labels[label] = len(num_labels)
        data.append((sent, label, sent_id))
    print(f"load {len(data)} data from {filename}")

  if flag == 'train':
    return data, len(num_labels)
  else:
    return data
  


# 조기 종료 클래스 정의
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None


    # F1 score가 개선되면 True 반환, 아니면 카운터 증가
    def step(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False
        


# Joint 학습 함수 (SST + cascade 구조)
def train_joint(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    wandb.login(key="24401929caaf123d1efe18d29a5e6a4f27e273a1")    # wandb 로그인
    wandb.init(project="nlproj", name="sst_joint_training", config=vars(args))

    # 데이터 로딩
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')
    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)

    # DataLoader 구성
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

    # 각 단계별 class weight 계산 (imbalanced data 대응)
    train_labels = [y for (_, y, _) in train_data]
    coarse_labels = [int(y >= 3) for y in train_labels]
    fine0_labels = [y for y in train_labels if y in [0, 1, 2]]
    fine1_labels = [y - 3 for y in train_labels if y in [3, 4]]

    coarse_class_weights = get_class_weights(coarse_labels, 2).to(device)
    fine0_class_weights = get_class_weights(fine0_labels, 3).to(device)
    fine1_class_weights = get_class_weights(fine1_labels, 2).to(device)

    # 모델 초기화 및 class weight 설정
    config = SimpleNamespace(hidden_dropout_prob=args.hidden_dropout_prob, num_labels=num_labels, hidden_size=768, data_dir='.', fine_tune_mode=args.fine_tune_mode)
    model = CascadeSentimentClassifier(config).to(device)
    model.set_class_weights(coarse_class_weights, fine0_class_weights, fine1_class_weights)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    early_stopper = EarlyStopping(patience=args.patience)

    # 학습 루프
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"train-{epoch}", disable=TQDM_DISABLE):
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            optimizer.zero_grad()
            loss = model(b_ids, b_mask, b_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 검증 성능 평가 및 모델 저장
        dev_acc, dev_f1, dev_pred, dev_true, *_ = cascade_model_eval(dev_dataloader, model, device)
        if early_stopper.step(dev_f1):  # or acc
            torch.save({'model': model.state_dict(), 'model_config': args}, "sst-classifier_cascade.pt")

        print(f"Epoch {epoch} | Train loss: {total_loss:.3f} | Dev acc: {dev_acc:.3f} | Dev f1: {dev_f1:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": total_loss, "dev_acc": dev_acc, "dev_f1": dev_f1})

        # Confusion matrix 시각화 및 로깅
        cm = confusion_matrix(dev_true, dev_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[0,1,2,3,4],
                    yticklabels=[0,1,2,3,4])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix (Dev) - Epoch {epoch+1}")
        wandb.log({f"confusion_matrix_epoch_{epoch+1}": wandb.Image(plt)})

        plt.close()
      
    wandb.finish()



# 모델 저장 함수
def save_model(model, optimizer, args, config, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'model_config': config,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")



# Cascade 모델 평가 함수 (dev set)
def cascade_model_eval(dataloader, model, device):
    model.eval()  # 평가 모드 (드롭아웃 등 비활성화)
    y_true = []
    y_pred = []

    for batch in tqdm(dataloader, desc='eval', disable=TQDM_DISABLE):
        b_ids = batch['token_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            coarse_logits = model.coarse(b_ids, b_mask)
            coarse_preds = torch.argmax(coarse_logits, dim=1)

            final_preds = []
            for i in range(len(coarse_preds)):
                inp = b_ids[i:i+1]
                mask = b_mask[i:i+1]

                # coarse prediction에 따라 fine 분류기 선택
                if coarse_preds[i] == 0:
                    fine0_logits = model.fine0(inp, mask)
                    fine0_pred = torch.argmax(fine0_logits, dim=1).item()
                    final_preds.append(fine0_pred)
                else:
                    fine1_logits = model.fine1(inp, mask)
                    fine1_pred = torch.argmax(fine1_logits, dim=1).item()
                    final_preds.append(3 + fine1_pred)  # fine1: 0/1 → 3/4로 복원

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(final_preds)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    return acc, f1, y_pred, y_true



# Cascade 모델 테스트셋 예측 및 결과 저장 함수
def full_cascade_inference(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath, weights_only=False)
    config = saved['model_config']

    # Cascade 모델 로드
    model = CascadeSentimentClassifier(config).to(device)
    model.load_state_dict(saved['model'])
    model.eval()  # 평가 모드 (드롭아웃 등 비활성화)
    print(f"load cascade model from {args.filepath}")

    # 테스트 데이터 로드 및 DataLoader 구성
    test_data = load_data(args.test, 'test')
    test_dataset = SentimentTestDataset(test_data, args)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

    final_preds = []
    sent_ids = []

    # 배치별 테스트 수행
    for batch in tqdm(test_dataloader, desc='inference-full', disable=TQDM_DISABLE):
        b_ids = batch['token_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        b_sids = batch['sent_ids']

        coarse_logits = model.coarse(b_ids, b_mask)
        coarse_preds = torch.argmax(coarse_logits, dim=1)

        for i in range(len(coarse_preds)):
            inp = b_ids[i:i+1]
            mask = b_mask[i:i+1]
            sid = b_sids[i]

            # coarse 결과에 따라 fine 분류기 수행
            if coarse_preds[i] == 0:
                fine0_logits = model.fine0(inp, mask)
                fine0_pred = torch.argmax(fine0_logits, dim=1).item()
                final_preds.append(fine0_pred)
            else:
                fine1_logits = model.fine1(inp, mask)
                fine1_pred = torch.argmax(fine1_logits, dim=1).item()
                final_preds.append(3 + fine1_pred)

            sent_ids.append(sid)

    # 결과 저장
    with open(args.test_out, "w", encoding="utf-8") as f:
        f.write("id,Predicted_Sentiment\n")
        for sid, label in zip(sent_ids, final_preds):
            f.write(f"{sid},{label}\n")
    print("✅ Saved full cascade prediction to", args.test_out)



# 단일 GPT2 감정 분류기 학습 함수 (FOR CFIMDB)
def train(args):

  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  wandb.login(key="24401929caaf123d1efe18d29a5e6a4f27e273a1")
  wandb.init(project="nlproj", name="sentiment_task", config=vars(args))

  # 데이터 로드 및 DataLoader 생성
  train_data, num_labels = load_data(args.train, 'train')
  dev_data = load_data(args.dev, 'valid')

  train_dataset = SentimentDataset(train_data, args)
  dev_dataset = SentimentDataset(dev_data, args)

  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                collate_fn=train_dataset.collate_fn)
  dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                              collate_fn=dev_dataset.collate_fn)

  # 모델 설정
  config = {'hidden_dropout_prob': args.hidden_dropout_prob,
            'num_labels': num_labels,
            'hidden_size': 768,
            'data_dir': '.',
            'fine_tune_mode': args.fine_tune_mode}
  config = SimpleNamespace(**config)

  model = GPT2SentimentClassifier(config)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)
  best_dev_acc = 0

  # 학습 루프
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask, b_labels = (batch['token_ids'],
                                 batch['attention_mask'], batch['labels'])
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      b_labels = b_labels.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, b_labels)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    # dev 성능 평가 및 best model 저장
    train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
    dev_acc, dev_f1, dev_pred, dev_true, *_ = model_eval(dev_dataloader, model, device)

    if dev_acc >= best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, config, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "dev_acc": dev_acc,
        "dev_f1": dev_f1
    })

  # confusion matrix 시각화 및 wandb 로깅
  cm = confusion_matrix(dev_true, dev_pred)
  plt.figure(figsize=(6, 5))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
              xticklabels=list(range(num_labels)),
              yticklabels=list(range(num_labels)))
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.title("Confusion Matrix (Dev)")
  wandb.log({"confusion_matrix": wandb.Image(plt)})

  wandb.finish()



# 일반 모델 평가 함수 (dev, train 공통).
def model_eval(dataloader, model, device):
  model.eval()  # 평가 모드 (드롭아웃 등 비활성화)
  y_true = []
  y_pred = []
  sents = []
  sent_ids = []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
                                                   batch['labels'], batch['sents'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

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



# 일반 모델 테스트 및 결과 저장 함수
def test(args):
  with torch.no_grad():
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath, weights_only=False)
    config = saved['model_config']

    # 학습된 모델 불러오기
    model = GPT2SentimentClassifier(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    print(f"load model from {args.filepath}")

    # 데이터셋 로드 및 DataLoader 구성
    dev_data = load_data(args.dev, 'valid')
    dev_dataset = SentimentDataset(dev_data, args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    test_data = load_data(args.test, 'test')
    test_dataset = SentimentTestDataset(test_data, args)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn)
    # dev 성능 평가
    dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
    print(f"✅ Dev Acc: {dev_acc:.4f}, F1: {dev_f1:.4f}")

    # test 예측 수행
    test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
    print('DONE Test')

    # dev 예측 결과 저장
    with open(args.dev_out, "w+") as f:
      print(f"dev acc :: {dev_acc :.3f}")
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(dev_sent_ids, dev_pred):
        f.write(f"{p}, {s} \n")

    # test 예측 결과 저장
    with open(args.test_out, "w+") as f:
      f.write(f"id \t Predicted_Sentiment \n")
      for p, s in zip(test_sent_ids, test_pred):
        f.write(f"{p}, {s} \n")



# 테스트셋 전용 평가 함수 (label 없이 예측만 수행)
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

    logits = model(b_ids, b_mask)                # 모델 예측
    logits = logits.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()  # 가장 높은 확률의 클래스 선택

    y_pred.extend(preds)
    sents.extend(b_sents)
    sent_ids.extend(b_sent_ids)

  return y_pred, sents, sent_ids



# 명령줄 인자 정의 함수
def get_args():
  parser = argparse.ArgumentParser()  # ArgumentParser 객체 생성
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=30)
  parser.add_argument("--fine-tune-mode", type=str,
                      help='last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated; full-model: GPT parameters are updated as well',
                      choices=('last-linear-layer', 'full-model'), default="full-model")
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=32)
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
  parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                      default=1e-5)

  args = parser.parse_args()  # 명령줄 인자 파싱
  return args



# 메인 실행 함수
if __name__ == "__main__":
    
      args = get_args()           # 명령줄 인자 파싱
      seed_everything(args.seed)  # 시드 고정으로 재현성 확보

      print('Training Sentiment Classifier on SST...')
      # SST 데이터셋에 대해 cascade 구조로 학습
      config_sst_joint = SimpleNamespace(
          filepath='cascade-joint.pt',  # joint 학습 중간 모델 저장 경로
          lr=args.lr,
          use_gpu=args.use_gpu,
          epochs=args.epochs,
          batch_size=args.batch_size,
          hidden_dropout_prob=args.hidden_dropout_prob,
          train='data/ids-sst-train.csv',
          dev='data/ids-sst-dev.csv',
          test='data/ids-sst-test-student.csv',
          patience=5,
          task="sst",
          fine_tune_mode=args.fine_tune_mode,
          dev_out='predictions/' + args.fine_tune_mode + '-sst-dev-out_cascade.csv',
          test_out='predictions/' + args.fine_tune_mode + '-sst-test-out_cascade.csv'
      )
      train_joint(config_sst_joint)   # SST 학습 수행

      config_sst_joint.filepath = "sst-classifier_cascade.pt"   # 최종 모델 경로 업데이트
      full_cascade_inference(config_sst_joint)          # SST 테스트셋 예측 수행

      # CFIMDB 데이터셋은 일반 GPT2 분류기로 학습
      print('Training Sentiment Classifier on cfimdb...')
      config_cfimdb = SimpleNamespace(
          filepath='cfimdb-classifier_cascade.pt',
          lr=args.lr,
          use_gpu=args.use_gpu,
          epochs=args.epochs,
          batch_size=8,
          hidden_dropout_prob=args.hidden_dropout_prob,
          train='data/ids-cfimdb-train.csv',
          dev='data/ids-cfimdb-dev.csv',
          test='data/ids-cfimdb-test-student.csv',
          task="cfimdb",
          fine_tune_mode=args.fine_tune_mode,
          dev_out='predictions/' + args.fine_tune_mode + '-cfimdb-dev-out_cascade.csv',
          test_out='predictions/' + args.fine_tune_mode + '-cfimdb-test-out_cascade.csv'
      )

      train(config_cfimdb)  # CFIMDB 학습
      print('Evaluating on cfimdb...')
      test(config_cfimdb)   # CFIMDB 테스트 및 결과 저장