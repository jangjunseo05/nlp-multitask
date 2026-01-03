# !/usr/bin/env python3


"""
이 파일은 Quora의 Paraphrase Detection을 위한 Dataset 클래스를 포함한다. 추가 데이터 소스로 훈련시키거나
Quora 데이터셋의 처리 방식(예: 데이터 증강 등)을 변경하려는 경우 이 파일을 수정할 수 있다.
"""

import csv

import re
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def preprocess_string(s):
  return ' '.join(s.lower()
                  .replace('.', ' .')
                  .replace('?', ' ?')
                  .replace(',', ' ,')
                  .replace('\'', ' \'')
                  .split())


class ParaphraseDetectionDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]  # (input_text, label)

    def collate_fn(self, all_data):
        texts = [x[0] for x in all_data]
        labels = [x[1] for x in all_data]
        sent_ids = [x[2] for x in all_data]

        encoding = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        token_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        labels = torch.LongTensor(labels)

        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sent_ids': sent_ids
        }



class ParaphraseDetectionTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]  # (input_text, sent_id)

    def collate_fn(self, all_data):
        texts = [x[0] for x in all_data]
        sent_ids = [x[1] for x in all_data]

        encoding = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        token_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sent_ids': sent_ids
        }



def load_paraphrase_data(paraphrase_filename, split='train'):
    paraphrase_data = []
    with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        for record in reader:
            try:
                input_text = record['input_text'].strip()
                if split == 'test':
                    sent_id = record.get('id', '')  # test set일 경우 ID만 있으면 사용
                    paraphrase_data.append((input_text, sent_id))
                else:
                    label = int(record['label'])
                    sent_id = record.get('id', '')  # ID가 있으면 쓰고 없으면 공백
                    paraphrase_data.append((input_text, label, sent_id))
            except Exception as e:
                print(f"❗️ Error while reading a record: {e}")
                continue

    print(f"✅ Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
    return paraphrase_data


class SonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return (idx, self.sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data
