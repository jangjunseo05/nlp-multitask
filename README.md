# GPT-2 기반 멀티태스크 NLP: 모델과 사람의 기대 사이 간극 좁히기

[English](README.en.md)

## 📌 프로젝트 개요
GPT-2로 멀티태스크 NLP 시스템을 구현하며 **사람과 AI의 상호작용**이 어디까지 가능한지 살펴본 프로젝트입니다. 감성 분류, 패러프레이즈 탐지, 소넷 생성 세 과제에서 모델의 용량과, 사람이 만든 데이터에 본래 들어 있는 모호함 사이의 간극을 좁히는 데 집중했습니다.

## 🛠️ 핵심 연구 포인트와 결정

### 1️⃣ 감성 분류: 클래스 불균형 극복
- **전략:** **Coarse-to-Fine Cascade 모델**(2단계 · 3단계)과 데이터 리프레이징 기법을 적용했습니다.
- **결정:** 실험으로 비교한 뒤 최종 구조로 **2단계 모델**을 택했습니다. 분석해 보니 더 깊은 cascade(3단계)는 **오류 전파** 문제가 있었고, 세밀한 분류 과제에서는 모델 복잡도와 학습 안정성 사이의 균형이 필요하다는 것을 확인했습니다.

### 2️⃣ 패러프레이즈 탐지: 효율과 정밀도
- **전략:** 파라미터 효율적 미세조정을 위해 **LoRA(Low-Rank Adaptation)** 를, 그리고 **Hard Negative Priority(HNP)** 학습을 적용했습니다.
- **결정:** HNP에서 **신뢰도와 잡음 사이의 트레이드오프**를 분석했습니다. HNP는 어려운 샘플을 겨냥하지만, 높은 신뢰도로 틀린 샘플로 학습하면 오히려 잡음처럼 작용할 수 있다는 것을 발견했고, 그래서 hard negative를 원래 학습 데이터와 합쳐 학습하는 방식으로 전략을 다듬었습니다.

### 3️⃣ 소넷 생성: 사람이 정한 형식에 맞추기
- **전략:** 마지막 단어 임베딩의 코사인 유사도를 기반으로 한 자체 **각운 손실(Rhyme Loss)** 을 넣은 셰익스피어 소넷 생성기를 개발했습니다.
- **결정:** 구조적 정확성을 위해 14행 후처리 로직을 구현했습니다. 그 결과 의미적 일관성(BERTScore)을 떨어뜨리지 않으면서 **각운 정확도가 0.00 → 0.12로 크게 개선**되어, 생성 모델에 시의 형식 제약을 거는 방식이 효과가 있음을 보였습니다.

## 📊 결과와 솔직한 평가
* **감성 분류:** 리프레이징으로 F1 점수는 올랐지만(0.41 → 0.44), Cascade 구조는 1단계(coarse) 분류기의 정확도가 병목이라는 점이 드러났습니다.
* **패러프레이즈:** 기본 GPT-2(5 epoch)가 HNP만 적용한 튜닝보다 더 안정적이었고, 샘플 선택에서 신뢰도 임계값에 매우 민감하다는 점이 확인됐습니다.
* **소넷:** Perplexity를 낮추고(35.68 → 34.15) 구조적 완성도를 높였지만, 정성 분석에서는 긴 흐름의 의미 연결에 여전히 어려움이 보였습니다.

## 💡 사람 중심의 인사이트
AI가 어려움을 겪는 이유는 용량 부족이 아니라, 사람이 만든 데이터에 본래 들어 있는 모호함과 잡음 때문이라는 생각을 이 프로젝트로 굳혔습니다.
  
## 실행 가이드
이 저장소의 `main.ipynb`를 Colab 또는 로컬 Jupyter 환경에서 실행하여 프로젝트를 확인할 수 있습니다.<br>
main.ipynb 파일에 들어가서 위부터 Shift + Enter만 입력하면 Part-1부터 Part-2까지 모델의 학습, 모델 가중치 저장, test output csv 저장을 진행하도록 jupyter notebook을 작성했습니다.

Part-2에서 사용되는 pronouncing, bert_score를 install하고 시작합니다. (Part-1에서는 활용하지 않습니다.)
해당 주피터 노트북을 모두 실행시키면 각 폴더에 각 모델의 .pt파일(모델 가중치)과 test 데이터에 대한 예측 파일이 각 폴더 내의 predictions 폴더에 .csv 파일 형태로 저장됩니다.

> 현재 저장소에 포함된 결과물은 디버깅 목적으로 임의의 데이터와 하이퍼파라미터로 실험한 산출물입니다.

## Part-1
nlp2025-1_part1 폴더에서 코드 채우기 과제를 수행했습니다.

## Part-2
nlp2025-1_part2 폴더에서 모델의 성능을 높이는 과제를 수행했습니다.
만들어지는 결과물은 다음과 같이 총 7개의 모델(classifier 모델 4개, sonnet 모델 1개, paraphrase 모델 2개)과 4개의 test output이 있습니다.
- Classifier
  - base 모델은 cascade 기법을 적용하지 않은 단일 모델입니다.
  - nlp2025-1_part2/sst-classifier_base.pt
  - nlp2025-1_part2/cfimdb-classifier_base.pt
  - nlp2025-1_part2/sst-classifier_cascade.pt
  - nlp2025-1_part2/cfimdb-classifier_cascade.pt
- Sonnet
  - nlp2025-1_part2/best_sonnet_generation.pt
- Paraphrase
  - Paraphrase 모델은 hard negative priority의 적용 여부에 따라 모델이 저장됩니다. 5epoch를 통해 best 모델을 저장 후 해당 모델을 load하여 hard negative priority를 적용하고 -hnp.pt 모델을 저장합니다.
  - nlp2025-1_part2/{NUM}-1e-5-paraphrase.pt
  - nlp2025-1_part2/{NUM}-1e-5-paraphrase-hnp.pt

모델의 test output은 predictions 폴더에 각각 저장되도록 하였습니다.
