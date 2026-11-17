# Multi-task NLP with GPT-2: Bridging the Gap between Model and Human Expectation

## 📌 Project Overview
This project explores the boundaries of **Human-AI Interaction** by implementing a multi-task NLP system using GPT-2. The focus was on bridging the gap between model capacity and the inherent ambiguity of human-generated data across three tasks: Sentiment Classification, Paraphrase Detection, and Sonnet Generation.

## 🛠️ Key Research Points & Decisions

### 1️⃣ Sentiment Classification: Overcoming Imbalance
- **Strategy:** Implemented a **Coarse-to-Fine Cascade model** (2-stage & 3-stage) and data rephrasing techniques.
- **Research Decision:** After empirical evaluation, I opted for the **2-stage model** as the final architecture.My analysis showed that deeper cascades (3-stage) suffered from **error propagation**, proving that model complexity must be balanced with training stability in fine-grained tasks.

### 2️⃣ Paraphrase Detection: Efficiency & Precision 
- **Strategy:** Applied **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning and **Hard Negative Priority (HNP)** training.
- **Research Decision:** Analyzed the **trade-off between confidence and noise** in HNP. While HNP targets difficult samples, I discovered that training on high-confidence incorrect samples can sometimes act as noise, leading to a refined strategy of merging hard negatives with original training data.

### 3️⃣ Sonnet Generation: Aligning with Human Constraints 
- **Strategy:** Developed a Shakespearean sonnet generator incorporating a custom **Rhyme Loss** based on the cosine similarity of end-word embeddings.
- **Research Decision:** Implemented a 14-line post-processing logic to ensure structural fidelity. This resulted in a **significant improvement in Rhyme Accuracy (0.00 → 0.12)** without degrading semantic coherence (BERTScore), demonstrating the efficacy of enforcing poetic constraints in generative models.

## 📊 Results & Honest Evaluation
* **Sentiment:** While rephrasing improved the F1-score (0.41 → 0.44), the Cascade structure revealed a bottleneck in the coarse classifier's accuracy.
* **Paraphrase:** Baseline GPT-2 (5 epoch) remained more stable than HNP-only tuning, highlighting the sensitivity of confidence thresholds in sample selection.
* **Sonnet:** Successfully reduced Perplexity (35.68 → 34.15) and improved structural completedness, though qualitative analysis still shows challenges in long-term semantic flow.

## 💡 Human-centered Insights
This project reinforced my perspective that AI struggles not due to capacity, but due to the inherent ambiguity and noise in human-generated data.
  
## Guide
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
