# 자연어처리 2025-1 지정주제 기말 프로젝트: GPT-2 구축

## PART-I:

#### 다음 각 모듈에서 누락된 코드 블록을 완성해야 한다.
* `modules/attention.py`
* `modules/gpt2_layer.py`
* `models/gpt2.py`
* `classifier.py`
* `optimizer.py`

#### 다음 모듈들을 실행하여 PART-I의 구현을 테스트한다.

* `optimizer_test.py`: `optimizer.py` 구현을 테스트.
* `sanity_check.py`: GPT 모델 구현을 테스트.
* `classifier.py`: 모델을 사용한 감정 분류 수행.

## PART-II

#### 다음 모듈들을 실행하여 PART-II의 구현을 테스트한다.

* `paraphrase_detection.py`: 패러프레이즈 탐지 수행.
* `sonnet_generation.py`: 소네트 생성 수행.

**주목**: 사용하는 GPU 사양에 따라 batch_size 같은 하이퍼파라미터를 조정하여 성능을 최적화하고 메모리 부족 오류를 방지해야 한다.

#### PART-II 테스트의 핵심 포인트

두 파일에 있는 누락된 코드 블록을 완성하는 것도 중요하지만, PART-II의 핵심은 기능의 확장에 있다. GPT-2 모델을 수정하여 한 문장이 다른 문장의 패러프레이즈인지 판단하는 능력과 소네트를 생성하는 능력을 개선하는 방법에 촛점을 맞추도록 하자.

## 환경 설정
**주목**: .yml 파일의 버전을 변경하지 말것.

conda env create -f env.yml
conda activate nlp_final
