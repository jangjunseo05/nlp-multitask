# nlp
colab에서 https://github.com/hocheol0303/nlp/main.ipynb 파일을 실행하여 프로젝트를 확인할 수 있습니다.<br>
main.ipynb 파일에 들어가서 위부터 Shift + Enter만 입력하면 Part-1부터 Part-2까지 모델의 학습, 모델 가중치 저장, test output csv 저장을 진행하도록 jupyter notebook을 작성했습니다.

Part-2에서 사용되는 pronouncing bert_score를 install하고 시작합니다.(Part-1에서는 활용하지 않습니다.)
해당 주피터 노트북을 모두 실행시키면 각 폴더에 각 모델의 .pt파일(모델 가중치)과 test 데이터에 대한 예측 파일이 각 폴더 내의 predictions 폴더에 .csv 파일 형태로 저장됩니다.

이미 실행되어있는 내용은 디버깅을 위해 임의의 데이터와 임의의 하이퍼파라미터를 설정하여 실험한 결과이니, 부디, 채점에 해당 내용이 사용되지 않았으면 하는 바람입니다!

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

이상으로 9조의 README.md를 마치겠습니다.
한 학기동안 좋은 강의를 해주셔서 감사드립니다!
