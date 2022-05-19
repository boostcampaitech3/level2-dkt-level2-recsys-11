# DKT (Deep Knowledge Tracing)

## 프로젝트 개요

<p align="center"><img src="https://user-images.githubusercontent.com/69205130/169274207-54874e48-80fc-49f7-8874-4e06852ea6fc.png"></p>

KT는 Knowledge Tracing의 약자로, 학생들의 지난 교육 기록을 활용하여, 아직 풀지 않는 문제에 대해, 학생이 그 문제를 맞출지 예측하는 Task입니다.

DKT를 활용하면, 학생 개개인의 학습상태 예측이 가능해지고, 이를 기반으로 학생의 부족한 영역에 대한 문제 추천을 함으로써, 개개인별 맞춤화된 교육을 제공해줄 수 있습니다.

이번 대회에서 Iscream 데이터셋을 이용하여 DKT모델을 구축합니다. 하지만 학생 개개인의 이해도를 가리키는 지식 상태를 예측하는 일보다는, 주어진 문제를 맞출지 틀릴지 예측하는 것에 집중합니다.

각 학생이 푼 문제 리스트와 정답 여부가 담긴 데이터를 받아서, 시험지의 최종 문제를 맞출지, 틀릴지 예측하는 모델을 구축하는 것이 목표입니다.


## 평가 방법
DKT는 주어진 마지막 문제를 맞았는지 틀렸는지 분류하는 이진 분류 문제다. 그래서 평가를 위해 AUROC(Area Under the ROC curve)와 Accuracy를 사용한다.


## Members

|                                                  [김연요](https://github.com/arkdusdyk)                                                   |                                                                          [김진우](https://github.com/Jinu-uu)                                                                           |                                                 [박정훈](https://github.com/iksadNorth)                                                  |                                                                        [이호진](https://github.com/ili0820)                                                                         |                                                                         [최준혁](https://github.com/JHchoiii)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/69205130?s=400&u=a14d779da6a9023a45e60e44072436d356a9461c&v=4)](https://github.com/arkdusdyk) | [![Avatar](https://avatars.githubusercontent.com/u/82719310?v=4)](https://github.com/Jinu-uu) | [![Avatar](https://avatars.githubusercontent.com/u/66674140?v=4)](https://github.com/iksadNorth) | [![Avatar](https://avatars.githubusercontent.com/u/65278309?v=4)](https://github.com/ili0820) | [![Avatar](https://avatars.githubusercontent.com/u/99862931?v=4)](https://github.com/JHchoiii) |


## 코드 구조
```bash
code
├── README.md
├── Ensemble
│  ├── ensemble.py
├── Hojin/dkt
│  ├── args.py
│  ├── inference.py
│  ├── requirements.txt
│  ├── train.py
├── JunHyeok
│  ├── FE_1.ipynb
│  ├── LGBM_FE.ipynb
├── JungHoon/Ensemble
│  ├── soft_voting.py
├── LightGCN(Recbole)
│  ├── LightGCN_(RecBole).py
│  ├── README.md
├── MF
│  ├── MF_NMF.py
│  ├── MF_SVD.py
│  ├── MF_기반_분석_(SVD).ipynb
├── baseline
│  ├── dkt
│  ├── lightgcn
│  ├── README.md
│  ├── iksad_EDA.ipynb
```

## 실험 기록
<p align="center"><img src="https://user-images.githubusercontent.com/69205130/169275163-d0409a1f-58ea-4914-9289-edde4339b4eb.png"></p>


## Modeling
### LightGCN (0.5927) 

기존 DKT에서 쓰이던 방식이 아닌 GNN 모델을 새롭게 적용해보고자 사용했던 모델.

문제의 순서를 고려하지 않은 모델이기에 해당 도메인에서는 큰 성과를 내지 못했다.

### LSTM (0.7198)

위 모델과 달리 순서성를 고려한 모델. 상당히 좋은 성과를 보여준 모델

### LSTM with Attention (0.7181)

모든 문제가 차후의 풀게 될 문제에 영향을 준다기 보다 몇몇 문제가 큰 영향을 줄 것이라는 아이디어에서 사용한 모델

특별히 좋은 성과를 내지 못했다.

### BERT (0.6900)
최근 대부분의 도메인에서 좋은 성과를 낸 Transformer를 사용하고자 사용한 모델.

역시나 LSTM에 비해 좋은 성과를 내지 못했다.

### SAINT (0.7425)
Exercise embedding을 encoder에 Response Embedding을 decoder에 seperately 넣어 사용하는 transformer 기반 모델

자체 성능은 나쁘지 않았지만, 더많은 feature를 exercise embedding으로 추가한다고 점수가 높은 폭으로 상승 하진 않아, hyper parameter tuning 말고 모델을 개선하기엔 어려움이 있었다.

### SAINT+ (0.5453)
Saint에서 elapsed time과 lap time을 추가적으로 사용하여 decoder의 입력으로 사용하고, 조금의 모델구조를 변화한 모델

데이터 특성상, elapsed time 과 lag time을 둘다 구할 수는 없었으므로 점수가 좋지 않게 나온듯 하다. 다른 feature 로 대신하여 학습을 시도해보았지만 여전히 성능이 별로 좋지 않았다.

bert, saint, saint+에서 결과가 그렇게 뛰어나지 않은 것을 보아서는 transformer 기반의 모델의성능이 잘 나오지 않은 것을 볼 수 있었고, 이는 데이터가 생각보다 부족해서가 아닐까하는 추측을 해 볼 수 있었다.

### LGBM (0.7345)
틀린 부분에 가중치를 더하는 Tree 기반 ML 알고리즘으로 DL모델에 적용하기 상대적으로 적은 데이터에 잘 적용될 것을 기대하며 사용한 모델.

custom Feature Engineering을 함으로써 성능을 조금씩 올려 갔다.

**Custom Feature Engineering**
- User , Question features 관련 (UserID, assessmentItemID, testId, KnowledgeTag, Timestamp와 정답과의 관계)
- answerCode와 Value의 sum, mean, std, skew
- Value로부터 얻을 수 있는 분류적인 부분 (day_of_week,grade, last_problem, etc… )
 


## Cross Validation
### K-Fold
validation dataset이 훈련에 활용되지 못하는 점을 보완하기 위해 Kfold 방법을 적용하고자 사용.

데이터를 Train, valid dataset으로 나눌 때, User 단위로 분리가 되게 끔 유도하여 학습했던 유저에 대해 다시 유추하는 Cheating이 일어나지 않게 유도했다.

### 결과

#### SAINT		(전: 0.7409 , 후: 0.7439)

#### LGBM		(전: 0.7631 , 후: 0.7755)


## 앙상블 (Ensemble)
### Oof_stacking (0.7844)
ML모델 성능 향상을 위해 모델마다 하나의 OOF를 생성할 수 있는데 모델들이 생성한 OOF들을 모두 모아서(Stacking), 이 데이터를 활용해 meta learning을 하는 OOF stacking을 시도해 보았다.


### Average of best models (0.7957)
좋은 성능을 보인 모델 결과물들의 prediction 값 average 를 계산해서 저장

위의 모델들 중 좋은 성능을 낸 결과물들을 대상으로 (oof - stacking 포함) 평균을 낸 결과, 성능이 가장 높게 기록되었다.
