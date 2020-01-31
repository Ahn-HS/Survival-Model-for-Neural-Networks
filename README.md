# Deep Learning-based survival analysis for prediction of the probability of disease with NHIS cohort

국민건강보험 건강검진 코호트 데이터를 이용한 질병 예측 딥러닝 모델 개발


## 개발 목표 및 서론
The objective of this study is to demonstrate the improved accuracy of the deep learning models in identifying high risk individuals by comparing the performance of a Cox hazard regression model.

기존 의학 통계 분야에서 생존 분석은 일정 시점에 대한 이벤트 발생 예측을 말한다.
일반적으로 사람이 죽는지 안죽는지에 간단한 예측과 달리
생존 분석에서는 대표적으로 발병 시점과 발병 확률 두 결과에 대한 예측이 필요하고
대표적으로 COX 모델이 있다.

this study attempts to evaluate the discriminative accuracy of a deep learning based algorithm with repeated health data for prediction by comparing the result of conventional Cox hazard regression analysis. The forecasts for two models were calculated for a specific time period through multi-class classification.

본 연구에서는 전통적인 통계 기반 모델보다 우수한 성능의 딥러닝 기반 모델을 연구하였다.

의료 분야에서 이러한 분석의 특성상 현재 생존하고 있는 환자의 관측되지 않은 데이터(censored)와
특정 시점에서 관측된 데이터(non-censored)를 별도로 고려하여야 하기 때문에 일반적인 딥러닝 모델을
적용하는데 어려움이 있다.

따라서 이들의 가중치를 적절하게 반영할 수 있는 비용함수를 제안하고, 성능을 평가하였다.


## 학습 모델 설정
모델
```
	┗ 은닉층 개수 : 3개, 차원 : 7차원 설정
	┗ 활성 함수 = relu
	┗ 활성 함수 = sigmoid
	┗ 반복 학습 횟수 = 100,000 번
	┗ 비용 함수 = modified-likelihood
	┗ 정규화 함수 = L2 regularization
	┗ duration interval = 20일
	┗ backpropagation 최적화 함수 = RMSprop
	┗ dropout rate = 0.5
	┗ batch size = 72
```
평가지표
```
	┗ accuracy
	┗ precision
	┗ recall
	┗ f1score
	┗ RMSE
	┗ AUROC
	┗ C-index
```


## 데이터
데이터 총 건수 : 467,705 건
```
-training : testing = 8 : 2
-training: 374,166 건
-testing: 93,541 건
```

