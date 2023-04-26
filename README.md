## GD-MLP (Gated Decomposition MLP)

<b>실생활의 불안정한 변동요인을 고려한 효율적인 장기 시계열 예측 모델 연구</b>

A study of an efficient long-term time series forecasting model considering unstable fluctuation factors in real life.

_______________________

본 연구에서는 실생활의 응용을 위한 유연성과 효율성의 확보를 고려하여 LSTF 모델을 연구하였다.

유연성과 효율성을 확보하기 위해 트랜스포머의 가열된 연구에서 벗어나 기본적인 MLP 구조를 기반으로 한 GD-MLP(Gated Decomposition MLP) 모델을 디자인하였다. 

이 모델은 시계열 요소 분해를 통해 추세와 계절, 나머지 변동들을 각각 MLP에 적용하는데, 이때 게이트를 배치하여 변동들을 선택적으로 학습하는 것이 특징이다.
