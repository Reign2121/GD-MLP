## GD-MLP (Gated Decomposition MLP) (making, unfinished)

<b>실생활의 불안정한 변동요인을 고려한 효율적인 장기 시계열 예측 모델 연구</b>

A study of an efficient long-term time series forecasting model considering unstable fluctuation factors in real life.

<img width="753" alt="image" src="https://github.com/Reign2121/GD-MLP/assets/121419113/ad33dce1-8528-4484-adb9-7539444a5f69">


모델 기본 구조 출처) D-Linear https://github.com/cure-lab/LTSF-Linear


_______________________

Background IDEA:

- 실생활의 시계열은 도메인에 따라 각각이 복잡한 패턴으로 얽혀있다. (도메인에 따라 지배적인 변동이 알려져 있기도 하다.)

- 실생활의 환경은 과거의 스케일, 변동과 전혀 다른 양상을 보일 수 있는 매우 불안정한 환경이다. (EX. COVID-19으로 인한 매출 급감 등)


Solution IDEA: 

- 복잡한 변동을 분해하여 명확한 패턴을 학습하고, 각 변동의 영향력을 가중치로 환산하여 반영하자.

- 불안정한 환경에 대응하기 위해 예측시점의 Locality를 효과적으로 포착하고 반영해야 한다.


Previous works:

- 트랜스포머는 Locality를 반영하지 못한다. (어텐션 매커니즘이 순서와 무관하게 동작하기 때문에)

- 트랜스포머 변형 모델들의 주장과 달리, 여전히 메모리와 시간의 비효율성이 개선되지 못했다.

- 최근에는 하나의 Layer, MLP 구조의 단순한 모델이 트랜스포머의 성능을 뛰어넘는 결과를 보이고 있다.

_______________________

이에 위와 같은 아이디어를 구체화하여 시계열의 변동들을 유연하게 파악하는 모델을 디자인한다. 

GD-MLP

본 연구에서는 실생활의 특성을 고려한 LSTF 모델을 연구하였다.

이를 위해 최근의 연구 결과를 바탕으로 트랜스포머 구조가 아닌 MLP 구조를 기반으로 한 GD-MLP(Gated Decomposition MLP) 모델을 디자인하였다. 

이 모델의 핵심은 두 가지 Gate로, 
Input Gate로 인풋 시리즈의 Locality를 반영하며, Output Gate로 분해된 변동의 영향력을 반영한다. 

________________________

Experiment

데이터: 벤치마크 데이터 + 서대문구 상권 데이터 (2020년)

서대문구 상권 데이터, 코로나로 인해 매출의 추세가 매우 불안정했던 2020년의 시계열을 예측한다.
