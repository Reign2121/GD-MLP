## GD-MLP (Gated Decomposition MLP) (making, unfinished)

<b>실생활의 불안정한 변동요인을 고려한 효율적인 장기 시계열 예측 모델 연구</b>

A study of an efficient long-term time series forecasting model considering unstable fluctuation factors in real life.

<img width="753" alt="image" src="https://github.com/Reign2121/GD-MLP/assets/121419113/ad33dce1-8528-4484-adb9-7539444a5f69">


모델 기본 구조 출처) D-Linear https://github.com/cure-lab/LTSF-Linear


_______________________

IDEA:

비즈니스와 금융 도메인 등 실생활의 많은 도메인들은 외부 요인에 따라 추세가 매우 탄력적인 경우가 많다. (EX. COVID-19으로 인한 매출 급감 등)

이에 실생활의 불안정한 변동을 포착하기 위해서는 추세를 유연하게 파착하는 것이 매우 중요하다. (과적합을 경계)

그러나 트랜스포머는 과대 적합 문제와 함께 효율성 측면에서 실생활의 응용에 부적합하다.

최근에는 하나의 Layer, MLP 구조의 모델이 트랜스포머의 성능을 뛰어넘는 결과를 보이고 있다.

이에 위와 같은 아이디어를 구체화하여 시계열의 변동들을 유연하게 파악하는 모델을 디자인한다. 

_______________________

GD-MLP

본 연구에서는 실생활의 응용을 위한 유연성과 효율성의 확보를 고려하여 LSTF 모델을 연구하였다.

유연성과 효율성을 확보하기 위해 트랜스포머의 가열된 연구에서 벗어나 기본적인 MLP 구조를 기반으로 한 GD-MLP(Gated Decomposition MLP) 모델을 디자인하였다. 

이 모델은 시계열 요소 분해를 통해 추세와 계절, 나머지 변동들을 각각 MLP에 적용하는데, 이때 게이트를 통해 변동들의 영향력을 가중치로 학습하는 것이 특징이다.

________________________

Experiment

데이터: 서대문구 소상공인 데이터 (2020년)

코로나로 인해 매출의 추세가 매우 불안정했던 2020년의 시계열을 예측한다.
