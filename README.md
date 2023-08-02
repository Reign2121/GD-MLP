## GD-MLP (Gated Decomposition MLP) 

<b> 실생활의 불안정한 환경에 적합한 효율적인 장기 시계열 예측 모델 연구 </b>

A study of an efficient long-term time series forecasting model considering unstable fluctuation factors in real life.


<br> Basic Architecture </br>

<img width="896" alt="image" src="https://github.com/Reign2121/GD-MLP/assets/121419113/6272fe4f-c947-465e-bb79-891823d0a0fe">


모델 기본 구조 출처) D-Linear https://github.com/cure-lab/LTSF-Linear


_______________________

Background IDEA:

- 실생활의 환경은 과거의 스케일, 변동과 전혀 다른 양상을 보일 수 있는 매우 불안정한 환경이다. (EX. COVID-19으로 인한 매출 급감 등)

- 실생활의 시계열은 매우 복잡한 패턴으로 얽혀있다. 


Solution IDEA: 

- 현실세계의 불안정한 환경에 대응하기 위해 적절한 trend와 그에 따른 scale(locality)을 잘 포착하도록 하자.

- 복잡한 변동을 분해(Decomposition)하여 각 변동 요소의 명확한 패턴을 학습하고, 각 요소의 영향력을 가중치로 환산하여 반영하자.


Previous works:

- 트랜스포머는 Temporal Relation, Trend를 적절하게 포착하지 못한다. (어텐션 매커니즘이 순서와 무관하게 동작한다.)

- 최근에는 하나의 Layer, MLP 구조의 단순한 모델이 트랜스포머의 성능을 뛰어넘는 결과를 보이고 있다.

- 트랜스포머 변형 모델들의 주장과 달리, 여전히 메모리와 시간의 비효율성이 개선되지 못했다.

_______________________

이에 위와 같은 아이디어를 구체화하여 시계열의 변동들을 효과적으로 파악하는 모델을 디자인한다. 

GD-MLP

본 연구에서는 실생활의 특성을 고려한 LSTF 모델을 연구하였다.

이를 위해 최근의 연구 결과를 바탕으로 트랜스포머 구조가 아닌 MLP 구조를 기반으로 한 GD-MLP(Gated Decomposition MLP) 모델을 디자인하였다. 

이 모델의 핵심은 두 가지 Gate로, 
Input Gate로 trend 변동과 residual 변동이 서로의 영향력을 나눠가지도록 하며, Output Gate로 그 영향력을 추가적으로 반영한다. 

________________________

Data

유동인구의 변동
![image](https://github.com/Reign2121/GD-MLP/assets/121419113/4d6b3a73-411e-4968-a338-ad7ba7a4db2c)

매출건수의 변동 
![image](https://github.com/Reign2121/GD-MLP/assets/121419113/5b9fa967-178b-4550-8744-5b3e0875e7dd)

데이터: 벤치마크 데이터 + 서대문구 상권 데이터 (2020년) (target: 유동인구, 매출 건수)

서대문구 상권 데이터, 코로나로 인해 매출의 추세가 매우 불안정했던 2020년의 시계열을 예측한다. 약 (2200개 포인트)


Experiment (🔥 진행 중 )

- Univariate 예측

- Multivariate 예측

- 파라미터 최적화

- Gate, MLP 활성화 함수에 따른 파라미터 최적화

- Ma의 커널 사이즈, 은닉층 노드 수 등 하이퍼 파라미터 최적화

- 로컬 회귀로 추세 변동 추출하기

https://reign2121.notion.site/GD-MLP-e6154d388eb14c1bbdc5f2bfdd0ecfb0 (실험일지 참조)

