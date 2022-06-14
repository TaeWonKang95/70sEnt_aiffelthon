# 실시간 쿠폰 발행을 위한 oo시 구군단위 시간별 수요예측
※본 프로젝트는 모두의 연구소 aiffel 쏘카 2기 활동의 일환으로 이뤄졌으며, 쏘카의 실제 데이터를 기반으로 진행하였습니다. 따라서 보안 협약에 의해 데이터를 비식별 처리 하였음을 알려드립니다.   
### 팀 이름
  - 70sEnt
### 프로젝트 기간 
  - 2022.04.18 ~ 2022.06.09
### 프로젝트원 
  - 김동영, 강태원, 한승민


## 문제 인식과 솔루션
- 전국의 각 지역은 각각 다양한 특성 존재
	- 미시적 수요 예측을 위해서는 지역 선정이 필요
- 연구 지역의 문제점
  - 연구 지역, 쏘카 주 수요층인 특정 연령대의 유출이 심각
- 솔루션
  - 구·군별 수요 예측 기반의 실시간 할인 쿠폰 발행을 통한 수요 촉진
  - 매출 하락 시점을 예측해 쿠폰 발급으로 수요 촉진

## Target 설정 및 EDA
- Target 설정
	- 실시간 쿠폰 발행을 위해 수요 예측 단위를 시간 단위로 함
	- '시간대별 운행 대수'를 수요를 나타내는 지표로 선정함
- 대표 EDA 및 feature
  - 대학 학사일정은 쏘카 수요에 영향을 미침
  - 자동차 보유율과 쏘카 이용율의 상관관계를 확인
  - 시간대별 온도, 습도와 같은 날씨는 쏘카 수요와 높은 상관관계를 가짐

  <table>
  <tr><td width=33%>
  <img src="https://user-images.githubusercontent.com/33904461/173379721-b2e2723c-32ba-4ef7-8762-a1bcb810fa8d.png"></td><td width=33%><img src="https://user-images.githubusercontent.com/33904461/173381100-e751670c-0090-40de-9ce1-9b677b3c43d4.png"></td><td width=33%><img src="https://user-images.githubusercontent.com/33904461/173381779-8e368ea8-868f-4e69-9247-b729f5b024ce.png"></td></tr></table>   
       
       
## Model  
- 모델 선정 기준은 Time series와 지역 특징을 표현할 수 있는 Static variable 학습 여부로 함
- Baseline Model
	- Bi-Directional LSTM + Single Layered Perceptron Model
	- Ablation Study를 통해 모델이 Static variable을 충분히 학습하지 못한다는 점 파악함
- TFT (Temporal Fusion Transformer)
  - ‘현재까지 관측된 값’, ‘미래에도 변하지 않는 값’, ‘미래 시점에도 알 수 있는 값’을 지정하여 다른 시점의 feature를 학습 가능
  - 고정 변수를 VSN, LSTM 인코더, GRN을 통해 충분한 정보를 전체 레이어에 전달 가능함
- Baseline Model과 TFT 성능 비교
  - TFT에서 feature의 학습과 표현력이 높아져 Baseline Model보다 높은 성능을 보임

  <table>
  <tr height = 10%>
  <img src="https://user-images.githubusercontent.com/33904461/173383745-473a93e4-5d2d-498b-8f45-284416fb0b73.png"></tr><tr height = 10% ><img src="https://user-images.githubusercontent.com/33904461/173384360-11377c5f-df63-4a1b-822f-8ed35d5f38f5.png"></tr></table>   

## 실시간 쿠폰 발행 과정   
- Slack을 통한 쿠폰 발행 시점 자동 알림
	- Python과 Airflow를 활용하여 자동화
- Tableau를 통해 마케팅 담당자에게 대시보드 전달
	- 다양한 지표에서의 현황 및 예측 값 직관적 파악 가능
	- 업무 자동화 프로세스 접목 가능
- Airflow를 통한 Pipeline 자동화
	- 전체 프로세스 자동화 구축으로 업무 
     편의성 증대
<table>
  <tr><td width=33%>
  <img src="https://user-images.githubusercontent.com/33904461/173385713-3fd44071-1eca-421a-9afc-6d091b1751b3.png"></td><td width=33%><img src="https://user-images.githubusercontent.com/33904461/173472107-a14c9438-a2fa-4ccf-852f-05a1c54a9dc0.png"></td><td width=33%><img src="https://user-images.githubusercontent.com/33904461/173388238-93108584-838e-48c9-812d-b9026d5cc136.png"></td></tr></table>    
  
## 향후 계획    
- A/B 테스트를 통한 쿠폰 효과 검증
- 타 지역에 동일 모델 적용 및 검증



