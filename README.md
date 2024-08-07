# 2022 DACON 제주도 도로 교통량 예측 AI 경진대회
[2022 DACON 제주도 도로 교통량 예측 AI 경진대회](https://dacon.io/competitions/official/235985/overview/description)

# 1. 대회 정보
1. 대회명

    제주 테크노파크 도로 교통량 예측 AI 경진대회


2. 배경

   제주도내 주민등록인구는 2022년 기준 약 68만 명으로, 연평균 1.3%정도 매년 증가하고 있습니다. 또한 외국인과 관광객까지 고려하면 전체 상주인구는 90만 명을 넘을 것으로 추정되며, 제주도민 증가와 외국인의 증가로 현재 제주도의 교통체증이 심각한 문제로 떠오르고 있습니다.


3. 주제

    제주도 도로 교통량 예측 AI 알고리즘 개발
    

4. 규칙
   - 평가 산식 : MAE
   - Public score : 전체 테스트 데이터 중 30%
   - Private score : 전체 테스트 데이터 중 70%
   - 평가 방식
        - 1차 평가 : 리더보드 Private score
        - 2차 평가 : Private Score 상위 10팀 코드 및 PPT 제출 후 코드 평가
   - 외부 데이터
        - 사용에 법적 제약이 없고 누가나 사용할 수 있는 **외부 데이터 사용 가능**
        - 법적 제약이 없으며 **공개된 사전 학습 모델(Pre-trained Model)** 사용 가능
   
# 2. 개발 환경

[Prepared CUDA based Image for this project]()

```
$ docker pull lotee/road-traffic
```
[국토교통부 교통체계](https://www.law.go.kr/LSW/admRulInfoP.do?admRulSeq=2100000157569)
