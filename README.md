# 2022 DACON 제주도 도로 교통량 예측 AI 경진대회
[2022 DACON 제주도 도로 교통량 예측 AI 경진대회](https://dacon.io/competitions/official/235985/overview/description)

# 1. 대회 정보
1. 대회명

    제주 테크노파크 도로 교통량 예측 AI 경진대회


2. 배경

   제주도내 주민등록인구는 2022년 기준 약 68만 명으로, 연평균 1.3%정도 매년 증가하고 있습니다. 또한 외국인과 관광객까지 고려하면 전체 상주인구는 90만 명을 넘을 것으로 추정되며, 제주도민 증가와 외국인의 증가로 현재 제주도의 교통체증이 심각한 문제로 떠오르고 있습니다.


3. 주제

    제주도 도로 교통량 예측 AI 알고리즘 개발


4. 일정
   - 대회 기간 : 2022년 10월 3일 10:00 ~ 2022년 11월 14일 10:00
   - 대회 종료 : 2022년 11월 14일 10:00
   - 코드 제출 마감 : 2022년 11월 17일 23:59
   - 코드 평가 : 2022년 11월 18일 00:00 ~2022년 11월 25일 23:59
   - 최종 수상자 발표 : 2022년 11월 28일 10:00(예정)
   - 시상식 : 11월 30일(예정)
    

5. 규칙
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

# 3. 데이터

```
id : 샘플 별 아이디(ex: TRAIN_0000001)
base_date : 날짜(ex: 20220623)
day_of_week : 요일
base_hour : 시간대(00시 ~ 23시)
lane_count : 차로수(차선 개수)
road_rating : 도로등급(국토교통부 교통체계 링크 참조. 101부터 107까지 총 7개 분류)
multi_linked : 중용구간(2개 이상의 노선이 도로의 일정 구간을 공동으로 사용) 여부
connect_code : 연결로 코드(국토교통부 교통체계 링크 참조. 000 ~ 108 총 9개 분류)
maximum_speed_limit : 최고속도제한(단위: km/h)
weight_restricted : 통과제한하중
height_restricted : 통과제한높이
road_type : 도로유형(국토교통부 교통체계 링크 참조. 000 ~ 004)
start_latitude : 시작지점 위도
start_longitude : 시작지점 경도
end_latitude : 도착지점 위도
end_longitude : 도착지점 경도
start_turn_restricted : 시작지점 회전제한 유무(ex: 있음, 없음)
end_turn_restricted: 도착지점 회전제한 유무(ex: 있음, 없음)
road_name : 도로명
start_node_name : 시작지점명
end_node_name : 도착지점명
vehicle_restricted : 통과제한차량
target : 도로 위 차량의 평균속도(단위 : km/h)
```
[국토교통부 교통체계](https://www.law.go.kr/LSW/admRulInfoP.do?admRulSeq=2100000157569)