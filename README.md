영상의 씬스틸러 검출

## 🖥️ 프로젝트 소개
[아래](https://please-amend.tistory.com/260)의 블로그에서 씬스틸러를 검출하는 내용을 보고 시작<br>
해당 블로그는 영상의 씬스틸러를 랜덤으로 만든 시선데이터로 검출<br>
개인적인 생각은 "영상의 객체 움직임 -> 시선이 쏠림 -> 시선 데이터" 로 생각<br>
따라서 이미지에 움직인 객체가 해당 이미지의 씬스틸러로 간주하여 해당 객체 추출<br>
https://please-amend.tistory.com/260<br>

## 🕰️ 개발 기간
* 24.01.10일 ~ 24.01.18일 ※ 1차 완료

### ⚙️ 개발 환경
- `python`
- **IDE** : Pycharm

## 📌 데이터 분석
#### 영상데이터 분리 및 프레임(이미지) 추출
- 영상은 애니매이션 데이터 사용

![poppyes (1)](https://github.com/whcjfdudwkd/scene_stealer_detect/assets/70883264/5c89dc10-bfa8-4eeb-a993-501b27b69fa0)


- 영상의 데이터를 씬 단위로 분리
- 씬으로 분리된 데이터를 다시 프레임 단위로 분리
- 프레임 단위 이미지에서 씬스틸러 검출

## 🌏 씬스틸러 검출
### 방법 1
#### 이미지에서 객체를 추출(1)하고 움직인 객체를 추출(2)하여 병합
- 객체 추출 모델은 YOLOv8n 모델을 사용
- YOLOv8n을 사용하여 객체를 추출(1)
- 움직인 객체는 현재 프레임과 이후 프레임의 차를 이용하여 검출(2)
- 현재 프레임과 이후 프레임을 gray-scale변환
- cv2.absdiff함수를 사용하여 두 프레임의 차를 검출(차프레임)
- cv2.threshold를 사용하여 차프레임을 이진화
- 여기서 사용한 thresh hold값은 임의로 설정(30)
- 방법 1에서 추출한 객체를 이미지에 그려본 결과 아래의 이미지와 같은 결과가 발생

![frame_0000 png](https://github.com/whcjfdudwkd/scene_stealer_detect/assets/70883264/942439bd-bded-4717-8cc8-096b8b335ee4)
 <br>-> 초록색 박스는 객체 / 파랑색 박스는 움직인 객체
 
- YOLOv8n모델에서 검출이 안된 객체가 존재 -> 씬스틸러여도 검출을 못함
- YOLOv8n모델모다 더 많이 학습된 YOLOv8l모델을 사용해도 같은 문제가 발생할거로 생각됨
- 만약 YOLOv8l모델이 객체를 많이 추출해도 어떤 객체가 씬스틸러인지 구별 불가
- 여러가지 이유로 방법 1은 사용 불가

### 방법 2
#### "움직인 객체만 가지고 이용 할수는 없나??"의 아이디어에서 시작
 - YOLO모델을 사용하지않고 객체의 움직임이 있는 것만 추출
 - cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE를 사용하여 외곽을 그릴 수 있는 포인트만 추출
 - cv2.contourArea를 통해 면적이 100이상인 것만 추출(너무 작은 데이터 제거)
 - cv2.boundingRect를 사용하여 바운딩 박스 검출 및 좌표 저장
 - 검출된 바운딩 박스 중 좌상단 포인트를 이용하여 DBSCAN을 사용하여 클러스터링
   <br>-> 조금이라도 움직임이 있으면 검출 되기 때문
   <br>-> 프레임의 핵심 씬스틸러를 검출하기 위해 사용
   <br>-> 바운딩 박스의 좌표가 어떻게 분포될지 모르기 때문에 K-means 사용 불가
 - 클러스터링 된 포인트 중 가장 많은 클러스터링된 포인트 집합을 생성
 - 검출된 포인트의 바운딩 박스를 모두 감싸는 새로운 바운딩 박스 생성
 - 해당 바운딩 박스가 해당 프레임의 씬스틸러
 - 방법 2를 사용한 결과는 아래의 이미지와 같다

![frame_0004](https://github.com/whcjfdudwkd/scene_stealer_detect/assets/70883264/264f768e-a6c4-4c3f-992f-d5d34fa17ea6)


## 🐃 평가
 - 해당방법을 이용하니 각 프레임 마다의 씬스틸러를 찾을 수 있었다
 - 객체의 움직임에 따라 씬스틸러가 변하는것도 검출 할 수 있었다.
   <br>-> 두객체가 동시에 움직일 경우
   <br>-> 한 객체에서 한 객체로 움직임이 변화한 경우
 - 2개의 집단에서 클러스터링된 포인트 갯수가 같은 경우 씬스틸러의 박스가 2개가 생기는 문제 발생
 - 가끔 프레임에서 움직임이 검출안되는 문제 발생

## ♻️ 추후사항
 - 2개의 집단에서 클러스터링된 포인트 갯수가 같은 경우 2개의 집단의 모든 바운딩 박스를 포함하는 박스를 만들 필요가 존재
 - 가끔 프레임에서 움직임이 검출안되는 문제를 확인하여 수정 필요

   
