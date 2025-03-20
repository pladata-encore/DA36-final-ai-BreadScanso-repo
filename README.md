# DA36기 최종프로젝트 AI(이미지 인식 객체탐지) repo
## KIOSK 자동 주문 시스템 AI 모델

### 📂 관련 레포지토리
- **WEB repo** | https://github.com/pladata-encore/DA36-final-web-BreadScanso-repo
- **AI repo - 이미지 인식** | https://github.com/pladata-encore/DA36-final-ai-BreadScanso-repo
- **AI repo - chatbot** | https://github.com/pladata-encore/DA36-final-chatbot-BreadScanso-repo

---

## 📌 프로젝트 개요
- **목표:** 
   AI 모델을 통해 제품을 **자동으로 인식**하고, 제품명과 가격, 수량이 포함된 주문 목록을 생성하는 서비스 제공
  
## 주요 기능**
**1. 고객이 키오스크에서 빵을 촬영**
  
**2. 객체 탐지 모델이 촬영된 이미지를 분석하여 빵의 종류를 자동으로 분류**
  ![detect](https://github.com/user-attachments/assets/7438a62b-5e47-4b0c-879c-d2a1dc7c4448)  
  
**3. 자동으로 주문 목록 생성**  
![order](https://github.com/user-attachments/assets/a2e0cb51-5bff-40b1-a41d-f942c787b5f4)    

   - **신뢰도가 0.6 미만일 경우, 해당 제품명이 빨간색으로 표시**  
     ![lessred](https://github.com/user-attachments/assets/82886085-23fc-479a-bc89-943bc0e879a5)  

   - **제품이 잘못 분류되었을 경우, 수동으로 제품 추가 가능**  
     ![add](https://github.com/user-attachments/assets/1227f210-43e4-49e3-b789-682516c16380)

---

## 🎞️ 데이터셋
같은 환경에서 각도와 밝기를 조절하며 촬영된 이미지와 동영상을 프레임 단위로 저장하여 활용  

![croissant](https://github.com/user-attachments/assets/b6de4f43-9d53-4910-bf3b-7e276a396ca5)

- **개수**: 1,749장의 이미지파일(.jpg)
- **입력 이미지 크기**: 640x640 픽셀
- **클래스 수**: 8개(슈크림빵, 식빵, 단팥빵, 피자빵, 소금빵, 크루아상, 소보로, 베이글)
- **Train : Validation : Test** = 7:2:1로 분할  
  

### 🎀 데이터 라벨링
Roboflow 프레임워크를 활용하여 데이터 라벨링 및 bounding box, segmentation 데이터 생성

- **Bounding Box**: 객체를 감싸는 사각형을 그려서 해당 객체의 위치를 식별하고 구분하는 방법  
  ![bounding](https://github.com/user-attachments/assets/f83f7893-c665-482f-a2df-f6b2552a31b4)  

- **Segmentation**: 객체의 형태를 더 정확하게 구분하기 위해 픽셀 단위로 객체의 경계를 표시하는 작업  
  ![seg](https://github.com/user-attachments/assets/fd3ef975-754d-4c33-b436-49e770fe985d)

---

## 🤼‍♂️ 모델 성능 비교

본 프로젝트에서는  YOLOv8s, Faster R-CNN, Mask R-CNN 세 가지 모델을 학습 시킨 후 결과를 비교함 

| **모델**         | **속도 (FPS)**  | **Precision** | **Recall** | **F1-score** | **mAP@50** |
|------------------|-----------------|---------------|------------|---------------|-----------|
| **YOLOv8s**      | 50 FPS          | 0.9953         | 1.0000       | 0.9976          | 0.9950      |
| **Faster R-CNN** | 10 FPS          | 0.9934         | 1.0000       | 0.9967          | 0.9934      |
| **Mask R-CNN**   | 8 FPS           | 0.7596         | 0.7596       | 0.7596          | 0.9260      |


### 성능 지표 설명
- **속도 (FPS)**: 초당 처리할 수 있는 프레임 수로, 모델이 얼마나 빠르게 작업을 처리할 수 있는지 나타냄
- **F1-score**: 모델의 성능을 평가하는 지표로, **정밀도(Precision)**와 **재현율(Recall)**의 균형을 보여줌
- **Recall (재현율)**: 실제 존재하는 객체 중 모델이 얼마나 잘 찾아냈는지를 나타냄
- **Precision (정밀도)**: 모델이 탐지한 객체 중 실제로 맞는 비율을 나타냄
- **mAP (mean Average Precision)**: 여러 클래스에 대해 모델의 전체 성능을 평가하는 지표로, 모델이 모든 클래스에서 얼마나 정확하게 예측했는지를 나타냄

### 결과
임베디드 시스템에 최적화된 속도와 가벼운 실행을 고려하여 **YOLOv8s**를 최종모델로 선정

---

## 🦴 구조 설계
![architecture](https://github.com/user-attachments/assets/7e18fc7e-592d-42ef-97d5-a769721f9157)

1. **이미지 촬영 및 전송 (KIOSK → 추론 서버)**  
   사용자가 KIOSK에서 촬영한 제품 이미지는 객체 탐지 모델이 실행되고 있는 추론 서버로 전송됨

2. **객체 탐지 (추론 서버 - YOLO 모델)**  
   추론 서버의 **YOLO 모델**은 입력된 이미지를 처리하여 빵의 위치를 식별하고, 각 객체에 대해 **bounding box**와 **라벨 값**을 예측

3. **결과 전달 및 출력 (추론 서버 → 웹 서버 → 사용자)**  
   예측된 결과는 웹 서버로 전달되며, 웹 서버는 사용자가 해당 결과를 확인할 수 있도록 화면에 출력함.

---

## 🛠 기술 스택
- **모델 및 알고리즘:** YOLOv8, Faster R-CNN, Mask R-CNN
- **데이터 처리 및 라벨링**: Roboflow, OpenCV
- **프레임워크 및 라이브러리**: Python, TensorFlow / PyTorch
- **클라우드:** AWS (EC2, EB)  
- **시스템 및 배포**: Flask, Docker

---

## 🪸 하이퍼파라미터 및 데이터 증강 기법

**하이퍼파라미터**
- epochs: 50, batch: 16, lr0: 0.001
- mosaic, mixup: True
- degrees: 10, scale: 0.5, shear: 10
- flipud, fliplr: 0.5, patience: 0

**데이터 증강 기법**
- Mosaic, Mixup: 이미지 합성 및 섞기
- 회전, 크기 조정, 왜곡, 뒤집기: 다양한 변형 학습

---

## 💡 개선 방안
- **Model 변경/튜닝**
  - Mask R-CNN 등 2단계 탐지 모델을 사용하여 더 정교한 객체 분류
  - 하이퍼파라미터 튜닝을 통해 모델 성능 최적화
  - 로깅 시스템을 도입하여 추론 데이터를 실시간으로 모니터링 및 분석
 
- **데이터 보강**
  - 학습용 데이터셋를 추가하여 모델의 일반화 능력 향상
  - 여러 빵이 겹쳐 있는 상황을 반영한 데이터 추가
  - 이미지 증강 및 전처리 과정 개선을 통해 다양한 환경에 대응할 수 있도록 모델 강화

---

## 👥 참여자 및 기여 활동
- 강한결  
- 전민하  
