import cv2
from ultralytics import YOLO



# 모델 로드 (전역 변수로 한 번만 로드)
try:
    model = YOLO("/app/model/yolov8s_34.pt")
except Exception as e:
    raise RuntimeError(f"YOLO 모델 로드 실패: {str(e)}")


def predict(image_path: str):
    """
    YOLO 모델을 사용한 객체 감지 함수

    Args:
        image_path (str): 이미지 파일 경로

    Returns:
        list: 감지된 객체 리스트
    """
    try:
        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("이미지를 읽을 수 없습니다")

        # YOLO 추론 실행
        results = model(image)[0]
        prediction = []

        # 결과 처리
        if hasattr(results, "boxes") and results.boxes is not None:
            for box in results.boxes.data.tolist():
                x_min, y_min, x_max, y_max, confidence, class_id = box
                class_id = int(class_id)

                prediction.append({
                    "class": model.names[class_id] if class_id < len(model.names) else "Unknown",
                    "confidence": float(confidence),
                    "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)]
                })

        return prediction

    except Exception as e:
        raise Exception(f"예측 중 오류 발생: {str(e)}")