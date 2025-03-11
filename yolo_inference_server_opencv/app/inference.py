import torch
import numpy as np
import cv2
from typing import Tuple, List, Dict


class YOLOPyTorch:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # PyTorch 모델 로드
        self.model = self.load_model(model_path)
        self.model.eval()  # 평가 모드로 설정
        self.model.to(self.device)

    def load_model(self, model_path: str):
        # PyTorch 모델 파일 로드
        model = torch.load(model_path, map_location=self.device)

        # 모델이 state_dict 형태로 저장되었을 경우를 처리
        if isinstance(model, dict):
            if 'model' in model:
                model = model['model']
            elif 'state_dict' in model:
                model = model['state_dict']

        return model

    def preprocess_image(self, img: np.ndarray, input_size: Tuple[int, int] = (640, 640)):
        # 이미지 크기 조정
        img = cv2.resize(img, input_size)

        # BGR to RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)

        # 정규화 및 텐서 변환
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img.to(self.device)

    @torch.no_grad()
    def detect(self, image: np.ndarray, conf_threshold: float = 0.25,
               iou_threshold: float = 0.45) -> List[Dict]:
        # 이미지 전처리
        img = self.preprocess_image(image)

        # 추론 실행
        predictions = self.model(img)

        # predictions가 튜플이나 리스트인 경우 첫 번째 요소 사용
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        # Non-maximum suppression 적용
        if hasattr(self.model, 'non_max_suppression'):
            # YOLOv5/v8 스타일
            predictions = self.model.non_max_suppression(
                predictions,
                conf_thres=conf_threshold,
                iou_thres=iou_threshold
            )[0]
        else:
            # 일반적인 NMS 적용
            predictions = self.apply_nms(predictions, conf_threshold, iou_threshold)

        # 원본 이미지 크기로 좌표 변환
        original_height, original_width = image.shape[:2]
        predictions[:, [0, 2]] *= original_width / 640
        predictions[:, [1, 3]] *= original_height / 640

        # 결과 포맷팅
        detections = []
        for pred in predictions.cpu().numpy():
            x1, y1, x2, y2, conf, cls = pred
            detection = {
                'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],  # [x, y, width, height]
                'confidence': float(conf),
                'class_id': int(cls)
            }
            detections.append(detection)

        return detections

    def apply_nms(self, predictions, conf_threshold, iou_threshold):
        # confidence threshold 적용
        mask = predictions[..., 4] > conf_threshold
        predictions = predictions[mask]

        if len(predictions) == 0:
            return torch.zeros((0, 6))

        # NMS 적용
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        labels = predictions[:, 5]

        indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)

        return torch.cat((boxes[indices], scores[indices].unsqueeze(1),
                          labels[indices].unsqueeze(1)), dim=1)


# 모델 로드
model = YOLOPyTorch("D:\\workspaces\\ai_breadscanso\\yolo_inference_server_opencv\\model\\yolov8s_34.pt")

# 이미지 로드
image = cv2.imread("D:\\workspaces\\ai_breadscanso\\test4.jpg")

# 객체 감지 수행
detections = model.detect(
    image,
    conf_threshold=0.25,
    iou_threshold=0.45
)

# 결과 시각화
for det in detections:
    bbox = det['bbox']
    conf = det['confidence']
    class_id = det['class_id']

    # 박스 그리기
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)