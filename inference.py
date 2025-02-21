import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import numpy as np
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os


def setup_cfg(weights_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # 클래스 수
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 탐지 임계값
    cfg.MODEL.RPN.NMS_THRESH = 0.3  # NMS 임계값
    cfg.MODEL.WEIGHTS = weights_path  # 학습된 모델 가중치 경로
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def predict_image(predictor, image_path, output_path=None):
    # 이미지 읽기
    im = cv2.imread(image_path)
    if im is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # 추론 실행
    outputs = predictor(im)

    # 메타데이터 설정 (클래스 이름 포함)
    metadata = MetadataCatalog.get("my_dataset_train")
    metadata.thing_classes = ["bagel", "croissant", "custardcreambread", "pizzabread",
                              "redbeanbread", "saltbread", "soboro", "whitebread"]

    # 결과 시각화
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = out.get_image()[:, :, ::-1]

    # 결과 저장
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to {output_path}")

    # 탐지 결과 출력
    instances = outputs["instances"].to("cpu")
    for i in range(len(instances)):
        class_id = instances.pred_classes[i].item()
        score = instances.scores[i].item()
        bbox = instances.pred_boxes[i].tensor[0].numpy()
        print(f"Class: {metadata.thing_classes[class_id]}, Score: {score:.2f}")
        print(f"Bbox: {bbox}")

    return result_image


def main():
    # 모델 가중치 경로 설정
    weights_path = "D:\\workspaces\\breadscanso\\cocojson\\cocojson\\model_final.pth"  # 실제 경로로 수정 필요

    # 설정 및 predictor 초기화
    cfg = setup_cfg(weights_path)
    predictor = DefaultPredictor(cfg)

    # 테스트할 이미지가 있는 디렉토리
    test_dir = "D:\\workspaces\\breadscanso\\cocojson\\test"
    output_dir = "D:\\workspaces\\breadscanso\\cocojson\\output"

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 단일 이미지 테스트
    # image_path = "test_image.jpg"
    # output_path = os.path.join(output_dir, "result_" + os.path.basename(image_path))
    # result = predict_image(predictor, image_path, output_path)

    # 디렉토리 내 모든 이미지 테스트

    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, filename)
            output_path = os.path.join(output_dir, "result_" + filename)
            result = predict_image(predictor, image_path, output_path)


if __name__ == "__main__":
    main()