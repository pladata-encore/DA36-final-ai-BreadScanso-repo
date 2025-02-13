# !pip install ultralytics
# !pip install opencv-python matplotlib

from ultralytics import YOLO
from collections import Counter
import matplotlib.pyplot as plt
import cv2

model=YOLO('detect_model.pt')
image_path='test4.jpg'

class_mapping = {"bagel": "베이글", "croissant": "크루아상", "custardcreambread": "커스타드크림빵", "pizzabread": "피자빵",
                 "redbeanbread": "팥빵", "saltbread": "소금빵", "soboro": "소보로빵", "whitebread": "식빵"
}

result = model.predict(image_path)[0]
detected_classes = result.boxes.cls.tolist()    # 탐지한 객체의 클래스를 리스트 형식으로 반환 →[4.0, 5.0, 2.0, 2.0, 1.0]

detected_breads = [class_mapping[result.names[int(cls_id)]] for cls_id in detected_classes]
bread_counts = Counter(detected_breads)    # 각 빵의 개수 계산

result.show()

print("\n🥐구매하시는 빵은 다음과 같습니다.🥐\n")
print('빵  〰️〰️  수량')
for bread, count in bread_counts.items():
    print(f"- {bread}: {count}개")

