import torch
from ultralytics import YOLO
from fastapi import FastAPI
import uvicorn
from pyngrok import ngrok
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Conda 환경 변수 가져오기
conda_env = os.getenv("CONDA_ENV_NAME")
ngrok_token = os.getenv("NGROK_AUTH_TOKEN")

# Conda 가상환경 활성화
if conda_env:
    print(f"🔹 Conda 가상환경 '{conda_env}' 활성화 중...")
    os.system(f"conda activate {conda_env}")
else:
    print("⚠️ .env 파일에서 Conda 환경 변수를 찾을 수 없습니다!")

# # Ngrok AuthToken 등록
# if ngrok_token:
#     print("🔹 Ngrok AuthToken 등록 중...")
#     os.system(f"ngrok authtoken {ngrok_token}")
# else:
#     print("⚠️ .env 파일에서 Ngrok AuthToken을 찾을 수 없습니다!")

print("🔹 FastAPI 서버 시작 중...")

# YOLO 모델 로드
model34 = YOLO("D:\\workspaces\\breadscanso\\localgpu\\yolov8s_34.pt").to("cuda")
print("✅ YOLO 모델 로드 완료!")

# FastAPI 앱 생성
app = FastAPI()
print("✅ FastAPI 앱 생성 완료!")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 🔹 CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (특정 도메인만 허용하고 싶다면 ["http://localhost:8000"] 등 설정 가능)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 헤더 허용
)


# YOLO 모델 로드
model34 = YOLO("D:\\workspaces\\breadscanso\\localgpu\\yolov8s_34.pt").to("cuda")


# 요청 데이터 모델 정의
class ImageData(BaseModel):
    image: str  # Base64로 인코딩된 이미지 데이터


@app.post("/predict/")
async def predict(data: ImageData):
    try:
        # Base64 디코딩
        image_bytes = base64.b64decode(data.image.split(",")[1])  # "data:image/jpeg;base64,..." 제거 후 디코딩
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # OpenCV 형식으로 변환
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # YOLO 모델 추론
        results = model34(image_np)

        # 결과 반환
        return {"prediction": results[0].tojson()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 오류: {str(e)}")


if __name__ == "__main__":
    # 🔹 ngrok을 먼저 실행하고 FastAPI 서버를 실행해야 함
    # public_url = ngrok.connect(8001)  # 8001번 포트를 외부 공개
    # print(f"🌍 외부에서 접근 가능: {public_url}")

    # 🔹 FastAPI 서버 실행
    print([route.path for route in app.routes])
    uvicorn.run(app, host="0.0.0.0", port=8001)




