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

# Ngrok AuthToken 등록
if ngrok_token:
    print("🔹 Ngrok AuthToken 등록 중...")
    os.system(f"ngrok authtoken {ngrok_token}")
else:
    print("⚠️ .env 파일에서 Ngrok AuthToken을 찾을 수 없습니다!")

print("🔹 FastAPI 서버 시작 중...")

# YOLO 모델 로드
model34 = YOLO("D:\\workspaces\\breadscanso\\localgpu\\yolov8s_34.pt").to("cuda")
print("✅ YOLO 모델 로드 완료!")

# FastAPI 앱 생성
app = FastAPI()
print("✅ FastAPI 앱 생성 완료!")

@app.get("/predict/")
async def predict(img_path: str):
    """
    로컬 GPU에서 YOLO 추론 실행
    """
    print(f"📷 이미지 경로: {img_path}")
    results = model34(img_path)
    print("✅ YOLO 추론 완료!")
    return {"prediction": results[0].tojson()}

if __name__ == "__main__":
    # 🔹 ngrok을 먼저 실행하고 FastAPI 서버를 실행해야 함
    public_url = ngrok.connect(8001)  # 8001번 포트를 외부 공개
    print(f"🌍 외부에서 접근 가능: {public_url}")

    # 🔹 FastAPI 서버 실행
    print([route.path for route in app.routes])
    uvicorn.run(app, host="0.0.0.0", port=8001)