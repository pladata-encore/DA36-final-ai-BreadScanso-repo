import torch
from ultralytics import YOLO
from fastapi import FastAPI
import uvicorn
from pyngrok import ngrok

print("🔹 FastAPI 서버 시작 중...")

# YOLO 모델 로드
model34 = YOLO("D:\\workspaces\\breadscanso\\localgpu\\yolov8s_34.pt").to("cuda")
print("✅ YOLO 모델 로드 완료!")

# FastAPI 앱 생성
app = FastAPI()
print("✅ FastAPI 앱 생성 완료!")

ngrok.set_auth_token("2syUqUKIbaogeANfgPIonaerZK6_4ZcqtqZsvaQCes9JKKag7")


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

