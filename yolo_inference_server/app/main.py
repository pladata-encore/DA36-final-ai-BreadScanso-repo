from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os
from app.inference import predict

app = FastAPI()


class ImageData(BaseModel):
    file: str  # Base64 encoded image data


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/predict/")
async def upload_file(image: ImageData):
    """이미지 업로드 및 YOLO 추론 API"""
    try:
        # Base64 데이터 파싱
        image_data = image.file.split(",")[1] if "," in image.file else image.file
        image_bytes = base64.b64decode(image_data)

        # 파일 저장
        file_path = os.path.join(UPLOAD_DIR, "temp.jpg")
        with open(file_path, "wb") as buffer:
            buffer.write(image_bytes)

        try:
            # YOLO 추론 실행
            predictions = predict(file_path)
            return {"status": "success", "predictions": predictions}
        finally:
            # 임시 파일 삭제
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류가 발생했습니다: {str(e)}")