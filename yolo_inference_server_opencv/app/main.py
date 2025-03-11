from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
from app.inference import predict

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/predict/")
async def upload_file(file: UploadFile = File(...)):
    """
    이미지 업로드 및 YOLO 추론 API
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # 파일 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            # YOLO 추론 실행 (동기 함수)
            results = predict(file_path)
            return {"status": "success", "predictions": results}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"추론 중 오류 발생: {str(e)}")

        finally:
            # 임시 파일 삭제
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류 발생: {str(e)}")