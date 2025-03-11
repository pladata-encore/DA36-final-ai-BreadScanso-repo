# 베이스 이미지 (Python 3.9)
FROM python:3.9

# 작업 디렉토리 설정
WORKDIR /app

# 필수 라이브러리 설치 (libGL.so.1 포함)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# 필요한 파일 복사
COPY requirements.txt .
COPY app/ app/
COPY model/ model/

# 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]