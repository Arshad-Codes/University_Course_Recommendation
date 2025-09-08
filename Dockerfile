FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app
COPY backend/ ./backend/
WORKDIR /app/backend
ENV MODEL_DIR=/app/backend/model DATA_DIR=/app/backend/data
EXPOSE 8080
CMD ["uvicorn","src.app:app","--host","0.0.0.0","--port","8080"]
