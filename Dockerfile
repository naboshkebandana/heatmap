FROM python:3.12-slim-bookworm

# Системные зависимости: ffmpeg для перекодирования видео в H.264
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости (кэшируется отдельным слоем)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Проверка импортов — сборка упадёт сразу, если чего-то не хватает
RUN python -c "import cv2, numpy, streamlit"

# Код приложения
COPY .streamlit/ .streamlit/
COPY app.py .
COPY heatmap_core.py .
COPY video_processor.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
