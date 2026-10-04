FROM python:3.12-slim-bookworm

# System dependencies: ffmpeg for video transcoding to H.264
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (cached as a separate layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify imports — build fails immediately if anything is missing
RUN python -c "import cv2, numpy, streamlit, imageio"

# Application code
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
