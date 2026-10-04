# Motion Heatmap Generator

**Motion Heatmap Generator** is a web application for automated video analysis and motion activity visualization using dynamic heatmaps.

## Features

- **Interactive Web Interface:** Upload and analyze videos directly in your browser.
- **Motion Detection:** Background subtraction using the MOG2 algorithm to identify moving objects.
- **Heatmap Accumulation:** Accumulates motion density over time and renders a color-coded activity heatmap.
- **Dual Export Options:** Produces both an H.264 encoded video with the heatmap overlay and a static summary image.
- **Format Support:** Supports MP4, AVI, MOV, and MKV video formats.
- **Flexible Time Range:** Analyze the entire video or specify custom start/end timestamps.
- **Customizable Parameters:** Fine-tune sensitivity thresholds, Gaussian blur radius, opacity, and colormaps (Turbo, Jet, Inferno, Viridis, Hot, Ocean).
- **Exclusion Zones:** Ignore motion in header or footer regions (e.g., timestamps, watermarks, UI elements).
- **Direct Downloads:** Export processed MP4 video and PNG static image directly from the browser.

## Limitations

- Maximum upload file size: 200 MB
- Video frames are scaled to 640px width for fast and consistent processing
- Does not track individual object identities (measures aggregate spatial activity)

## System Requirements

- Python 3.12+
- OS: Windows / Linux / macOS
- RAM: 4 GB+
- Free disk space: ~500 MB
- Docker & Docker Compose (optional, for containerized deployment)

## Getting Started

### Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

### Docker Deployment

To build and run the application container:
```bash
docker compose up --build
```
The application will be accessible at `http://localhost:8501`.

### How to Use

1. Open the application in your browser.
2. Click **"Select video file"** and choose a video file.
3. Configure detection and visualization parameters in the sidebar as needed.
4. Click **"Start Processing"**.
5. Once processing finishes, view the results and download the MP4 video and PNG heatmap image.

## Built With

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.12+ | Core programming language |
| Streamlit | 1.65.0 | Web application framework and UI |
| OpenCV (opencv-python-headless) | 5.0.0 | Video processing, MOG2 background subtraction, heatmap rendering |
| NumPy | 2.5.3 | Numerical array operations |
| imageio | 2.38+ | Output video stream writer |
| imageio-ffmpeg | 0.5+ | H.264 browser-compatible video encoding |
| Docker | — | Application containerization |
