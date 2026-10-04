import os
import tempfile
import cv2
import streamlit as st

from heatmap_core import Params
from video_processor import VideoProcessor

# Page configuration
st.set_page_config(
    page_title="Motion Heatmap Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colormaps
COLORMAPS = {
    "TURBO": cv2.COLORMAP_TURBO,
    "JET": cv2.COLORMAP_JET,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "HOT": cv2.COLORMAP_HOT,
    "OCEAN": cv2.COLORMAP_OCEAN
}

# Sidebar
st.sidebar.header("Processing Parameters")

# Time range
st.sidebar.subheader("Time Range")
start_time = st.sidebar.number_input("From (sec)", min_value=0.0, value=0.0, step=0.5)
end_time = st.sidebar.number_input("To (sec, 0 = full video)", min_value=0.0, value=0.0, step=0.5)

# Motion detection
st.sidebar.subheader("Motion Detection")
frame_step = st.sidebar.slider("Frame step", min_value=1, max_value=5, value=2)
var_threshold = st.sidebar.slider("Sensitivity threshold", min_value=4, max_value=100, value=16)

# Heatmap settings
st.sidebar.subheader("Heatmap Settings")
blur_pct = st.sidebar.slider("Blur radius (%)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
alpha = st.sidebar.slider("Overlay opacity", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
selected_colormap = st.sidebar.selectbox("Colormap", list(COLORMAPS.keys()), index=0)

# Exclusion zones
st.sidebar.subheader("Exclusion Zones")
ignore_top_pct = st.sidebar.slider("Ignore top area (%)", min_value=0.0, max_value=40.0, value=0.0, step=1.0)
ignore_bottom_pct = st.sidebar.slider("Ignore bottom area (%)", min_value=0.0, max_value=40.0, value=0.0, step=1.0)

# Main area
st.title("Motion Heatmap Generator")
st.write("Upload a video to analyze motion activity and generate a dynamic heatmap.")

uploaded_file = st.file_uploader("Select video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded input file
    tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile_in.write(uploaded_file.read())
    input_path = tfile_in.name
    tfile_in.close()

    # Read video metadata
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = total_frames / fps if fps > 0 else 0.0

    st.info(f"Video properties: {w}x{h} | FPS: {fps:.1f} | Duration: {duration:.1f} sec ({total_frames} frames)")

    # Run button
    if st.button("Start Processing", type="primary"):
        try:
            params = Params(
                start=start_time,
                end=end_time,
                frame_step=frame_step,
                var_threshold=var_threshold,
                blur_pct=blur_pct,
                alpha=alpha,
                colormap=COLORMAPS[selected_colormap],
                ignore_top_pct=ignore_top_pct,
                ignore_bottom_pct=ignore_bottom_pct
            )

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(pct: float, current_fps: float):
                progress_bar.progress(pct)
                status_text.text(f"Processing: {int(pct * 100)}% | Speed: {current_fps:.1f} FPS")

            processor = VideoProcessor(params, progress_callback=update_progress)

            out_dir = tempfile.mkdtemp()
            out_video_path = os.path.join(out_dir, "heatmap_video.mp4")
            out_image_path = os.path.join(out_dir, "heatmap_static.png")

            with st.spinner("Processing video..."):
                stats = processor.process(input_path, out_video_path, out_image_path)

            progress_bar.progress(1.0)
            status_text.text("Complete!")
            st.success(f"Processing completed in {stats['elapsed_sec']} sec ({stats['fps']} FPS, {stats['processed_frames']} frames)")

            # Read output results into memory
            with open(out_video_path, "rb") as vf:
                video_bytes = vf.read()
            with open(out_image_path, "rb") as img_f:
                image_bytes = img_f.read()

            st.session_state["results"] = {
                "video_bytes": video_bytes,
                "image_bytes": image_bytes,
                "stats": stats
            }

        except Exception as e:
            st.error(f"Processing error: {e}")

# Display results
if "results" in st.session_state:
    res = st.session_state["results"]
    st.divider()
    st.subheader("Processing Results")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Video with Heatmap Overlay**")
        st.video(res["video_bytes"])
        st.download_button(
            label="Download Video (MP4)",
            data=res["video_bytes"],
            file_name="motion_heatmap.mp4",
            mime="video/mp4"
        )

    with col2:
        st.markdown("**Final Static Heatmap**")
        st.image(res["image_bytes"], width="stretch")
        st.download_button(
            label="Download Heatmap (PNG)",
            data=res["image_bytes"],
            file_name="motion_heatmap.png",
            mime="image/png"
        )