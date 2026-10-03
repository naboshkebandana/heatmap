import os
import tempfile
import cv2
import streamlit as st

from heatmap_core import Params
from video_processor import VideoProcessor

# Конфигурация страницы
st.set_page_config(
    page_title="Генератор тепловых карт движения",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Палитры
COLORMAPS = {
    "TURBO": cv2.COLORMAP_TURBO,
    "JET": cv2.COLORMAP_JET,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "HOT": cv2.COLORMAP_HOT,
    "OCEAN": cv2.COLORMAP_OCEAN
}

# Боковая панель
st.sidebar.header("Параметры обработки")

# Временной диапазон
st.sidebar.subheader("Временной диапазон")
start_time = st.sidebar.number_input("С (сек)", min_value=0.0, value=0.0, step=0.5)
end_time = st.sidebar.number_input("По (сек, 0 = всё видео)", min_value=0.0, value=0.0, step=0.5)

# Детекция движения
st.sidebar.subheader("Детекция движения")
frame_step = st.sidebar.slider("Шаг кадров", min_value=1, max_value=5, value=2)
var_threshold = st.sidebar.slider("Чувствительность", min_value=4, max_value=100, value=16)
min_area_pct = st.sidebar.slider("Мин. площадь объекта (%)", min_value=0.01, max_value=5.0, value=0.1, step=0.05)

# Тепловая карта
st.sidebar.subheader("Тепловая карта")
blur_pct = st.sidebar.slider("Сглаживание (%)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
alpha = st.sidebar.slider("Прозрачность", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
selected_colormap = st.sidebar.selectbox("Цветовая схема", list(COLORMAPS.keys()), index=0)

# Зоны исключения
st.sidebar.subheader("Зоны исключения")
ignore_top_pct = st.sidebar.slider("Игнорировать сверху (%)", min_value=0.0, max_value=40.0, value=0.0, step=1.0)
ignore_bottom_pct = st.sidebar.slider("Игнорировать снизу (%)", min_value=0.0, max_value=40.0, value=0.0, step=1.0)

# Основная область
st.title("Генератор тепловых карт движения")
st.write("Загрузите видео для детекции активности и формирования тепловой карты.")

uploaded_file = st.file_uploader("Выберите видеофайл", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Сохранение входного файла
    tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile_in.write(uploaded_file.read())
    input_path = tfile_in.name
    tfile_in.close()

    # Считывание информации о видео
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = total_frames / fps if fps > 0 else 0.0

    st.info(f"Параметры видео: {w}x{h} | FPS: {fps:.1f} | Длительность: {duration:.1f} сек ({total_frames} кадров)")

    # Кнопка запуска
    if st.button("Запустить обработку", type="primary"):
        try:
            params = Params(
                start=start_time,
                end=end_time,
                frame_step=frame_step,
                var_threshold=var_threshold,
                min_area_pct=min_area_pct,
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
                status_text.text(f"Обработка: {int(pct * 100)}% | Скорость: {current_fps:.1f} FPS")

            processor = VideoProcessor(params, progress_callback=update_progress)

            out_dir = tempfile.mkdtemp()
            out_video_path = os.path.join(out_dir, "heatmap_video.mp4")
            out_image_path = os.path.join(out_dir, "heatmap_static.png")

            with st.spinner("Идет обработка видео..."):
                stats = processor.process(input_path, out_video_path, out_image_path)

            progress_bar.progress(1.0)
            status_text.text("Готово!")
            st.success(f"Обработка завершена за {stats['elapsed_sec']} сек ({stats['fps']} FPS, {stats['processed_frames']} кадров)")

            # Чтение результатов в память
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
            st.error(f"Ошибка при обработке: {e}")

# Отображение результатов
if "results" in st.session_state:
    res = st.session_state["results"]
    st.divider()
    st.subheader("Результаты обработки")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Видео с тепловой картой**")
        st.video(res["video_bytes"])
        st.download_button(
            label="Скачать видео (MP4)",
            data=res["video_bytes"],
            file_name="motion_heatmap.mp4",
            mime="video/mp4"
        )

    with col2:
        st.markdown("**Финальная тепловая карта**")
        st.image(res["image_bytes"], use_container_width=True)
        st.download_button(
            label="Скачать карту (PNG)",
            data=res["image_bytes"],
            file_name="motion_heatmap.png",
            mime="image/png"
        )