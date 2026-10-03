import os
import time
import shutil
import subprocess
import tempfile
from typing import Callable, Optional, Dict, Any
import cv2
import numpy as np

from heatmap_core import Params, HeatmapEngine

class VideoProcessor:
    def __init__(self, params: Params, progress_callback: Optional[Callable[[float, float], None]] = None):
        self.params = params
        self.progress_callback = progress_callback
        self.engine = HeatmapEngine(params)

    def process(self, input_video_path: str, output_video_path: str, output_image_path: str) -> Dict[str, Any]:
        # Проверка существования файла
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Файл не найден: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Ошибка открытия видео: {input_video_path}")

        # Параметры видео
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if orig_w <= 0 or orig_h <= 0 or total_frames <= 0:
            cap.release()
            raise RuntimeError("Некорректные параметры видеофайла")

        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        # Масштабирование
        target_w = 640
        target_h = int(orig_h * (target_w / orig_w))
        if target_h % 2 != 0:
            target_h -= 1

        # Определение диапазона
        start_frame = int(self.params.start * fps)
        end_frame = int(self.params.end * fps) if self.params.end > 0 else total_frames
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))

        # Разогрев фона
        warmup_frames = 30
        if start_frame > 0:
            warmup_start = max(0, start_frame - warmup_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
            for _ in range(start_frame - warmup_start):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_res = cv2.resize(frame, (target_w, target_h))
                self.engine.process_frame(frame_res, is_warmup=True)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(min(warmup_frames, total_frames)):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_res = cv2.resize(frame, (target_w, target_h))
                self.engine.process_frame(frame_res, is_warmup=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Подготовка записи
        out_fps = max(1.0, fps / self.params.frame_step)
        has_ffmpeg = shutil.which("ffmpeg") is not None
        
        if has_ffmpeg:
            temp_avi_file = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
            raw_output_path = temp_avi_file.name
            temp_avi_file.close()
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        else:
            raw_output_path = output_video_path
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter(raw_output_path, fourcc, out_fps, (target_w, target_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Ошибка инициализации VideoWriter")

        total_steps = (end_frame - start_frame + self.params.frame_step - 1) // self.params.frame_step
        processed_count = 0
        first_frame = None
        start_time = time.time()

        try:
            curr_frame_idx = start_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)

            while curr_frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_res = cv2.resize(frame, (target_w, target_h))
                if first_frame is None:
                    first_frame = frame_res.copy()

                # Обработка и рендер
                rects = self.engine.process_frame(frame_res, is_warmup=False)
                rendered = self.engine.render(frame_res, rects=rects)
                writer.write(rendered)

                processed_count += 1

                # Пропуск кадров
                skip = self.params.frame_step - 1
                for _ in range(skip):
                    if not cap.grab():
                        break
                curr_frame_idx += self.params.frame_step

                # Прогресс
                if self.progress_callback and total_steps > 0:
                    elapsed = time.time() - start_time
                    curr_fps = processed_count / elapsed if elapsed > 0 else 0.0
                    progress_pct = min(1.0, processed_count / total_steps)
                    self.progress_callback(progress_pct, curr_fps)

        finally:
            cap.release()
            writer.release()

        # Итоговая статическая тепловая карта
        if first_frame is not None:
            final_heatmap = self.engine.render(first_frame, rects=None)
            cv2.imwrite(output_image_path, final_heatmap)

        # Конвертация в H.264
        if has_ffmpeg:
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", raw_output_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    output_video_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            finally:
                if os.path.exists(raw_output_path):
                    os.remove(raw_output_path)

        total_elapsed = time.time() - start_time
        avg_fps = processed_count / total_elapsed if total_elapsed > 0 else 0.0

        return {
            "processed_frames": processed_count,
            "elapsed_sec": round(total_elapsed, 2),
            "fps": round(avg_fps, 1),
            "video_path": output_video_path,
            "image_path": output_image_path
        }
