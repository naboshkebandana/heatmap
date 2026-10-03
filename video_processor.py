import os
import time
from typing import Callable, Optional, Dict, Any
import cv2
import numpy as np
import imageio

from heatmap_core import Params, HeatmapEngine

class VideoProcessor:
    def __init__(self, params: Params, progress_callback: Optional[Callable[[float, float], None]] = None):
        self.params = params
        self.progress_callback = progress_callback
        self.engine = HeatmapEngine(params)

    def process(self, input_video_path: str, output_video_path: str, output_image_path: str) -> Dict[str, Any]:
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

        # Масштабирование до 640px по ширине
        target_w = 640
        target_h = int(orig_h * (target_w / orig_w))
        target_h -= target_h % 2  # чётная высота для H.264

        # Диапазон кадров
        start_frame = int(self.params.start * fps)
        end_frame = int(self.params.end * fps) if self.params.end > 0 else total_frames
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))

        # Разогрев фона (30 кадров до начала)
        warmup_count = 30
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - warmup_count))
            for _ in range(min(warmup_count, start_frame)):
                ret, frame = cap.read()
                if not ret:
                    break
                self.engine.process_frame(cv2.resize(frame, (target_w, target_h)), is_warmup=True)
        else:
            for _ in range(min(warmup_count, total_frames)):
                ret, frame = cap.read()
                if not ret:
                    break
                self.engine.process_frame(cv2.resize(frame, (target_w, target_h)), is_warmup=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Запись через imageio-ffmpeg (H.264, браузер-совместимый)
        out_fps = max(1.0, fps / self.params.frame_step)
        writer = imageio.get_writer(
            output_video_path,
            fps=out_fps,
            codec="libx264",
            quality=None,
            pixelformat="yuv420p",
            output_params=["-movflags", "+faststart"],
        )

        total_steps = (end_frame - start_frame + self.params.frame_step - 1) // self.params.frame_step
        processed_count = 0
        first_frame = None
        start_time = time.time()

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            curr_idx = start_frame

            while curr_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_res = cv2.resize(frame, (target_w, target_h))
                if first_frame is None:
                    first_frame = frame_res.copy()

                # Обработка и рендер
                self.engine.process_frame(frame_res)
                rendered = self.engine.render(frame_res)

                # imageio ожидает RGB
                writer.append_data(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
                processed_count += 1

                # Пропуск кадров
                for _ in range(self.params.frame_step - 1):
                    if not cap.grab():
                        break
                curr_idx += self.params.frame_step

                # Прогресс
                if self.progress_callback and total_steps > 0:
                    elapsed = time.time() - start_time
                    curr_fps = processed_count / elapsed if elapsed > 0 else 0.0
                    self.progress_callback(min(1.0, processed_count / total_steps), curr_fps)

        finally:
            cap.release()
            writer.close()

        # Статическая тепловая карта
        if first_frame is not None:
            final_heatmap = self.engine.render(first_frame)
            cv2.imwrite(output_image_path, final_heatmap)

        total_elapsed = time.time() - start_time
        avg_fps = processed_count / total_elapsed if total_elapsed > 0 else 0.0

        return {
            "processed_frames": processed_count,
            "elapsed_sec": round(total_elapsed, 2),
            "fps": round(avg_fps, 1),
            "video_path": output_video_path,
            "image_path": output_image_path,
        }
