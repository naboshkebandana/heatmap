import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Params:
    start: float = 0.0
    end: float = 0.0
    frame_step: int = 2
    var_threshold: int = 16
    blur_pct: float = 2.0
    alpha: float = 0.6
    colormap: int = cv2.COLORMAP_TURBO
    ignore_top_pct: float = 0.0
    ignore_bottom_pct: float = 0.0

class HeatmapEngine:
    def __init__(self, params: Params):
        self.params = params
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=self.params.var_threshold,
            detectShadows=False
        )
        self.accumulator = None

    def process_frame(self, frame: np.ndarray, is_warmup: bool = False) -> None:
        """Обработать кадр и накопить движение в аккумулятор."""
        h, w = frame.shape[:2]
        if self.accumulator is None:
            self.accumulator = np.zeros((h, w), dtype=np.float32)

        # Вычитание фона
        fgmask = self.mog2.apply(frame)

        # Бинаризация (порог 200 убирает тени и шум)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Зоны исключения
        if self.params.ignore_top_pct > 0:
            top_h = int(h * self.params.ignore_top_pct / 100.0)
            fgmask[:top_h, :] = 0
        if self.params.ignore_bottom_pct > 0:
            bot_h = int(h * self.params.ignore_bottom_pct / 100.0)
            fgmask[h - bot_h:, :] = 0

        # Лёгкая морфология — убрать мелкий шум
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # Накопление
        if not is_warmup:
            self.accumulator += (fgmask / 255.0)

    def render(self, frame: np.ndarray, bg_frame: np.ndarray = None) -> np.ndarray:
        """Отрисовать тепловую карту поверх кадра."""
        if self.accumulator is None:
            return frame.copy()

        h, w = frame.shape[:2]

        # Сглаживание карты
        blur_r = int(w * self.params.blur_pct / 100.0)
        blur_r = max(1, blur_r | 1)  # нечётное число >= 1
        heatmap = cv2.GaussianBlur(self.accumulator, (blur_r, blur_r), 0)

        # Нормализация 0..1
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap = np.clip(heatmap, 0, 1)

        # Раскрашивание
        colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), self.params.colormap)

        # Наложение на кадр
        a = (heatmap * self.params.alpha)[:, :, np.newaxis]
        base = bg_frame if bg_frame is not None else frame
        out = (a * colored + (1 - a) * base).astype(np.uint8)

        return out
