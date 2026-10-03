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
    min_area_pct: float = 0.1
    blur_pct: float = 2.0
    alpha: float = 0.6
    colormap: int = cv2.COLORMAP_TURBO
    ignore_top_pct: float = 0.0
    ignore_bottom_pct: float = 0.0

    def __post_init__(self):
        if self.start < 0:
            raise ValueError("Параметр 'с' (start) должен быть >= 0")
        if self.end < 0:
            raise ValueError("Параметр 'по' (end) должен быть >= 0 (0 означает до конца)")
        if self.start >= self.end and self.end != 0:
            raise ValueError("Параметр 'с' должен быть меньше 'по'")
        if not 1 <= self.frame_step <= 5:
            raise ValueError("Шаг кадров (frame_step) должен быть от 1 до 5")
        if not 4 <= self.var_threshold <= 100:
            raise ValueError("Чувствительность (var_threshold) должна быть от 4 до 100")
        if not 0.01 <= self.min_area_pct <= 5.0:
            raise ValueError("Порог срабатывания (min_area_pct) должен быть от 0.01 до 5.0")
        if not 0.5 <= self.blur_pct <= 8.0:
            raise ValueError("Сглаживание карты (blur_pct) должно быть от 0.5 до 8.0")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("Прозрачность (alpha) должна быть от 0.0 до 1.0")
        if not 0.0 <= self.ignore_top_pct <= 40.0:
            raise ValueError("Игнорирование верха (ignore_top_pct) от 0 до 40")
        if not 0.0 <= self.ignore_bottom_pct <= 40.0:
            raise ValueError("Игнорирование низа (ignore_bottom_pct) от 0 до 40")

class HeatmapEngine:
    def __init__(self, params: Params):
        self.params = params
        self.mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=self.params.var_threshold, detectShadows=True)
        self.accumulator = None
        
        # Морфология
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
    def process_frame(self, frame: np.ndarray, is_warmup: bool = False) -> List[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        if self.accumulator is None:
            self.accumulator = np.zeros((h, w), dtype=np.float32)
            
        # Сглаживание кадра
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Вычитание фона
        fgmask = self.mog2.apply(blurred)
        
        # Бинаризация
        _, fgmask = cv2.threshold(fgmask, 199, 255, cv2.THRESH_BINARY)
        
        # Зоны исключения
        if self.params.ignore_top_pct > 0:
            top_h = int(h * (self.params.ignore_top_pct / 100.0))
            fgmask[:top_h, :] = 0
        if self.params.ignore_bottom_pct > 0:
            bottom_h = int(h * (self.params.ignore_bottom_pct / 100.0))
            fgmask[h - bottom_h:, :] = 0
            
        # Фильтрация шума
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel_open)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.kernel_close)
        
        # Защита от вспышек
        fg_ratio = np.count_nonzero(fgmask) / (w * h)
        if fg_ratio > 0.5:
            return []
            
        # Поиск контуров
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = (w * h) * (self.params.min_area_pct / 100.0)
        
        clean_mask = np.zeros((h, w), dtype=np.uint8)
        rects = []
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)
                rects.append(cv2.boundingRect(contour))
                
        if not is_warmup:
            self.accumulator += (clean_mask / 255.0)
            
        return rects

    def render(self, frame: np.ndarray, rects: List[Tuple[int, int, int, int]] = None, bg_frame: np.ndarray = None) -> np.ndarray:
        if self.accumulator is None:
            return frame.copy()
            
        h, w = frame.shape[:2]
        
        # Сглаживание карты
        blur_radius = int(w * (self.params.blur_pct / 100.0))
        if blur_radius % 2 == 0:
            blur_radius += 1
        blur_radius = max(1, blur_radius)
        
        heatmap_blurred = cv2.GaussianBlur(self.accumulator, (blur_radius, blur_radius), 0)
        
        # Нормализация
        positives = heatmap_blurred[heatmap_blurred > 0]
        if len(positives) > 0:
            p99 = np.percentile(positives, 99)
            if p99 > 0:
                normalized = heatmap_blurred / p99
            else:
                normalized = heatmap_blurred
        else:
            normalized = heatmap_blurred
            
        # Масштабирование
        normalized = np.clip(normalized, 0, 1)
        normalized = np.sqrt(normalized)
        
        # Наложение цвета
        heatmap_8u = (normalized * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_8u, self.params.colormap)
        
        # Смешивание слоев
        a = (normalized * self.params.alpha)[:, :, np.newaxis]
        
        base_frame = bg_frame if bg_frame is not None else frame
        
        out = (a * colored_heatmap + (1 - a) * base_frame).astype(np.uint8)
        
        # Отрисовка рамок
        if rects:
            for x, y, rw, rh in rects:
                cv2.rectangle(out, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
                
        return out
