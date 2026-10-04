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
        """Process frame and accumulate motion into the accumulator."""
        h, w = frame.shape[:2]
        if self.accumulator is None:
            self.accumulator = np.zeros((h, w), dtype=np.float32)

        # Background subtraction
        fgmask = self.mog2.apply(frame)

        # Binarization (threshold 200 removes shadows and noise)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Exclusion zones
        if self.params.ignore_top_pct > 0:
            top_h = int(h * self.params.ignore_top_pct / 100.0)
            fgmask[:top_h, :] = 0
        if self.params.ignore_bottom_pct > 0:
            bot_h = int(h * self.params.ignore_bottom_pct / 100.0)
            fgmask[h - bot_h:, :] = 0

        # Light morphology to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # Accumulation
        if not is_warmup:
            self.accumulator += (fgmask / 255.0)

    def render(self, frame: np.ndarray, bg_frame: np.ndarray = None) -> np.ndarray:
        """Render heatmap overlay on top of the frame."""
        if self.accumulator is None:
            return frame.copy()

        h, w = frame.shape[:2]

        # Heatmap smoothing
        blur_r = int(w * self.params.blur_pct / 100.0)
        blur_r = max(1, blur_r | 1)  # odd number >= 1
        heatmap = cv2.GaussianBlur(self.accumulator, (blur_r, blur_r), 0)

        # Normalize to [0, 1]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap = np.clip(heatmap, 0, 1)

        # Apply colormap
        colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), self.params.colormap)

        # Overlay onto frame
        a = (heatmap * self.params.alpha)[:, :, np.newaxis]
        base = bg_frame if bg_frame is not None else frame
        out = (a * colored + (1 - a) * base).astype(np.uint8)

        return out
